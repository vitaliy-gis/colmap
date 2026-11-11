// One local bundle adjustment, with a replay on the same Ceres Problem.
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace colmap {
namespace bootstrap_probe {

using TrackKey = std::pair<point3D_t, image_t>;

struct State {
  std::map<TrackKey, point2D_t> registered_observations;
  bool ran = false;
};

inline State& Get() {
  static State state;
  return state;
}

inline void RecordRegistered(const Reconstruction& rec) {
  if (rec.NumRegFrames() != 3 || Get().ran) {
    return;
  }
  auto& observations = Get().registered_observations;
  for (const auto& [point_id, point] : rec.Points3D()) {
    for (const auto& observation : point.track.Elements()) {
      const bool inserted = observations.emplace(
          TrackKey{point_id, observation.image_id}, observation.point2D_idx).second;
      THROW_CHECK(inserted);
    }
  }
  THROW_CHECK(!observations.empty());
}

inline std::shared_ptr<BundleAdjustmentSummary> Solve(
    BundleAdjuster& adjuster, Reconstruction& rec) {
  State& state = Get();
  if (rec.NumRegFrames() != 3 || state.ran ||
      state.registered_observations.empty()) {
    return adjuster.Solve();
  }
  state.ran = true;
  auto* ceres_adjuster = dynamic_cast<CeresBundleAdjuster*>(&adjuster);
  THROW_CHECK_NOTNULL(ceres_adjuster);
  ceres::Problem& problem = *ceres_adjuster->Problem();

  std::unordered_map<const double*, point3D_t> points;
  std::unordered_map<const double*, frame_t> frames;
  std::unordered_map<const double*, camera_t> cameras;
  std::map<std::pair<frame_t, camera_t>, image_t> images;
  for (const auto& [point_id, point] : rec.Points3D()) {
    points.emplace(point.xyz.data(), point_id);
  }
  for (const image_t image_id : rec.RegImageIds()) {
    const Image& image = rec.Image(image_id);
    const Frame& frame = *image.FramePtr();
    frames.emplace(frame.RigFromWorld().params.data(), frame.FrameId());
    cameras.emplace(image.CameraPtr()->params.data(), image.CameraId());
    images.emplace(std::make_pair(frame.FrameId(), image.CameraId()), image_id);
  }

  std::vector<ceres::ResidualBlockId> blocks;
  std::vector<ceres::ResidualBlockId> old_blocks, protected_blocks, new_blocks;
  problem.GetResidualBlocks(&blocks);
  const auto factors_path = diagnostic_trace::NextFile("probe_factors", "csv");
  std::ofstream factors(factors_path);
  factors << "point_id,image_id,point2D_id,frame_id,old,constant_point\n";
  for (const auto block : blocks) {
    std::vector<double*> parameters;
    problem.GetParameterBlocksForResidualBlock(block, &parameters);
    THROW_CHECK_EQ(parameters.size(), 3);
    const auto point_id = points.at(parameters[0]);
    const auto frame_id = frames.at(parameters[1]);
    const auto image_id = images.at({frame_id, cameras.at(parameters[2])});
    point2D_t point2D_idx = kInvalidPoint2DIdx;
    for (const auto& observation : rec.Point3D(point_id).track.Elements()) {
      if (observation.image_id == image_id) {
        THROW_CHECK_EQ(point2D_idx, kInvalidPoint2DIdx);
        point2D_idx = observation.point2D_idx;
      }
    }
    THROW_CHECK_NE(point2D_idx, kInvalidPoint2DIdx);
    const auto old = state.registered_observations.find({point_id, image_id});
    const bool was_registered = old != state.registered_observations.end() &&
                                old->second == point2D_idx;
    const bool constant_point = problem.IsParameterBlockConstant(parameters[0]);
    factors << point_id << ',' << image_id << ',' << point2D_idx << ','
            << frame_id << ',' << was_registered << ',' << constant_point << '\n';
    if (was_registered) {
      old_blocks.push_back(block);
    } else if (constant_point) {
      protected_blocks.push_back(block);
    } else {
      new_blocks.push_back(block);
    }
  }
  factors.close();
  THROW_CHECK(!old_blocks.empty());
  THROW_CHECK(!new_blocks.empty());

  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  using Values = std::vector<std::vector<double>>;
  const auto snapshot = [&]() {
    Values values;
    values.reserve(parameter_blocks.size());
    for (double* parameter : parameter_blocks) {
      values.emplace_back(parameter,
                          parameter + problem.ParameterBlockSize(parameter));
    }
    return values;
  };
  const auto restore = [&](const Values& values) {
    for (size_t i = 0; i < values.size(); ++i) {
      std::copy(values[i].begin(), values[i].end(), parameter_blocks[i]);
    }
  };
  const auto report_path = diagnostic_trace::NextFile("probe_costs", "csv");
  std::ofstream costs(report_path);
  costs << std::setprecision(17) << "state,group,residual_blocks,robust_cost\n";
  const auto cost = [&](const char* label, const char* group,
                        const std::vector<ceres::ResidualBlockId>& selected) {
    if (selected.empty()) {
      return;
    }
    ceres::Problem::EvaluateOptions opts;
    opts.residual_blocks = selected;
    double value;
    THROW_CHECK(problem.Evaluate(opts, &value, nullptr, nullptr, nullptr));
    costs << label << ',' << group << ',' << selected.size() << ',' << value << '\n';
  };
  const auto common_costs = [&](const char* label) {
    cost(label, "registered", old_blocks);
    cost(label, "new_constant_point", protected_blocks);
  };

  const Values initial = snapshot();
  diagnostic_trace::Capture("probe_before_solve", rec);
  common_costs("before");
  cost("before", "new_nonconstant_point", new_blocks);
  const auto native = adjuster.Solve();
  const Values after_a = snapshot();
  diagnostic_trace::Capture("probe_after_a", rec);
  common_costs("after_a");
  cost("after_a", "new_nonconstant_point", new_blocks);

  restore(initial);
  for (const auto block : new_blocks) {
    problem.RemoveResidualBlock(block);
  }
  const auto test = adjuster.Solve();
  diagnostic_trace::Capture("probe_after_b", rec);
  common_costs("after_b");
  restore(after_a);
  std::ofstream solves(diagnostic_trace::NextFile("probe_solves", "txt"));
  solves << "A " << native->BriefReport() << "\nB " << test->BriefReport() << '\n';
  diagnostic_trace::EndBootstrapProbe();
  return native;
}

}  // namespace bootstrap_probe
}  // namespace colmap
