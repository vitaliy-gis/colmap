#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace colmap;
using PointInImage = std::pair<image_t, point3D_t>;
using Observation = std::tuple<image_t, point2D_t, point3D_t>;
using ObservationSet = std::set<Observation>;

ObservationSet ReadObservations(const std::filesystem::path& path,
                                const Reconstruction& model) {
  std::ifstream in(path);
  THROW_CHECK(in.is_open());
  ObservationSet observations;
  image_t image;
  point2D_t index;
  point3D_t point;
  while (in >> image >> index >> point) {
    const Image& row = model.Image(image);
    THROW_CHECK_EQ(row.Point2D(index).point3D_id, point);
    THROW_CHECK(observations.emplace(image, index, point).second);
  }
  THROW_CHECK(in.eof());
  THROW_CHECK_EQ(observations.size(), 6974);
  return observations;
}

void WriteGauge(const std::filesystem::path& path,
                Reconstruction& model,
                const ceres::Problem& problem) {
  std::ofstream out(path);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << "image_id,constant,tangent_size,fixed_translation_coordinate\n";
  size_t constant = 0, partial = 0;
  for (const image_t id : model.RegImageIds()) {
    const auto& params = model.Image(id).FramePtr()->RigFromWorld().params;
    THROW_CHECK(problem.HasParameterBlock(params.data()));
    const bool fixed = problem.IsParameterBlockConstant(params.data());
    const int tangent = problem.ParameterBlockTangentSize(params.data());
    int fixed_axis = -1;
    if (!fixed && tangent != 6) {
      THROW_CHECK_EQ(tangent, 5);
      const ceres::Manifold* manifold = problem.GetManifold(params.data());
      THROW_CHECK_NOTNULL(manifold);
      Eigen::Matrix<double, 7, 5, Eigen::RowMajor> jacobian;
      THROW_CHECK(manifold->PlusJacobian(params.data(), jacobian.data()));
      for (int axis = 0; axis < 3; ++axis) {
        if (jacobian.row(axis + 4).isZero(0)) {
          THROW_CHECK_EQ(fixed_axis, -1);
          fixed_axis = axis;
        }
      }
      THROW_CHECK_GE(fixed_axis, 0);
      ++partial;
    }
    constant += fixed;
    out << id << ',' << fixed << ',' << tangent << ',' << fixed_axis << '\n';
  }
  THROW_CHECK_EQ(constant, 1);
  THROW_CHECK_EQ(partial, 1);
}

void ApplyTreatment(const ObservationSet& target,
                    const std::filesystem::path& out,
                    const bool remove,
                    Reconstruction& model,
                    ceres::Problem& problem) {
  std::unordered_map<const double*, image_t> poses;
  for (const image_t id : model.RegImageIds()) {
    const auto& params = model.Image(id).FramePtr()->RigFromWorld().params;
    THROW_CHECK(poses.emplace(params.data(), id).second);
  }
  std::unordered_map<const double*, point3D_t> points;
  for (const auto& [id, point] : model.Points3D()) {
    THROW_CHECK(points.emplace(point.xyz.data(), id).second);
  }
  THROW_CHECK_EQ(model.Cameras().size(), 1);
  const double* camera = model.Cameras().begin()->second.params.data();
  THROW_CHECK(problem.IsParameterBlockConstant(camera));
  std::vector<ceres::ResidualBlockId> blocks;
  problem.GetResidualBlocks(&blocks);
  THROW_CHECK_EQ(blocks.size(), 483121);
  std::map<PointInImage, std::vector<point2D_t>> candidates;
  for (const auto& [image, index, point] : target) {
    candidates.try_emplace({image, point});
  }
  for (const image_t id : model.RegImageIds()) {
    const auto& image = model.Image(id);
    for (point2D_t index = 0; index < image.NumPoints2D(); ++index) {
      const auto candidate = candidates.find({id, image.Point2D(index).point3D_id});
      if (candidate != candidates.end()) candidate->second.push_back(index);
    }
  }
  size_t expected_checked = 0, ambiguous_pairs = 0;
  for (const auto& [key, indices] : candidates) {
    expected_checked += indices.size();
    ambiguous_pairs += indices.size() > 1;
  }
  ObservationSet checked;
  std::map<Observation, ceres::ResidualBlockId> matched;
  std::map<point3D_t, image_t> active_points;
  std::map<image_t, image_t> components;
  for (const auto& [address, id] : poses) components[id] = id;
  auto root = [&components](image_t id) {
    while (components.at(id) != id) id = components.at(id);
    return id;
  };
  std::set<image_t> active_images;
  std::map<point3D_t, std::pair<bool, const ceres::Manifold*>> parameterization;
  for (const auto& [address, id] : points) {
    THROW_CHECK(problem.HasParameterBlock(address));
    parameterization[id] = {problem.IsParameterBlockConstant(address), problem.GetManifold(address)};
  }
  std::vector<double*> params;
  for (const auto block : blocks) {
    problem.GetParameterBlocksForResidualBlock(block, &params);
    THROW_CHECK_EQ(params.size(), 3);
    THROW_CHECK(params[2] == camera);
    const PointInImage key(poses.at(params[1]), points.at(params[0]));
    bool is_target = false;
    if (const auto candidate = candidates.find(key); candidate != candidates.end()) {
      const auto& image = model.Image(key.first);
      const auto projected = image.CameraPtr()->ImgFromCam(
          image.CamFromWorld() * model.Point3D(key.second).xyz);
      THROW_CHECK(projected.has_value());
      const std::array<const double*, 3> inputs{params[0], params[1], params[2]};
      Eigen::Vector2d residual;
      THROW_CHECK(problem.GetCostFunctionForResidualBlock(block)->Evaluate(
          inputs.data(), residual.data(), nullptr));
      size_t count = 0;
      point2D_t index = 0;
      for (const point2D_t possible : candidate->second) {
        if ((*projected - image.Point2D(possible).xy - residual).cwiseAbs().maxCoeff() <= 1e-8) {
          ++count;
          index = possible;
        }
      }
      THROW_CHECK_EQ(count, 1);
      const Observation obs(key.first, index, key.second);
      THROW_CHECK(checked.insert(obs).second);
      is_target = target.count(obs);
      if (is_target) THROW_CHECK(matched.emplace(obs, block).second);
    }
    if (!remove || !is_target) {
      active_images.insert(key.first);
      auto [entry, added] = active_points.emplace(key.second, key.first);
      if (!added) components[root(key.first)] = root(entry->second);
    }
  }
  THROW_CHECK_EQ(checked.size(), expected_checked);
  THROW_CHECK_EQ(matched.size(), target.size());
  std::ofstream log(out / "matched_observations.txt");
  log.exceptions(std::ios::failbit | std::ios::badbit);
  for (const auto& [obs, block] : matched) {
    const auto [image, index, point] = obs;
    log << image << ' ' << index << ' ' << point << '\n';
    if (remove) problem.RemoveResidualBlock(block);
  }
  THROW_CHECK_EQ(problem.NumResidualBlocks(), 483121 - (remove ? 6974 : 0));
  THROW_CHECK_EQ(problem.NumResiduals(), 2 * problem.NumResidualBlocks());
  for (const auto& [address, id] : points) {
    THROW_CHECK(problem.HasParameterBlock(address));
    const auto [constant, manifold] = parameterization.at(id);
    THROW_CHECK_EQ(problem.IsParameterBlockConstant(address), constant);
    THROW_CHECK(problem.GetManifold(address) == manifold);
  }
  std::set<image_t> roots;
  for (const image_t id : active_images) roots.insert(root(id));
  std::ofstream support(out / "active_factors.json");
  support.exceptions(std::ios::failbit | std::ios::badbit);
  support << "{\"images\":" << active_images.size() << ",\"points\":" << active_points.size()
          << ",\"inactive_points\":" << points.size() - active_points.size()
          << ",\"components\":" << roots.size()
          << ",\"pixel_factor_identities_checked\":" << checked.size()
          << ",\"image_point_pairs_with_multiple_observations\":" << ambiguous_pairs
          << ",\"point_parameterization_unchanged\":true}\n";
}
}  // namespace

int main(int argc, char** argv) {
  using namespace colmap;
  InitializeGlog(argv);
  THROW_CHECK_GE(argc, 6);
  const std::string branch = argv[1];
  THROW_CHECK(branch == "A" || branch == "B");
  const std::filesystem::path input(argv[2]), out(argv[3]);
  THROW_CHECK(std::getenv("COLMAP_PHYSICAL_RIG_CONFIG") == nullptr);
  std::vector<char*> option_argv{argv[0]};
  option_argv.insert(option_argv.end(), argv + 6, argv + argc);
  const auto option_argc = option_argv.size();
  option_argv.push_back(nullptr);
  OptionManager options;
  options.AddMapperOptions();
  THROW_CHECK(options.Parse(option_argc, option_argv.data()));
  const auto ba = options.mapper->GlobalBundleAdjustment();
  THROW_CHECK_NOTNULL(ba.ceres);
  THROW_CHECK(!ba.ceres->use_gpu && !ba.refine_focal_length &&
              !ba.refine_principal_point && !ba.refine_extra_params &&
              !ba.refine_sensor_from_rig);
  THROW_CHECK_EQ(ba.ceres->solver_options.max_num_iterations, 50);
  THROW_CHECK(ba.ceres->loss_function_type ==
              CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL);
  Reconstruction model;
  model.Read(input);
  THROW_CHECK(model.IsValid());
  THROW_CHECK_EQ(model.NumRegFrames(), 665);
  THROW_CHECK_EQ(model.ComputeNumObservations(), 483121);
  const ObservationSet target = ReadObservations(argv[4], model);
  std::ifstream order(argv[5]);
  THROW_CHECK(order.is_open());
  BundleAdjustmentConfig config;
  std::set<image_t> configured;
  image_t id;
  while (order >> id) {
    THROW_CHECK(configured.insert(id).second);
    THROW_CHECK(model.Image(id).IsRefInFrame() && model.Image(id).HasPose());
    config.AddImage(id);
  }
  THROW_CHECK(order.eof());
  THROW_CHECK_EQ(configured.size(), 665);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  auto adjuster = CreateDefaultBundleAdjuster(ba, config, model);
  auto* ceres_adjuster = dynamic_cast<CeresBundleAdjuster*>(adjuster.get());
  THROW_CHECK_NOTNULL(ceres_adjuster);
  auto& problem = *ceres_adjuster->Problem();
  WriteGauge(out / "gauge_full.csv", model, problem);
  ApplyTreatment(target, out, branch == "B", model, problem);
  WriteGauge(out / "gauge_used.csv", model, problem);
  const auto summary = adjuster->Solve();
  const auto* result = dynamic_cast<const CeresBundleAdjustmentSummary*>(summary.get());
  THROW_CHECK_NOTNULL(result);
  const auto& s = result->ceres_summary;
  std::ofstream report(out / "solver.txt");
  report.exceptions(std::ios::failbit | std::ios::badbit);
  report << s.FullReport() << '\n';
  std::ofstream values(out / "solve.json");
  values.exceptions(std::ios::failbit | std::ios::badbit);
  values << std::setprecision(17)
         << "{\"initial_cost\":" << s.initial_cost
         << ",\"final_cost\":" << s.final_cost
         << ",\"solution_usable\":" << (summary->IsSolutionUsable() ? "true" : "false")
         << ",\"termination_type\":" << static_cast<int>(s.termination_type)
         << ",\"threads\":" << s.num_threads_used
         << ",\"residual_blocks\":" << s.num_residual_blocks
         << ",\"residuals\":" << s.num_residuals
         << ",\"parameter_blocks\":" << s.num_parameter_blocks
         << ",\"reduced_parameter_blocks\":" << s.num_parameter_blocks_reduced << "}\n";
  model.Write(out / "model");
}
