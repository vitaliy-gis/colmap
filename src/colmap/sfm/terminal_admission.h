#pragma once

#include "terminal_trace.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/sfm/observation_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace colmap::terminal_admission {

struct Snapshot {
  std::map<image_t, std::vector<point3D_t>> assignments;
  size_t assigned = 0;
};
constexpr double kMaxErrorPixels = 4.0;

inline Snapshot Before(const Reconstruction& reconstruction) {
  const auto& state = terminal_trace::Get();
  if (!state.active || reconstruction.NumRegFrames() != 665 ||
      (state.block != "exhaustion_retry" && state.block != "final_refinement") ||
      std::getenv("COLMAP_TERMINAL_ADMISSION") == nullptr) {
    return {};
  }
  static thread_local bool entered_exhaustion = false;
  if (state.block == "exhaustion_retry") entered_exhaustion = true;
  if (!entered_exhaustion) return {};
  Snapshot result;
  for (const image_t id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(id);
    auto& occupied = result.assignments[id];
    occupied.reserve(image.NumPoints2D());
    for (const auto& point : image.Points2D()) {
      occupied.push_back(point.point3D_id);
      result.assigned += point.HasPoint3D();
    }
  }
  return result;
}

struct Decision {
  TrackElement observation;
  double depth;
  double error;
  const char* rejection;
};

struct PointPlan {
  point3D_t point;
  size_t protected_count = 0;
  size_t kept = 0;
  std::vector<Decision> new_observations;
};

inline Decision Evaluate(const Reconstruction& reconstruction,
                         const TrackElement& observation,
                         const Eigen::Vector3d& xyz) {
  const auto& image = reconstruction.Image(observation.image_id);
  const auto position = image.CamFromWorld() * xyz;
  const double depth = position.z();
  if (!std::isfinite(depth) || depth <= 0) {
    return {observation, depth, std::numeric_limits<double>::quiet_NaN(), "nonpositive_depth"};
  }
  const auto projection = image.CameraPtr()->ImgFromCam(position);
  const double error = projection.has_value()
                           ? (*projection - image.Point2D(observation.point2D_idx).xy).norm()
                           : std::numeric_limits<double>::infinity();
  return {observation, depth, error,
          std::isfinite(error) && error <= kMaxErrorPixels ? nullptr : "over_4px"};
}

inline size_t Apply(const char* operation,
                    const Snapshot& before,
                    Reconstruction& reconstruction,
                    ObservationManager& observations,
                    IncrementalTriangulator& triangulator) {
  if (before.assignments.empty()) return 0;
  auto& state = terminal_trace::Get();
  static thread_local size_t serial = 0;
  const size_t event = serial++;
  const size_t count_before_gate = reconstruction.ComputeNumObservations();
  std::vector<PointPlan> plans;
  std::vector<std::tuple<image_t, point2D_t, point3D_t, point3D_t>> merges;
  size_t cohort = 0, accepted = 0, rejected = 0, rejected_depth = 0;
  size_t rejected_error = 0, rejected_support = 0, accepted_new_points = 0;
  size_t accepted_extensions = 0, accepted_new_points_observations = 0;
  size_t protected_observations = 0;
  for (const auto& [pid, point] : reconstruction.Points3D()) {
    PointPlan plan;
    plan.point = pid;
    for (const auto& track : point.track.Elements()) {
      const auto old_pid = before.assignments.at(track.image_id).at(track.point2D_idx);
      if (old_pid != kInvalidPoint3DId) {
        ++plan.protected_count;
        if (old_pid != pid) merges.emplace_back(track.image_id, track.point2D_idx, old_pid, pid);
      } else {
        plan.new_observations.push_back(Evaluate(reconstruction, track, point.xyz));
      }
    }
    protected_observations += plan.protected_count;
    if (plan.new_observations.empty()) continue;
    std::set<image_t> passing_views;
    for (const auto& entry : plan.new_observations) {
      if (entry.rejection == nullptr) passing_views.insert(entry.observation.image_id);
    }
    const bool new_point = plan.protected_count == 0;
    const bool sufficient_support = !new_point || passing_views.size() >= 2;
    plan.kept = plan.protected_count;
    for (auto& entry : plan.new_observations) {
      ++cohort;
      if (entry.rejection == nullptr && !sufficient_support) {
        entry.rejection = "fewer_than_two_views";
      }
      if (entry.rejection == nullptr) {
        ++plan.kept;
        ++accepted;
        accepted_extensions += !new_point;
        accepted_new_points_observations += new_point;
      } else {
        ++rejected;
        const std::string reason = entry.rejection;
        rejected_depth += reason == "nonpositive_depth";
        rejected_error += reason == "over_4px";
        rejected_support += reason == "fewer_than_two_views";
      }
    }
    if (plan.kept == 1) {
      throw std::runtime_error("admission would discard a protected observation via singleton removal");
    }
    accepted_new_points += new_point && plan.kept >= 2;
    plans.push_back(std::move(plan));
  }
  if (before.assigned != protected_observations) {
    throw std::runtime_error("native track operation lost protected observations before admission");
  }
  if (!merges.empty()) {
    std::sort(merges.begin(), merges.end());
    std::ofstream merge_log(state.root / ("admission_merges_" + std::to_string(event) + ".csv"));
    merge_log.exceptions(std::ios::failbit | std::ios::badbit);
    merge_log << "image_id,point2D_idx,old_point3D_id,new_point3D_id\n";
    for (const auto& [image, index, old_pid, new_pid] : merges) {
      merge_log << image << ',' << index << ',' << old_pid << ',' << new_pid << '\n';
    }
  }
  std::sort(plans.begin(), plans.end(),
            [](const auto& a, const auto& b) { return a.point < b.point; });
  std::ofstream log(state.root / ("admission_" + std::to_string(event) + ".csv"));
  log.exceptions(std::ios::failbit | std::ios::badbit);
  log << std::setprecision(17) << "image_id,point2D_idx,point3D_id,protected_track_observations,depth,error_px,decision\n";
  for (const auto& plan : plans) {
    for (const auto& entry : plan.new_observations) {
      log << entry.observation.image_id << ',' << entry.observation.point2D_idx << ','
          << plan.point << ',' << plan.protected_count << ',' << entry.depth << ','
          << entry.error << ',' << (entry.rejection == nullptr ? "accepted" : entry.rejection) << '\n';
    }
    if (plan.kept == 0) {
      observations.DeletePoint3D(plan.point);
    } else {
      bool modified = false;
      for (const auto& entry : plan.new_observations) {
        if (entry.rejection == nullptr) continue;
        observations.DeleteObservation(entry.observation.image_id, entry.observation.point2D_idx);
        modified = true;
      }
      if (modified) triangulator.AddModifiedPoint3D(plan.point);
    }
    for (const auto& entry : plan.new_observations) {
      const bool present = reconstruction.Image(entry.observation.image_id)
                               .Point2D(entry.observation.point2D_idx).HasPoint3D();
      if (present != (entry.rejection == nullptr)) {
        throw std::runtime_error("admission factor assignment differs from the plan");
      }
    }
  }
  if (cohort != accepted + rejected || count_before_gate != protected_observations + cohort ||
      reconstruction.ComputeNumObservations() != count_before_gate - rejected) {
    throw std::runtime_error("admission observation count differs from the exact removal list");
  }
  std::ofstream index(state.root / "admission_events.jsonl", std::ios::app);
  index.exceptions(std::ios::failbit | std::ios::badbit);
  index << "{\"event\":" << event << ",\"block\":" << state.block_index
        << ",\"branch\":\"" << state.block << "\",\"operation\":\"" << operation
        << "\",\"frames\":" << reconstruction.NumRegFrames()
        << ",\"protected\":" << protected_observations << ",\"new\":" << cohort
        << ",\"accepted\":" << accepted << ",\"rejected\":" << rejected
        << ",\"rejected_depth\":" << rejected_depth << ",\"rejected_error\":" << rejected_error
        << ",\"rejected_support\":" << rejected_support
        << ",\"accepted_extensions\":" << accepted_extensions
        << ",\"accepted_new_points\":" << accepted_new_points
        << ",\"accepted_new_points_observations\":" << accepted_new_points_observations
        << ",\"protected_pid_changes\":" << merges.size()
        << ",\"observations_after\":" << reconstruction.ComputeNumObservations() << "}\n";
  return rejected;
}

}  // namespace colmap::terminal_admission
