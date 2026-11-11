// Inspect the first local solves without changing their parameters or factors.
#pragma once

#include <filesystem>
#include <fstream>
#include <set>

namespace colmap {
namespace recovery_trace {

inline void RecordAnchors(BundleAdjuster& adjuster, const Reconstruction& rec) {
  auto* ceres_adjuster = dynamic_cast<CeresBundleAdjuster*>(&adjuster);
  THROW_CHECK_NOTNULL(ceres_adjuster);
  auto& problem = *ceres_adjuster->Problem();
  const auto path = diagnostic_trace::NextFile("bootstrap_points", "csv");
  std::ofstream file(path);
  file << "point_id,constant,track_length,distinct_frames\n";
  for (const auto& [point_id, point] : rec.Points3D()) {
    if (!problem.HasParameterBlock(point.xyz.data())) {
      continue;
    }
    std::set<frame_t> frames;
    for (const auto& element : point.track.Elements()) {
      frames.insert(rec.Image(element.image_id).FrameId());
    }
    file << point_id << ','
         << problem.IsParameterBlockConstant(point.xyz.data()) << ','
         << point.track.Length() << ',' << frames.size() << '\n';
  }
}

}  // namespace recovery_trace
}  // namespace colmap
