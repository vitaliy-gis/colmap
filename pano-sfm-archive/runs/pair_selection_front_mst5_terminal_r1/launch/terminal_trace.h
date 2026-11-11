#pragma once

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/reconstruction.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace colmap::terminal_trace {

struct State {
  std::filesystem::path root;
  std::string block;
  size_t block_index = 0;
  size_t serial = 0;
  uintmax_t written_bytes = 0;
  bool active = false;
};

inline State& Get() {
  static thread_local State state;
  return state;
}

inline std::filesystem::path Capture(const char* stage,
                                     const Reconstruction& reconstruction) {
  auto& state = Get();
  if (!state.active) return {};
  constexpr uintmax_t gib = 1ull << 30;
  // Reserve space before every full model. Bounds are separate from mapper logic.
  constexpr uintmax_t max_next_model = 1 * gib;
  if (std::filesystem::space(state.root).available < 100 * gib + max_next_model ||
      state.written_bytes + max_next_model > 10 * gib) {
    throw std::runtime_error("terminal checkpoint disk budget reached");
  }
  std::ostringstream label;
  label << std::setw(4) << std::setfill('0') << state.serial++ << '_' << stage;
  const std::string basename = label.str();
  const auto path = state.root / basename;
  if (!std::filesystem::create_directory(path)) {
    throw std::runtime_error("terminal checkpoint already exists");
  }
  reconstruction.Write(path);
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file()) state.written_bytes += entry.file_size();
  }
  std::ofstream index(state.root / "events.jsonl", std::ios::app);
  index.exceptions(std::ios::failbit | std::ios::badbit);
  index << "{\"block\":" << state.block_index << ",\"branch\":\"" << state.block
        << "\",\"stage\":\"" << stage << "\",\"model\":\"" << basename
        << "\",\"frames\":" << reconstruction.NumRegFrames()
        << ",\"points\":" << reconstruction.NumPoints3D()
        << ",\"observations\":" << reconstruction.ComputeNumObservations() << "}\n";
  return path;
}

class Block {
 public:
  Block(const char* name, const Reconstruction& reconstruction, bool enabled = true) {
    const char* root = std::getenv("COLMAP_TERMINAL_TRACE");
    if (!enabled || root == nullptr || *root == '\0') return;
    auto& state = Get();
    if (state.active) throw std::logic_error("nested terminal refinement");
    state.root = root;
    state.block = name;
    ++state.block_index;
    state.active = true;
    Capture("block_input", reconstruction);
  }

  ~Block() { Get().active = false; }

  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  void Finish(const Reconstruction& reconstruction) {
    Capture("block_output", reconstruction);
  }
};

inline void CaptureSolve(const Reconstruction& reconstruction,
                         const BundleAdjustmentSummary& summary,
                         const bool usable) {
  const auto path = Capture("after_global_BA_solve", reconstruction);
  if (path.empty()) return;
  std::ofstream report(path / "solver.txt");
  report.exceptions(std::ios::failbit | std::ios::badbit);
  report << "solution_usable=" << usable << '\n'
         << "termination_enum=" << static_cast<int>(summary.termination_type) << '\n'
         << "residuals=" << summary.num_residuals << '\n'
         << summary.BriefReport() << '\n';
  if (const auto* ceres = dynamic_cast<const CeresBundleAdjustmentSummary*>(&summary)) {
    report << ceres->ceres_summary.FullReport() << '\n';
    std::ofstream iterations(path / "solver_iterations.csv");
    iterations.exceptions(std::ios::failbit | std::ios::badbit);
    iterations << std::setprecision(17)
               << "iteration,cost,cost_change,gradient_max_norm,step_norm,step_successful\n";
    for (const auto& step : ceres->ceres_summary.iterations) {
      iterations << step.iteration << ',' << step.cost << ',' << step.cost_change
                 << ',' << step.gradient_max_norm << ',' << step.step_norm
                 << ',' << step.step_is_successful << '\n';
    }
  }
}

}  // namespace colmap::terminal_trace
