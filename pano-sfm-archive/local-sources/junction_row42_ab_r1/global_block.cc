// Diagnostic entry point only: calls the unchanged native global refinement.
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/math/random.h"
#include "colmap/util/logging.h"
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  using namespace colmap;
  InitializeGlog(argv);
  std::filesystem::path input_path, output_path;
  OptionManager options;
  options.AddDatabaseOptions(); options.AddImageOptions(); options.AddMapperOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  if (!options.Parse(argc, argv)) return 1;
  SetPRNGSeed(0);
  auto database = Database::Open(*options.database_path);
  DatabaseCache::Options cache_options;
  cache_options.min_num_matches = options.mapper->min_num_matches;
  cache_options.ignore_watermarks = options.mapper->ignore_watermarks;
  cache_options.load_all_images = options.mapper->load_all_images;
  cache_options.convert_pose_priors_to_enu = options.mapper->use_prior_position;
  THROW_CHECK(options.mapper->image_names.empty());
  auto cache = DatabaseCache::Create(*database, cache_options);
  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);
  IncrementalMapper mapper(cache);
  mapper.BeginReconstruction(reconstruction);
  reconstruction->Write(output_path / "loaded");
  auto ba_options = options.mapper->GlobalBundleAdjustment();
  auto mapper_options = options.mapper->Mapper();
  std::cout << "GLOBAL_BLOCK_BEGIN frames=" << reconstruction->NumRegFrames()
            << " points=" << reconstruction->NumPoints3D() << std::endl;
  mapper.IterativeGlobalRefinement(options.mapper->ba_global_max_refinements,
      options.mapper->ba_global_max_refinement_change, mapper_options,
      ba_options, options.mapper->Triangulation());
  mapper.FilterFrames(mapper_options);
  reconstruction->Write(output_path / "final");
  std::cout << "GLOBAL_BLOCK_END frames=" << reconstruction->NumRegFrames()
            << " points=" << reconstruction->NumPoints3D() << std::endl;
  mapper.EndReconstruction(false);
  return 0;
}
