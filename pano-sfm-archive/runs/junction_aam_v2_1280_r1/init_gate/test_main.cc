#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/sfm/incremental_mapper_impl.h"
#include "colmap/util/logging.h"

#include <sqlite3.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace colmap { void RunMetricGateSyntheticTests(); }

int main(int argc, char** argv) {
  using namespace colmap;
  if (argc != 3) throw std::runtime_error("probe <database> <new output>");
  InitializeGlog(argv);
  RunMetricGateSyntheticTests();
  sqlite3_config(SQLITE_CONFIG_URI, 1);
  const std::string source = "file:" + std::string(argv[1]) + "?mode=ro&immutable=1";
  const std::string memory = "file:aam_probe?mode=memory&cache=shared";
  sqlite3 *src = nullptr, *mem = nullptr;
  THROW_CHECK_EQ(sqlite3_open_v2(source.c_str(), &src, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr), SQLITE_OK);
  THROW_CHECK_EQ(sqlite3_open_v2(memory.c_str(), &mem, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr), SQLITE_OK);
  auto* backup = sqlite3_backup_init(mem, "main", src, "main");
  THROW_CHECK(backup);
  THROW_CHECK_EQ(sqlite3_backup_step(backup, -1), SQLITE_DONE);
  THROW_CHECK_EQ(sqlite3_backup_finish(backup), SQLITE_OK);
  sqlite3_close(src);
  const auto database = Database::Open(memory);
  DatabaseCache::Options cache_options;
  cache_options.min_num_matches = 10;
  const auto cache = DatabaseCache::Create(*database, cache_options);
  IncrementalMapper::Options options;
  options.init_min_num_inliers = 20;
  options.random_seed = 0;
  setenv("AAM_INIT_PROBE", argv[2], 1);
  const auto result = IncrementalMapperImpl::EstimateInitialTwoViewGeometry(options, *cache, 1210, 1884);
  if (!result) return 42;
  std::ofstream output(std::string(argv[2]) + "/cam2_from_cam1.txt");
  output << std::setprecision(17) << result->cam2_from_cam1.ToMatrix() << '\n';
  return 0;
}
