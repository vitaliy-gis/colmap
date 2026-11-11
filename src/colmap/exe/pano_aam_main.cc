// Standalone AAM mapper with a private, in-memory copy of the input database.
#include "colmap/exe/sfm.h"
#include "colmap/util/logging.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <sqlite3.h>

namespace {

using DatabaseHandle = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;

DatabaseHandle OpenDatabase(const std::string& name, int flags) {
  sqlite3* handle = nullptr;
  const int result = sqlite3_open_v2(name.c_str(), &handle, flags, nullptr);
  DatabaseHandle database(handle, sqlite3_close);
  THROW_CHECK_EQ(result, SQLITE_OK);
  return database;
}

}  // namespace

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  constexpr std::string_view kDatabaseOption = "--database_path";
  const std::string memory_uri = "file:pano_aam?mode=memory&cache=shared";

  int database_index = -1;
  bool inline_option = false;
  std::string database_path;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      return colmap::RunMapper(argc, argv);
    }
    if (argument == kDatabaseOption && index + 1 < argc) {
      database_index = ++index;
      inline_option = false;
      database_path = argv[index];
    } else if (argument.substr(0, kDatabaseOption.size() + 1) ==
               "--database_path=") {
      database_index = index;
      inline_option = true;
      database_path = argument.substr(kDatabaseOption.size() + 1);
    }
  }

  if (database_path.empty()) {
    LOG(ERROR) << "Provide --database_path to run with an in-memory database.";
    return EXIT_FAILURE;
  }

  THROW_CHECK_EQ(sqlite3_config(SQLITE_CONFIG_URI, 1), SQLITE_OK);
  // An absolute filesystem path also prevents a relative name starting with
  // "file:" from being interpreted as an SQLite URI.
  auto source = OpenDatabase(
      std::filesystem::absolute(database_path).string(), SQLITE_OPEN_READONLY);
  auto memory = OpenDatabase(
      memory_uri, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI);
  auto* backup = sqlite3_backup_init(memory.get(), "main", source.get(), "main");
  THROW_CHECK_NOTNULL(backup);
  const int step_result = sqlite3_backup_step(backup, -1);
  const int finish_result = sqlite3_backup_finish(backup);
  THROW_CHECK_EQ(step_result, SQLITE_DONE);
  THROW_CHECK_EQ(finish_result, SQLITE_OK);
  source.reset();

  std::string mapper_database = inline_option
                                    ? "--database_path=" + memory_uri
                                    : memory_uri;
  argv[database_index] = mapper_database.data();
  return colmap::RunMapper(argc, argv);
}
