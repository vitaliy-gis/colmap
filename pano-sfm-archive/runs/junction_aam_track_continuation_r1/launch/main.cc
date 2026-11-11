// Keep the real RunMapper controller. Only DB storage is redirected to RAM.
#include "colmap/exe/sfm.h"
#include "colmap/util/logging.h"
#include <sqlite3.h>
#include <string>
int main(int argc,char**argv) {
  using namespace colmap;
  sqlite3_config(SQLITE_CONFIG_URI,1);InitializeGlog(argv);
  int arg=-1;for(int i=1;i+1<argc;++i)if(std::string(argv[i])=="--database_path")arg=i+1;
  THROW_CHECK_GT(arg,0);
  const std::string source="file:"+std::string(argv[arg])+"?mode=ro&immutable=1";
  std::string memory="file:pano_aam_full?mode=memory&cache=shared";
  sqlite3 *src=nullptr,*mem=nullptr;
  THROW_CHECK_EQ(sqlite3_open_v2(source.c_str(),&src,SQLITE_OPEN_READONLY|SQLITE_OPEN_URI,nullptr),SQLITE_OK);
  THROW_CHECK_EQ(sqlite3_open_v2(memory.c_str(),&mem,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_URI,nullptr),SQLITE_OK);
  auto backup=sqlite3_backup_init(mem,"main",src,"main");THROW_CHECK(backup);
  THROW_CHECK_EQ(sqlite3_backup_step(backup,-1),SQLITE_DONE);THROW_CHECK_EQ(sqlite3_backup_finish(backup),SQLITE_OK);
  sqlite3_close(src);argv[arg]=memory.data();
  int result=RunMapper(argc,argv);sqlite3_close(mem);return result;
}
