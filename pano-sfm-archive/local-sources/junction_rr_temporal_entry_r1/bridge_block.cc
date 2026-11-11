#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/util/logging.h"
#include <sqlite3.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <cmath>
using namespace colmap;
int main(int argc,char**argv){sqlite3_config(SQLITE_CONFIG_URI,1);InitializeGlog(argv);
 std::filesystem::path input,output;OptionManager opt;opt.AddDatabaseOptions();opt.AddMapperOptions();opt.AddRequiredOption("input_path",&input);opt.AddRequiredOption("output_path",&output);if(!opt.Parse(argc,argv))return 1;
 sqlite3 *src=nullptr,*mem=nullptr;THROW_CHECK_EQ(sqlite3_open_v2(("file:"+opt.database_path->string()+"?mode=ro&immutable=1").c_str(),&src,SQLITE_OPEN_READONLY|SQLITE_OPEN_URI,nullptr),SQLITE_OK);THROW_CHECK_EQ(sqlite3_open_v2("file:pano_bridge?mode=memory&cache=shared",&mem,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_URI,nullptr),SQLITE_OK);auto backup=sqlite3_backup_init(mem,"main",src,"main");THROW_CHECK(backup);THROW_CHECK_EQ(sqlite3_backup_step(backup,-1),SQLITE_DONE);THROW_CHECK_EQ(sqlite3_backup_finish(backup),SQLITE_OK);sqlite3_close(src);
 auto db=Database::Open("file:pano_bridge?mode=memory&cache=shared");DatabaseCache::Options co;co.min_num_matches=opt.mapper->min_num_matches;co.ignore_watermarks=opt.mapper->ignore_watermarks;co.load_all_images=opt.mapper->load_all_images;auto cache=DatabaseCache::Create(*db,co);auto r=std::make_shared<Reconstruction>();r->Read(input);r->Load(*cache);IncrementalMapper mapper(cache);mapper.BeginReconstruction(r);auto opts=opt.mapper->Mapper();THROW_CHECK(!opts.fix_existing_frames);
 auto capture=[&](std::filesystem::path d,std::string stage,image_t target){std::ofstream f(d/(stage+".poses.csv"));f<<std::setprecision(17)<<"image_id,r00,r01,r02,tx,r10,r11,r12,ty,r20,r21,r22,tz\n";for(auto id:r->RegImageIds()){auto M=r->Image(id).CamFromWorld().ToMatrix();f<<id;for(int i=0;i<3;i++)for(int j=0;j<4;j++)f<<','<<M(i,j);f<<'\n';}std::ofstream o(d/(stage+".observations.csv"));o<<std::setprecision(17)<<"image_id,keypoint,pid,depth,error_px\n";for(auto data:r->Image(target).FramePtr()->ImageIds()){const auto&im=r->Image(data.id);if(!im.HasPose())continue;for(point2D_t k=0;k<im.NumPoints2D();k++){const auto&p=im.Point2D(k);if(!p.HasPoint3D())continue;auto z=im.CamFromWorld()*r->Point3D(p.point3D_id).xyz;auto uv=im.CameraPtr()->ImgFromCam(z);o<<data.id<<','<<k<<','<<p.point3D_id<<','<<z.z()<<','<<(uv?(*uv-p.xy).norm():1e100)<<'\n';}}};
 auto conflict=[&](){std::ifstream f("/scratch/pano-sfm/work/runs/junction_early_boundary_r1/replay/checks.txt");image_t a,b;int ma,mb,n;double angle;std::map<std::pair<int,int>,int>bad;while(f>>a>>b>>ma>>mb>>n>>angle){Eigen::Matrix3d R;Eigen::Vector3d t;for(int i=0;i<3;i++)for(int j=0;j<3;j++)f>>R(i,j);for(int i=0;i<3;i++)f>>t(i);if(!r->ExistsImage(a)||!r->ExistsImage(b)||!r->Image(a).HasPose()||!r->Image(b).HasPose())continue;auto pa=r->Image(a).CamFromWorld(),pb=r->Image(b).CamFromWorld();auto rr=(pb.rotation()*pa.rotation().conjugate()).toRotationMatrix();Eigen::Vector3d tt=pb.translation()-rr*pa.translation();if(tt.norm()>1e-12&&tt.dot(t)/tt.norm()/t.norm()<-.8&&Eigen::AngleAxisd(rr*R.transpose()).angle()*180/3.141592653589793<3)bad[{ma,mb}]++;}for(auto [pair,count]:bad)if(count>=4){std::ofstream(output/"conflict.txt")<<pair.first<<' '<<pair.second<<' '<<count;return true;}return false;};
 for(int step=0;step<3;step++){
  THROW_CHECK_GT(std::filesystem::space(output).available,100ull*1024*1024*1024);
  auto next=mapper.FindNextImages(opts);image_t target=kInvalidImageId;double highest=-1;auto d=output/("step"+std::to_string(step));std::filesystem::create_directory(d);std::ofstream ranks(d/"candidate_scores.csv");ranks<<std::setprecision(17)<<"native_rank,image_id,name,max_aam,best_source\n";
  for(size_t rank=0;rank<next.size();rank++){auto id=next[rank];const auto&im=r->Image(id);std::map<image_t,double>scores;for(point2D_t k=0;k<im.NumPoints2D();k++){auto a=cache->CorrespondenceGraph()->FindCorrespondences(id,k);for(auto c=a.beg;c<a.end;c++){const auto&si=r->Image(c->image_id);if(!si.HasPose())continue;auto b=cache->CorrespondenceGraph()->FindCorrespondences(c->image_id,c->point2D_idx);size_t degree=im.Name()<si.Name()?size_t(a.end-a.beg):size_t(b.end-b.beg);scores[c->image_id]+=std::pow(.5,double(degree)-1);}}float best=1e30;image_t source=kInvalidImageId;for(auto [sid,v]:scores){float delta=float(1/(v+1e-6));if(delta<best){best=delta;source=sid;}}double score=1.0/best;ranks<<rank<<','<<id<<','<<im.Name()<<','<<score<<','<<source<<'\n';if(score>highest||(score==highest&&id<target)){highest=score;target=id;}}
  if(target==kInvalidImageId){std::ofstream(output/"stop.txt")<<"no_candidate";break;}std::ofstream(d/"selected.txt")<<target<<' '<<r->Image(target).Name();capture(d,"before",target);
  // Freeze complete incoming data before native registration, without changing it.
  std::ofstream packet(d/"input.csv");packet<<std::setprecision(17)<<"image_id,keypoint,pid,source_image,source_keypoint,u,v,x,y,z\n";for(auto data:r->Image(target).FramePtr()->ImageIds()){const auto&im=r->Image(data.id);for(point2D_t k=0;k<im.NumPoints2D();k++){auto a=cache->CorrespondenceGraph()->FindCorrespondences(data.id,k);for(auto c=a.beg;c<a.end;c++){const auto&si=r->Image(c->image_id);if(!si.HasPose())continue;const auto&p=si.Point2D(c->point2D_idx);if(!p.HasPoint3D())continue;const auto&X=r->Point3D(p.point3D_id).xyz;auto xy=im.Point2D(k).xy;packet<<data.id<<','<<k<<','<<p.point3D_id<<','<<c->image_id<<','<<c->point2D_idx<<','<<xy.x()<<','<<xy.y()<<','<<X.x()<<','<<X.y()<<','<<X.z()<<'\n';}}}packet.close();
  bool ok=mapper.RegisterNextImage(opts,target);std::ofstream(d/"registration.txt")<<ok;if(!ok){std::ofstream(output/"stop.txt")<<"native_registration_failed";break;}
  if(step==0){std::ifstream f("/scratch/pano-sfm/work/runs/junction_rr_temporal_entry_r1/fit305/A0/pnp_pose.txt");Eigen::Matrix<double,3,4>M;for(int i=0;i<3;i++)for(int j=0;j<4;j++)f>>M(i,j);THROW_CHECK_LT((r->Image(target).FramePtr()->RigFromWorld().ToMatrix()-M).cwiseAbs().maxCoeff(),1e-10);}
  capture(d,"registered",target);if(conflict())break;
  for(auto data:r->Image(target).FramePtr()->ImageIds())mapper.TriangulateImage(opt.mapper->Triangulation(),data.id);
  capture(d,"triangulated",target);mapper.IterativeLocalRefinement(opt.mapper->ba_local_max_refinements,opt.mapper->ba_local_max_refinement_change,opts,opt.mapper->LocalBundleAdjustment(),opt.mapper->Triangulation(),target);capture(d,"local_refined",target);if(conflict())break;
 }
 std::filesystem::create_directory(output/"final");r->Write(output/"final");mapper.EndReconstruction(false);
}
