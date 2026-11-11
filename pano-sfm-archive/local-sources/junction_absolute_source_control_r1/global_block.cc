// Frozen generalized absolute-pose control; unchanged native estimators/refinement.
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/estimators/generalized_pose.h"
#include "colmap/util/logging.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <algorithm>
#include <cmath>
using namespace colmap;
struct Row{int index;image_t image;point2D_t key;point3D_t pid;size_t cam;Eigen::Vector2d xy;Eigen::Vector3d xyz;int strong;};
int main(int argc,char**argv){InitializeGlog(argv);std::filesystem::path output,packet;std::string selection;OptionManager options;options.AddDatabaseOptions();options.AddImageOptions();options.AddMapperOptions();options.AddRequiredOption("packet_path",&packet);options.AddRequiredOption("output_path",&output);options.AddRequiredOption("selection",&selection);if(!options.Parse(argc,argv))return 1;THROW_CHECK(selection=="all"||selection=="strong");auto db=Database::Open(*options.database_path);std::ifstream f(packet);size_t nc;f>>nc;std::vector<Rigid3d> cams;std::vector<Camera> base;
 for(size_t k=0;k<nc;++k){camera_t id;f>>id;Eigen::Matrix3x4d m;for(int i=0;i<3;++i)for(int j=0;j<4;++j)f>>m(i,j);Rigid3d p;p.rotation()=Eigen::Quaterniond(m.leftCols<3>());p.translation()=m.col(3);cams.push_back(p);base.push_back(db->ReadCamera(id));}
 Eigen::Matrix3x4d orig;for(int i=0;i<3;++i)for(int j=0;j<4;++j)f>>orig(i,j);Eigen::Vector3d source;for(int i=0;i<3;++i)f>>source[i];double step;size_t n;f>>step>>n;std::vector<Row> rows;std::vector<Eigen::Vector2d> xy;std::vector<Eigen::Vector3d> xyz;std::vector<size_t> ci;
 for(size_t k=0;k<n;++k){Row r;f>>r.index>>r.image>>r.key>>r.pid>>r.cam>>r.xy.x()>>r.xy.y()>>r.xyz.x()>>r.xyz.y()>>r.xyz.z()>>r.strong;if(selection=="strong"&&!r.strong)continue;rows.push_back(r);xy.push_back(r.xy);xyz.push_back(r.xyz);ci.push_back(r.cam);}
 const auto mo=options.mapper->Mapper();RANSACOptions ro;ro.max_error=mo.abs_pose_max_error;ro.min_inlier_ratio=mo.abs_pose_min_inlier_ratio;ro.random_seed=mo.random_seed;AbsolutePoseRefinementOptions refine;refine.refine_focal_length=false;refine.refine_extra_params=false;
 std::ofstream cfg(output/"native_options.txt");cfg<<"rows "<<rows.size()<<"\nmax_error "<<ro.max_error<<"\nmin_inlier_ratio "<<ro.min_inlier_ratio<<"\nmin_num_inliers "<<mo.abs_pose_min_num_inliers<<"\nrandom_seed "<<ro.random_seed<<"\n";
 std::ofstream out(output/"fits.csv");out<<std::setprecision(17)<<"repeat,success,inliers,unique_pixels,negative_depth,dist_original_steps,dist_source_steps,median_error_px,cx,cy,cz\n";const Eigen::Vector3d original=-orig.leftCols<3>().transpose()*orig.col(3);
 for(int repeat=0;repeat<20;++repeat){auto cameras=base;Rigid3d pose;size_t inliers=0;std::vector<char> mask;bool ok=EstimateGeneralizedAbsolutePose(ro,xy,xyz,ci,cams,cameras,&pose,&inliers,&mask);if(ok && inliers>=size_t(mo.abs_pose_min_num_inliers))ok=RefineGeneralizedAbsolutePose(refine,mask,xy,xyz,ci,cams,&pose,&cameras);else ok=false;
  if(!ok){out<<repeat<<",0,0,0,0,0,0,0,0,0,0\n";continue;}for(size_t k=0;k<cameras.size();++k)THROW_CHECK(cameras[k].params==base[k].params);
  std::ofstream pf(output/("fit_"+std::to_string(repeat)+".txt"));pf<<std::setprecision(17)<<pose.ToMatrix()<<'\n';std::ofstream mf(output/("fit_"+std::to_string(repeat)+"_rows.csv"));mf<<std::setprecision(17)<<"original_row,image,keypoint,point3D,native_inlier,error_px,depth\n";size_t neg=0;std::vector<double> errors;std::set<std::pair<image_t,point2D_t>> pixels;
  for(size_t k=0;k<rows.size();++k){auto q=(cams[ci[k]]*pose)*xyz[k];auto uv=cameras[ci[k]].ImgFromCam(q);double err=uv.has_value()?(*uv-xy[k]).norm():1e100;mf<<rows[k].index<<','<<rows[k].image<<','<<rows[k].key<<','<<rows[k].pid<<','<<int(mask[k])<<','<<err<<','<<q.z()<<'\n';if(mask[k]){neg+=q.z()<=0;errors.push_back(err);pixels.emplace(rows[k].image,rows[k].key);}}
  std::sort(errors.begin(),errors.end());auto C=Inverse(pose).translation();out<<repeat<<",1,"<<inliers<<','<<pixels.size()<<','<<neg<<','<<(C-original).norm()/step<<','<<(C-source).norm()/step<<','<<errors[errors.size()/2]<<','<<C.x()<<','<<C.y()<<','<<C.z()<<'\n';out.flush();
 }
}
