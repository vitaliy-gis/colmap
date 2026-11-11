#include "colmap/estimators/generalized_pose.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/scene/camera.h"
#include "colmap/util/logging.h"
#include <fstream>
#include <iomanip>
#include <set>
#include <iostream>
using namespace colmap;
int main(int argc,char**argv){InitializeGlog(argv);THROW_CHECK_EQ(argc,3);
 std::ifstream in(argv[1]);size_t nc,n;in>>nc;
 std::vector<Camera> cams(nc);std::vector<Rigid3d> ext;
 for(size_t i=0;i<nc;i++){auto&c=cams[i];c.camera_id=i+1;c.model_id=CameraModelId::kPinhole;in>>c.width>>c.height;c.params.resize(4);for(double&x:c.params)in>>x;Eigen::Matrix<double,3,4>M;for(int j=0;j<3;j++)for(int k=0;k<4;k++)in>>M(j,k);ext.emplace_back(Eigen::Quaterniond(M.leftCols<3>()),M.col(3));}
 in>>n;std::vector<Eigen::Vector2d>xy(n);std::vector<Eigen::Vector3d>xyz(n);std::vector<size_t>idx(n);std::vector<uint64_t>im(n),kp(n),pid(n);
 for(size_t k=0;k<n;k++)in>>idx[k]>>xy[k].x()>>xy[k].y()>>xyz[k].x()>>xyz[k].y()>>xyz[k].z()>>im[k]>>kp[k]>>pid[k];
 IncrementalMapper::Options defaults;RANSACOptions ro;ro.max_error=defaults.abs_pose_max_error;ro.min_inlier_ratio=defaults.abs_pose_min_inlier_ratio;ro.random_seed=defaults.random_seed;
 AbsolutePoseRefinementOptions ref;ref.refine_focal_length=false;ref.refine_extra_params=false;
 std::ofstream out(std::string(argv[2])+"_fits.csv");out<<std::setprecision(17)<<"repeat,stage,success,native_inliers,positive_threshold_rows,unique_pixels,unique_pids,cx,cy,cz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
 std::ofstream opts(std::string(argv[2])+"_options.txt");opts<<"max_error "<<ro.max_error<<"\nmin_inlier_ratio "<<ro.min_inlier_ratio<<"\nrandom_seed "<<ro.random_seed<<"\n";
 for(int r=0;r<20;r++){
  Rigid3d pose;size_t ni=0;std::vector<char>mask;
  bool ok=EstimateGeneralizedAbsolutePose(ro,xy,xyz,idx,ext,cams,&pose,&ni,&mask);
  for(int stage=0;stage<2;stage++){
   if(stage&&ok)ok=RefineGeneralizedAbsolutePose(ref,mask,xy,xyz,idx,ext,&pose,&cams);
   if(!ok){out<<r<<','<<stage<<",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";break;}
   std::ofstream rows(std::string(argv[2])+"_"+std::to_string(r)+"_"+std::to_string(stage)+".csv");rows<<std::setprecision(17)<<"row,native_inlier,error_px,depth\n";
   size_t count=0;std::set<std::pair<uint64_t,uint64_t>>pixels;std::set<uint64_t>points;
   for(size_t k=0;k<n;k++){auto z=(ext[idx[k]]*pose)*xyz[k];auto uv=cams[idx[k]].ImgFromCam(z);bool valid=uv.has_value();double e=valid?(*uv-xy[k]).norm():1e100;
    rows<<k<<','<<int(mask[k])<<','<<e<<','<<z.z()<<'\n';
    if(valid&&e<=ro.max_error){count++;pixels.insert({im[k],kp[k]});points.insert(pid[k]);}
   }
   auto C=Inverse(pose).translation();auto R=pose.rotation().toRotationMatrix();out<<r<<','<<stage<<",1,"<<ni<<','<<count<<','<<pixels.size()<<','<<points.size()<<','<<C.x()<<','<<C.y()<<','<<C.z();for(int i=0;i<3;i++)for(int j=0;j<3;j++)out<<','<<R(i,j);out<<'\n';out.flush();
  }
 }
}
