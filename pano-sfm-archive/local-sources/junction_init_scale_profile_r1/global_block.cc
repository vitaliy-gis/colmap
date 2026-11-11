// Read-only sensitivity audit: unchanged native residuals; no mapper, BA or fit.
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/estimators/solvers/generalized_relative_pose.h"
#include "colmap/util/logging.h"
#include <Eigen/SVD>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cmath>
using namespace colmap;
double Quantile(std::vector<double> x,double q){if(x.empty())return -1;std::sort(x.begin(),x.end());return x[std::min(x.size()-1,size_t(q*(x.size()-1)))];}
int main(int argc,char**argv){
 InitializeGlog(argv);std::filesystem::path input,output;
 OptionManager options;options.AddDatabaseOptions();options.AddImageOptions();options.AddMapperOptions();options.AddRequiredOption("input_path",&input);options.AddRequiredOption("output_path",&output);if(!options.Parse(argc,argv))return 1;
 auto db=Database::Open(*options.database_path);DatabaseCache::Options co;co.min_num_matches=options.mapper->min_num_matches;co.ignore_watermarks=options.mapper->ignore_watermarks;co.load_all_images=options.mapper->load_all_images;co.convert_pose_priors_to_enu=options.mapper->use_prior_position;auto cache=DatabaseCache::Create(*db,co);
 Reconstruction rec;rec.Read(input);
 const auto& f1=cache->Frame(cache->Image(2912).FrameId());const auto& f2=cache->Frame(cache->Image(2242).FrameId());const auto& rig1=cache->Rig(f1.RigId());const auto& rig2=cache->Rig(f2.RigId());
 Rigid3d pose=rec.Frame(f2.FrameId()).RigFromWorld()*Inverse(rec.Frame(f1.FrameId()).RigFromWorld());
 std::vector<GRNPObservation> obs1,obs2;std::vector<int> cohorts;std::vector<Eigen::Vector3d> origins;
 auto sensor=[](const Rig& rig,const Camera&cam){return rig.IsRefSensor(cam.SensorId())?Rigid3d():rig.SensorFromRig(cam.SensorId());};
 for(auto id:f1.ImageIds()){const auto&im=cache->Image(id.id);origins.push_back(Inverse(sensor(rig1,cache->Camera(im.CameraId()))).translation());}
 double B=0;for(auto&a:origins)for(auto&b:origins)B=std::max(B,(a-b).norm());THROW_CHECK_GT(B,0.0);
 std::ofstream frozen(output/"frozen_rows.csv");frozen<<std::setprecision(17)<<"image1,keypoint1,image2,keypoint2,camera1,camera2,u1,v1,u2,v2,cohort\n";
 for(auto id1:f1.ImageIds())for(auto id2:f2.ImageIds()){
  const auto&i1=cache->Image(id1.id);const auto&i2=cache->Image(id2.id);const auto&c1=cache->Camera(i1.CameraId());const auto&c2=cache->Camera(i2.CameraId());Rigid3d p1=sensor(rig1,c1),p2=sensor(rig2,c2);FeatureMatches matches;cache->CorrespondenceGraph()->ExtractMatchesBetweenImages(id1.id,id2.id,matches);
  int cohort=(Inverse(p1).translation()-Inverse(p2).translation()).norm()<1e-6*B?0:1;
  for(const auto&m:matches){const auto&u=i1.Point2D(m.point2D_idx1).xy;const auto&v=i2.Point2D(m.point2D_idx2).xy;obs1.push_back({p1,c1.CamRayFromImgWithJac(u).value_or(CamRayWithJac::Zero())});obs2.push_back({p2,c2.CamRayFromImgWithJac(v).value_or(CamRayWithJac::Zero())});cohorts.push_back(cohort);frozen<<id1.id<<','<<m.point2D_idx1<<','<<id2.id<<','<<m.point2D_idx2<<','<<c1.camera_id<<','<<c2.camera_id<<','<<u.x()<<','<<u.y()<<','<<v.x()<<','<<v.y()<<','<<cohort<<'\n';}
 }
 std::ofstream state(output/"frozen_pose.txt");state<<std::setprecision(17)<<"B "<<B<<"\nt_over_B "<<pose.translation().norm()/B<<"\npose\n"<<pose.ToMatrix()<<"\n";
 std::ofstream out(output/"profile.csv");out<<std::setprecision(17)<<"factor,t_over_B,cohort,rows,inliers,positive_inliers,median_error_px,p95_error_px,inlier_median_parallax_deg\n";
 const double maxerr=options.mapper->Mapper().init_max_error;
 for(double factor:{.001,.003,.01,.03,.1,.3,1.,3.,10.,30.,100.,300.,1000.,3000.,10000.}){
  auto model=pose;model.translation()*=factor;std::vector<double> res;GR6PEstimator::Residuals(obs1,obs2,model,&res);
  for(int group:{-1,0,1}){std::vector<double> errors,angles;size_t in=0,pos=0;
   for(size_t k=0;k<res.size();++k){if(group>=0&&cohorts[k]!=group)continue;errors.push_back(std::sqrt(res[k]));if(res[k]>maxerr*maxerr || !std::isfinite(res[k]))continue;++in;
    const auto p1=obs1[k].cam_from_rig;const auto p2=obs2[k].cam_from_rig*model;const auto P1=p1.ToMatrix();const auto P2=p2.ToMatrix();const auto x=obs1[k].ray_with_jac_in_cam.ray;const auto y=obs2[k].ray_with_jac_in_cam.ray;
    Eigen::Matrix4d A;A.row(0)=x.x()*P1.row(2)-x.z()*P1.row(0);A.row(1)=x.y()*P1.row(2)-x.z()*P1.row(1);A.row(2)=y.x()*P2.row(2)-y.z()*P2.row(0);A.row(3)=y.y()*P2.row(2)-y.z()*P2.row(1);Eigen::Vector4d h=A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);if(std::abs(h[3])<1e-30)continue;h/=h[3];if((P1*h).z()>0&&(P2*h).z()>0){++pos;auto d1=(p1.rotation().inverse()*x).normalized();auto d2=(p2.rotation().inverse()*y).normalized();angles.push_back(std::acos(std::clamp(d1.dot(d2),-1.,1.))*180./3.141592653589793);}
   }
   out<<factor<<','<<model.translation().norm()/B<<','<<group<<','<<errors.size()<<','<<in<<','<<pos<<','<<Quantile(errors,.5)<<','<<Quantile(errors,.95)<<','<<Quantile(angles,.5)<<'\n';
  }
 }
}
