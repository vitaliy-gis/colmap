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
#include "colmap/optim/loransac.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cmath>
using namespace colmap;
double Quantile(std::vector<double> x,double q){if(x.empty())return -1;std::sort(x.begin(),x.end());return x[std::min(x.size()-1,size_t(q*(x.size()-1)))];}
int main(int argc,char**argv){
 InitializeGlog(argv);std::filesystem::path output;int first_id=0,second_id=0;std::string selection;
 OptionManager options;options.AddDatabaseOptions();options.AddImageOptions();options.AddMapperOptions();options.AddRequiredOption("first_id",&first_id);options.AddRequiredOption("second_id",&second_id);options.AddRequiredOption("selection",&selection);options.AddRequiredOption("output_path",&output);if(!options.Parse(argc,argv))return 1;
 auto db=Database::Open(*options.database_path);DatabaseCache::Options co;co.min_num_matches=options.mapper->min_num_matches;co.ignore_watermarks=options.mapper->ignore_watermarks;co.load_all_images=options.mapper->load_all_images;co.convert_pose_priors_to_enu=options.mapper->use_prior_position;auto cache=DatabaseCache::Create(*db,co);
 THROW_CHECK(selection=="all" || selection=="within_only");
 const auto& f1=cache->Frame(cache->Image(first_id).FrameId());const auto& f2=cache->Frame(cache->Image(second_id).FrameId());const auto& rig1=cache->Rig(f1.RigId());const auto& rig2=cache->Rig(f2.RigId());

 std::vector<GRNPObservation> obs1,obs2;std::vector<int> cohorts,lenses;std::vector<Eigen::Vector3d> origins;
 auto sensor=[](const Rig& rig,const Camera&cam){return rig.IsRefSensor(cam.SensorId())?Rigid3d():rig.SensorFromRig(cam.SensorId());};
 for(auto id:f1.ImageIds()){const auto&im=cache->Image(id.id);origins.push_back(Inverse(sensor(rig1,cache->Camera(im.CameraId()))).translation());}
 std::ofstream calib(output/"calibration.csv");calib<<std::setprecision(17)<<"camera,fx,fy,cx,cy,r00,r01,r02,tx,r10,r11,r12,ty,r20,r21,r22,tz\n";
 for(auto id:f1.ImageIds()){const auto&im=cache->Image(id.id);const auto&cam=cache->Camera(im.CameraId());THROW_CHECK_EQ(cam.params.size(),4);calib<<cam.camera_id;for(double p:cam.params)calib<<','<<p;auto mat=sensor(rig1,cam).ToMatrix();for(int i=0;i<3;++i)for(int j=0;j<4;++j)calib<<','<<mat(i,j);calib<<'\n';}

 double B=0;for(auto&a:origins)for(auto&b:origins)B=std::max(B,(a-b).norm());THROW_CHECK_GT(B,0.0);
 std::ofstream frozen(output/"frozen_rows.csv");frozen<<std::setprecision(17)<<"image1,keypoint1,image2,keypoint2,camera1,camera2,u1,v1,u2,v2,cohort\n";
 for(auto id1:f1.ImageIds())for(auto id2:f2.ImageIds()){
  const auto&i1=cache->Image(id1.id);const auto&i2=cache->Image(id2.id);const auto&c1=cache->Camera(i1.CameraId());const auto&c2=cache->Camera(i2.CameraId());Rigid3d p1=sensor(rig1,c1),p2=sensor(rig2,c2);FeatureMatches matches;cache->CorrespondenceGraph()->ExtractMatchesBetweenImages(id1.id,id2.id,matches);
  int cohort=(Inverse(p1).translation()-Inverse(p2).translation()).norm()<1e-6*B?0:1;
  if(selection=="within_only" && cohort==1)continue;
  for(const auto&m:matches){const auto&u=i1.Point2D(m.point2D_idx1).xy;const auto&v=i2.Point2D(m.point2D_idx2).xy;obs1.push_back({p1,c1.CamRayFromImgWithJac(u).value_or(CamRayWithJac::Zero())});obs2.push_back({p2,c2.CamRayFromImgWithJac(v).value_or(CamRayWithJac::Zero())});cohorts.push_back(cohort);lenses.push_back(cohort==1?1:(Inverse(p1).translation().norm()<1e-6*B?2:3));frozen<<id1.id<<','<<m.point2D_idx1<<','<<id2.id<<','<<m.point2D_idx2<<','<<c1.camera_id<<','<<c2.camera_id<<','<<u.x()<<','<<u.y()<<','<<v.x()<<','<<v.y()<<','<<cohort<<'\n';}
 }
 std::ofstream state(output/"frozen_input.txt");state<<std::setprecision(17)<<"B "<<B<<"\nfirst_id "<<first_id<<"\nsecond_id "<<second_id<<"\nselection "<<selection<<"\n";
 const double maxerr=options.mapper->Mapper().init_max_error;state<<"max_error "<<maxerr<<"\nrandom_seed "<<options.mapper->Mapper().random_seed<<"\n";
 std::ofstream repeats(output/"stability.csv");repeats<<std::setprecision(17)<<"repeat,success,inliers,positive_inliers,t_over_B,rotation_deg,tx,ty,tz\n";
 for(int repeat=0;repeat<20;++repeat){
  RANSACOptions ro;ro.min_num_trials=30;ro.max_error=maxerr;ro.random_seed=options.mapper->Mapper().random_seed;
  LORANSAC<GR6PEstimator,GR8PEstimator> ran(ro);auto report=ran.Estimate(obs1,obs2);
  if(!report.success){repeats<<repeat<<",0,0,0,0,0,0,0,0\n";continue;}
  std::ofstream matrix(output/("fit_"+std::to_string(repeat)+".txt"));matrix<<std::setprecision(17)<<report.model.ToMatrix()<<'\n';
  size_t pos=0;std::vector<double> rs;GR6PEstimator::Residuals(obs1,obs2,report.model,&rs);
  std::ofstream maskfile(output/("fit_"+std::to_string(repeat)+"_rows.csv"));maskfile<<std::setprecision(17)<<"row,native_inlier,squared_error\n";
  for(size_t k=0;k<rs.size();++k)maskfile<<k<<','<<int(report.inlier_mask[k])<<','<<rs[k]<<'\n';
  for(size_t k=0;k<rs.size();++k){if(rs[k]>maxerr*maxerr||!std::isfinite(rs[k]))continue;auto P1=obs1[k].cam_from_rig.ToMatrix();auto P2=(obs2[k].cam_from_rig*report.model).ToMatrix();auto x=obs1[k].ray_with_jac_in_cam.ray;auto y=obs2[k].ray_with_jac_in_cam.ray;
   Eigen::Matrix4d A;A.row(0)=x.x()*P1.row(2)-x.z()*P1.row(0);A.row(1)=x.y()*P1.row(2)-x.z()*P1.row(1);A.row(2)=y.x()*P2.row(2)-y.z()*P2.row(0);A.row(3)=y.y()*P2.row(2)-y.z()*P2.row(1);Eigen::Vector4d h=A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);if(std::abs(h[3])>1e-30){h/=h[3];pos+=((P1*h).z()>0&&(P2*h).z()>0);}
  }
  repeats<<repeat<<",1,"<<report.support.num_inliers<<','<<pos<<','<<report.model.translation().norm()/B<<','<<Eigen::AngleAxisd(report.model.rotation()).angle()*180./3.141592653589793<<','<<report.model.translation().x()<<','<<report.model.translation().y()<<','<<report.model.translation().z()<<'\n';repeats.flush();
 }

}
