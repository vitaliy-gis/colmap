#include "colmap/scene/reconstruction.h"
#include "colmap/geometry/sim3.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/util/logging.h"
#include <ceres/product_manifold.h>
#include <ceres/sphere_manifold.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){
 using namespace colmap;InitializeGlog(argv);THROW_CHECK_EQ(argc,5);
 Reconstruction r;r.Read(argv[1]);
 auto ids=r.RegFrameIds();auto first=*std::min_element(ids.begin(),ids.end());
 auto anchor=r.Frame(first).RigFromWorld();
 r.Transform(Sim3d(1.,anchor.rotation(),anchor.translation()));
 frame_t far=first;double dist=0;
 for(auto id:ids){double d=r.Frame(id).RigFromWorld().translation().norm();if(d>dist){dist=d;far=id;}}
 std::ifstream in(argv[4]);point3D_t pid;while(in>>pid)r.DeletePoint3D(pid);
 double scale=std::stod(argv[3]);
 for(auto id:ids)r.Frame(id).RigFromWorld().translation()*=scale;
 for(const auto& [id,p]:r.Points3D())r.Point3D(id).xyz*=scale;
 std::filesystem::path out(argv[2]);r.Write(out/"sparse"/"initial");
 BundleAdjustmentConfig c;for(auto id:r.RegImageIds())c.AddImage(id);
 c.SetConstantRigFromWorldPose(first);c.FixGauge(BundleAdjustmentGauge::UNSPECIFIED);
 BundleAdjustmentOptions o;o.refine_focal_length=false;o.refine_extra_params=false;
 o.refine_principal_point=false;o.refine_sensor_from_rig=false;
 o.ceres->use_gpu=false;o.ceres->solver_options.num_threads=6;
 o.ceres->solver_options.max_num_iterations=200;
 o.ceres->solver_options.max_solver_time_in_seconds=180;
 o.ceres->solver_options.gradient_tolerance=1e-10;
 auto ba=CreateDefaultCeresBundleAdjuster(o,c,r);
 ba->Problem()->SetManifold(r.Frame(far).RigFromWorld().params.data(),
 new ceres::ProductManifold<ceres::EigenQuaternionManifold,ceres::SphereManifold<3>>());
 ceres::Problem::EvaluateOptions ev;double before=0,after=0;
 ba->Problem()->Evaluate(ev,&before,nullptr,nullptr,nullptr);
 auto s=ba->Solve();ba->Problem()->Evaluate(ev,&after,nullptr,nullptr,nullptr);
 auto cs=std::dynamic_pointer_cast<CeresBundleAdjustmentSummary>(s);
 std::ofstream(out/"termination.txt")<<cs->ceres_summary.FullReport();
 std::ofstream m(out/"native_metrics.json");m<<std::setprecision(17)
 <<"{\"anchor_frame\":"<<first<<",\"far_frame\":"<<far<<",\"target_distance\":"<<dist*scale
 <<",\"achieved_distance\":"<<r.Frame(far).RigFromWorld().translation().norm()
 <<",\"initial_cost\":"<<before<<",\"final_cost\":"<<after<<"}";
 r.Write(out/"sparse"/"final");std::cout<<cs->ceres_summary.FullReport();
}
