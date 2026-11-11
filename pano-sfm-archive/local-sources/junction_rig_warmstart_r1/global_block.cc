#include "colmap/scene/reconstruction.h"
#include "colmap/scene/database.h"
#include "colmap/scene/rig.h"
#include "colmap/geometry/sim3.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/util/logging.h"
#include <filesystem>
#include <iostream>
#include <fstream>
int main(int argc,char** argv) {
 using namespace colmap;
 InitializeGlog(argv);
 THROW_CHECK_EQ(argc,6);
 Reconstruction r; r.Read(argv[1]);
 const auto n=r.NumPoints3D();
 r.Transform(Sim3d(std::stod(argv[5]),Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero()));
 auto db=Database::Open(argv[2]);
 ApplyRigConfig(ReadRigConfig(argv[3]),*db,&r);
 THROW_CHECK_EQ(r.NumPoints3D(),n);
 std::filesystem::path out(argv[4]);
 r.Write(out/"sparse"/"initial");
 BundleAdjustmentConfig c;
 for(auto id:r.RegImageIds()) c.AddImage(id);
 auto ids=r.RegFrameIds();
 c.SetConstantRigFromWorldPose(*std::min_element(ids.begin(),ids.end()));
 c.FixGauge(BundleAdjustmentGauge::UNSPECIFIED);
 BundleAdjustmentOptions o;
 o.refine_focal_length=false;o.refine_principal_point=false;o.refine_extra_params=false;
 o.refine_sensor_from_rig=false;
 o.ceres->use_gpu=false;
 o.ceres->solver_options.num_threads=6;
 o.ceres->solver_options.max_num_iterations=100;
 o.ceres->solver_options.max_solver_time_in_seconds=180;
 auto ba=CreateDefaultBundleAdjuster(o,c,r);
 auto s=ba->Solve();
 std::ofstream(out/"termination.txt")<<s->BriefReport()<<"\n";
 r.Write(out/"sparse"/"final");
 THROW_CHECK_EQ(r.NumPoints3D(),n);
 std::cout<<s->BriefReport()<<std::endl;
}
