#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/correspondence_graph.h"
#include <set>
#include "colmap/util/logging.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
int main(int argc,char**argv){using namespace colmap;InitializeGlog(argv);
 std::filesystem::path input,output;OptionManager opt;opt.AddDatabaseOptions();opt.AddMapperOptions();opt.AddRequiredOption("input_path",&input);opt.AddRequiredOption("output_path",&output);if(!opt.Parse(argc,argv))return 1;
 auto db=Database::Open(*opt.database_path);DatabaseCache::Options co;co.min_num_matches=opt.mapper->min_num_matches;co.ignore_watermarks=opt.mapper->ignore_watermarks;co.load_all_images=opt.mapper->load_all_images;auto cache=DatabaseCache::Create(*db,co);
 auto r=std::make_shared<Reconstruction>();r->Read(input);IncrementalMapper mapper(cache);mapper.BeginReconstruction(r);
 THROW_CHECK(!r->Image(1579).HasPose());THROW_CHECK_EQ(r->Image(1579).FramePtr()->NumDataIds(),6);
 std::ofstream packet(output/"actual_packet.csv");packet<<std::setprecision(17)<<"image_id,keypoint_id,point3D_id,camera_idx,u,v,x,y,z\n";
 std::ofstream edges(output/"cache_edges.csv");edges<<"image_id,keypoint_id,source_image,source_keypoint\n";
 size_t ci=0;auto opts=opt.mapper->Mapper();
 for(const auto& data:r->Image(1579).FramePtr()->ImageIds()){
 const auto& im=r->Image(data.id);
 for(point2D_t k=0;k<im.NumPoints2D();k++){
 std::set<point3D_t> seen;auto range=cache->CorrespondenceGraph()->FindCorrespondences(data.id,k);
 for(auto corr=range.beg;corr<range.end;corr++){
 edges<<data.id<<','<<k<<','<<corr->image_id<<','<<corr->point2D_idx<<'\n';
 const auto& src=r->Image(corr->image_id);if(!src.HasPose())continue;const auto& pt=src.Point2D(corr->point2D_idx);if(!pt.HasPoint3D()||seen.count(pt.point3D_id))continue;
 if(src.CameraPtr()->HasBogusParams(opts.min_focal_length_ratio,opts.max_focal_length_ratio,opts.max_extra_param))continue;
 seen.insert(pt.point3D_id);const auto& X=r->Point3D(pt.point3D_id).xyz;const auto& xy=im.Point2D(k).xy;
 packet<<data.id<<','<<k<<','<<pt.point3D_id<<','<<ci<<','<<xy.x()<<','<<xy.y()<<','<<X.x()<<','<<X.y()<<','<<X.z()<<'\n';
 }}ci++;}packet.close();edges.close();if(std::getenv("CAPTURE_ONLY")){mapper.EndReconstruction(false);return 0;}
 auto capture=[&](const std::string& stage){std::ofstream f(output/(stage+".poses.csv"));f<<std::setprecision(17)<<"image_id,cx,cy,cz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
 for(auto id:r->RegImageIds()){auto p=r->Image(id).CamFromWorld();Eigen::Vector3d C=Inverse(p).translation();auto R=p.rotation().toRotationMatrix();f<<id<<','<<C.x()<<','<<C.y()<<','<<C.z();for(int i=0;i<3;i++)for(int j=0;j<3;j++)f<<','<<R(i,j);f<<'\n';}
 std::ofstream obs(output/(stage+".target.csv"));obs<<std::setprecision(17)<<"image_id,keypoint_id,pid,depth,error_px\n";
 for(auto id:r->RegImageIds()){const auto& im=r->Image(id);if(im.FrameId()!=r->Image(1579).FrameId())continue;auto pose=im.CamFromWorld();for(point2D_t k=0;k<im.NumPoints2D();k++){const auto& point=im.Point2D(k);if(!point.HasPoint3D())continue;Eigen::Vector3d z=pose*r->Point3D(point.point3D_id).xyz;auto uv=im.CameraPtr()->ImgFromCam(z);double e=uv.has_value()?(*uv-point.xy).norm():1e100;obs<<id<<','<<k<<','<<point.point3D_id<<','<<z.z()<<','<<e<<'\n';}}
 if(opt.mapper->Mapper().random_seed==0){std::filesystem::create_directories(output/stage);r->Write(output/stage);}};
 capture("before");bool success=mapper.RegisterNextImage(opt.mapper->Mapper(),1579);std::ofstream(output/"registration.txt")<<success<<'\n';if(!success){mapper.EndReconstruction(false);return 0;}
 capture("registered");for(const auto& data:r->Image(1579).FramePtr()->ImageIds())mapper.TriangulateImage(opt.mapper->Triangulation(),data.id);capture("triangulated");
 mapper.IterativeLocalRefinement(opt.mapper->ba_local_max_refinements,opt.mapper->ba_local_max_refinement_change,opt.mapper->Mapper(),opt.mapper->LocalBundleAdjustment(),opt.mapper->Triangulation(),1579);capture("local_refined");mapper.EndReconstruction(false);
}
