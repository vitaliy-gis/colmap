#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/estimators/generalized_pose.h"
#include <tuple>
#include <set>
#include "colmap/util/logging.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <map>
#include <algorithm>
#include <cmath>
int main(int argc,char**argv){using namespace colmap;InitializeGlog(argv);
 std::filesystem::path input,output,selected;OptionManager opt;opt.AddDatabaseOptions();opt.AddMapperOptions();opt.AddRequiredOption("input_path",&input);opt.AddRequiredOption("output_path",&output);opt.AddRequiredOption("selected",&selected);if(!opt.Parse(argc,argv))return 1;
 auto db=Database::Open(*opt.database_path);DatabaseCache::Options co;co.min_num_matches=opt.mapper->min_num_matches;co.ignore_watermarks=opt.mapper->ignore_watermarks;co.load_all_images=opt.mapper->load_all_images;auto cache=DatabaseCache::Create(*db,co);
 auto r=std::make_shared<Reconstruction>();r->Read(input);r->Load(*cache);
 auto root=output;
 for(image_t target:std::vector<image_t>{1578,1579,1580}){
 output=root/std::to_string(target);std::filesystem::create_directory(output);
 std::set<std::tuple<image_t,point2D_t,point3D_t>> selected_rows; std::vector<Eigen::Vector2d> xy;std::vector<Eigen::Vector3d> xyz;std::vector<size_t> camera_idx;std::vector<Camera> cameras;std::vector<Rigid3d> sensor;std::vector<std::tuple<image_t,point2D_t,point3D_t>> associations;
 THROW_CHECK(!r->Image(target).HasPose());THROW_CHECK_EQ(r->Image(target).FramePtr()->NumDataIds(),6);
 std::ofstream packet(output/"actual_packet.csv");packet<<std::setprecision(17)<<"image_id,keypoint_id,point3D_id,camera_idx,u,v,x,y,z\n";
 std::ofstream edges(output/"cache_edges.csv");edges<<"image_id,keypoint_id,source_image,source_keypoint\n";
 size_t ci=0;auto opts=opt.mapper->Mapper();THROW_CHECK(!opts.fix_existing_frames);
 std::set<std::pair<image_t,image_t>> reliable;
 std::ofstream scorefile(output/"sources.csv");scorefile<<"query,source,distance,rank,K,selected\n";
 for(auto d:r->Image(target).FramePtr()->ImageIds()){
 const auto& im=r->Image(d.id);std::map<image_t,double> scores;
 for(point2D_t k=0;k<im.NumPoints2D();k++){auto a=cache->CorrespondenceGraph()->FindCorrespondences(d.id,k);for(auto c=a.beg;c<a.end;c++){const auto& src=r->Image(c->image_id);if(!src.HasPose())continue;auto b=cache->CorrespondenceGraph()->FindCorrespondences(c->image_id,c->point2D_idx);size_t degree=im.Name()<src.Name()?size_t(a.end-a.beg):size_t(b.end-b.beg);scores[c->image_id]+=std::pow(.5,double(degree)-1);}}
 std::map<image_t,float> distances;float best=1e30;for(auto [id,v]:scores){distances[id]=float(1/(v+1e-6));best=std::min(best,distances[id]);}size_t K=0;for(auto [id,v]:distances)if(v<=2*best)K++;
 for(auto [id,v]:distances){size_t rank=1;for(auto [other,w]:distances)if(w<v)rank++;bool ok=rank<=K;if(ok)reliable.insert({d.id,id});scorefile<<d.id<<','<<id<<','<<v<<','<<rank<<','<<K<<','<<ok<<'\n';}
 for(point2D_t k=0;k<im.NumPoints2D();k++){auto a=cache->CorrespondenceGraph()->FindCorrespondences(d.id,k);for(auto c=a.beg;c<a.end;c++){const auto& src=r->Image(c->image_id);const auto& pt=src.Point2D(c->point2D_idx);if(src.HasPose()&&pt.HasPoint3D()&&reliable.count({d.id,c->image_id}))selected_rows.insert({d.id,k,pt.point3D_id});}}
 }
 THROW_CHECK(!selected_rows.empty());

 for(const auto& data:r->Image(target).FramePtr()->ImageIds()){
 const auto& im=r->Image(data.id);
 cameras.push_back(*im.CameraPtr());const auto& rig=*im.FramePtr()->RigPtr();sensor.push_back(rig.IsRefSensor(im.CameraPtr()->SensorId())?Rigid3d():rig.SensorFromRig(im.CameraPtr()->SensorId()));
 for(point2D_t k=0;k<im.NumPoints2D();k++){
 std::set<point3D_t> seen;auto range=cache->CorrespondenceGraph()->FindCorrespondences(data.id,k);
 for(auto corr=range.beg;corr<range.end;corr++){
 edges<<data.id<<','<<k<<','<<corr->image_id<<','<<corr->point2D_idx<<'\n';
 const auto& src=r->Image(corr->image_id);if(!src.HasPose())continue;const auto& pt=src.Point2D(corr->point2D_idx);if(!pt.HasPoint3D()||seen.count(pt.point3D_id))continue;
 if(src.CameraPtr()->HasBogusParams(opts.min_focal_length_ratio,opts.max_focal_length_ratio,opts.max_extra_param))continue;
 seen.insert(pt.point3D_id);const auto& X=r->Point3D(pt.point3D_id).xyz;const auto& uv=im.Point2D(k).xy;
 if(!selected_rows.empty()&&!selected_rows.count({data.id,k,pt.point3D_id}))continue;
 xy.push_back(im.Point2D(k).xy);xyz.push_back(X);camera_idx.push_back(ci);associations.push_back({data.id,k,pt.point3D_id});
 packet<<data.id<<','<<k<<','<<pt.point3D_id<<','<<ci<<','<<uv.x()<<','<<uv.y()<<','<<X.x()<<','<<X.y()<<','<<X.z()<<'\n';
 }}ci++;}packet.close();edges.close();if(std::getenv("CAPTURE_ONLY"))return 0;
 auto capture=[&](const std::string& stage){std::ofstream f(output/(stage+".poses.csv"));f<<std::setprecision(17)<<"image_id,cx,cy,cz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
 for(auto id:r->RegImageIds()){auto p=r->Image(id).CamFromWorld();Eigen::Vector3d C=Inverse(p).translation();auto R=p.rotation().toRotationMatrix();f<<id<<','<<C.x()<<','<<C.y()<<','<<C.z();for(int i=0;i<3;i++)for(int j=0;j<3;j++)f<<','<<R(i,j);f<<'\n';}
 std::ofstream obs(output/(stage+".target.csv"));obs<<std::setprecision(17)<<"image_id,keypoint_id,pid,depth,error_px\n";
 for(auto id:r->RegImageIds()){const auto& im=r->Image(id);if(im.FrameId()!=r->Image(target).FrameId())continue;auto pose=im.CamFromWorld();for(point2D_t k=0;k<im.NumPoints2D();k++){const auto& point=im.Point2D(k);if(!point.HasPoint3D())continue;Eigen::Vector3d z=pose*r->Point3D(point.point3D_id).xyz;auto uv=im.CameraPtr()->ImgFromCam(z);double e=uv.has_value()?(*uv-point.xy).norm():1e100;obs<<id<<','<<k<<','<<point.point3D_id<<','<<z.z()<<','<<e<<'\n';}}
 if((stage=="local_refined" && target==1580)||stage=="conflict"){std::filesystem::create_directories(output/stage);r->Write(output/stage);}};
 auto check=[&](){std::ifstream f(root/"checks.txt");image_t a,b;int ma,mb,n;double angle;std::map<std::pair<int,int>,int>bad;while(f>>a>>b>>ma>>mb>>n>>angle){Eigen::Matrix3d R;Eigen::Vector3d t;for(int i=0;i<3;i++)for(int j=0;j<3;j++)f>>R(i,j);for(int i=0;i<3;i++)f>>t(i);if(!r->ExistsImage(a)||!r->ExistsImage(b)||!r->Image(a).HasPose()||!r->Image(b).HasPose())continue;auto pa=r->Image(a).CamFromWorld(),pb=r->Image(b).CamFromWorld();auto rr=(pb.rotation()*pa.rotation().conjugate()).toRotationMatrix();Eigen::Vector3d tt=pb.translation()-rr*pa.translation();if(tt.norm()>1e-12&&tt.dot(t)/tt.norm()/t.norm()<-.8&&Eigen::AngleAxisd(rr*R.transpose()).angle()*180/3.141592653589793<3)bad[{ma,mb}]++;}for(auto [pair,count]:bad)if(count>=4){std::ofstream(root/"conflict.txt")<<pair.first<<' '<<pair.second<<' '<<count;return true;}return false;};
 capture("before");RANSACOptions ro;ro.max_error=opts.abs_pose_max_error;ro.min_inlier_ratio=opts.abs_pose_min_inlier_ratio;ro.random_seed=opts.random_seed;Rigid3d pose;size_t ni=0;std::vector<char> mask;
 bool success=EstimateGeneralizedAbsolutePose(ro,xy,xyz,camera_idx,sensor,cameras,&pose,&ni,&mask);AbsolutePoseRefinementOptions ref;ref.refine_focal_length=false;ref.refine_extra_params=false;
 if(success&&ni>=12)success=RefineGeneralizedAbsolutePose(ref,mask,xy,xyz,camera_idx,sensor,&pose,&cameras);else success=false;
 std::ofstream(output/"registration.txt")<<success<<'\n';std::ofstream(output/"pnp_support.txt")<<ni<<'\n';if(!success)return 0;
 std::ofstream pf(output/"pnp_pose.txt");pf<<std::setprecision(17)<<pose.ToMatrix()<<'\n';
 if(target==1578){std::ifstream f("/scratch/pano-sfm/work/runs/junction_rr_anchor248_r1/C0/pnp_pose.txt");Eigen::Matrix<double,3,4>M;for(int a=0;a<3;a++)for(int b=0;b<4;b++)f>>M(a,b);THROW_CHECK_LT((pose.ToMatrix()-M).cwiseAbs().maxCoeff(),1e-10);}
 auto frame_id=r->Image(target).FrameId();r->Frame(frame_id).SetRigFromWorld(pose);r->RegisterFrame(frame_id);
 std::vector<point3D_t> added;for(size_t k=0;k<mask.size();k++)if(mask[k]){auto [i,j,pid]=associations[k];if(!r->Image(i).Point2D(j).HasPoint3D()){r->AddObservation(pid,TrackElement(i,j));added.push_back(pid);}}
 IncrementalMapper mapper(cache);mapper.BeginReconstruction(r);for(auto pid:added)mapper.Triangulator().AddModifiedPoint3D(pid);
 capture("registered");if(check()){capture("conflict");mapper.EndReconstruction(false);return 0;}for(const auto& data:r->Image(target).FramePtr()->ImageIds())mapper.TriangulateImage(opt.mapper->Triangulation(),data.id);capture("triangulated");
 mapper.IterativeLocalRefinement(opt.mapper->ba_local_max_refinements,opt.mapper->ba_local_max_refinement_change,opt.mapper->Mapper(),opt.mapper->LocalBundleAdjustment(),opt.mapper->Triangulation(),target);capture("local_refined");if(check()){capture("conflict");mapper.EndReconstruction(false);return 0;}mapper.EndReconstruction(false);r->Load(*cache);
 }
}
