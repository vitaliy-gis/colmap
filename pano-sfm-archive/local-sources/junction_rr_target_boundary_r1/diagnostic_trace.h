// Diagnostic persistence only. Opt-in via COLMAP_DIAGNOSTIC_TRACE.
#pragma once
#include "colmap/scene/reconstruction.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace colmap { namespace diagnostic_trace {
bool Enabled();
bool StopRequested();
void SetImage(image_t image);
image_t CurrentImage();
std::filesystem::path NextFile(const std::string& kind, const std::string& ext);
void Capture(const std::string& stage, const Reconstruction& rec, bool full=true);
bool GlobalDue(bool native_due, size_t frames);
std::vector<std::pair<image_t,size_t>> PriorTrials();

inline void WritePose(std::ostream& f, const Rigid3d& pose) {
  const auto matrix=pose.ToMatrix(); f << '[';
  for(int r=0;r<3;++r) for(int c=0;c<4;++c) {
    if(r || c) f << ',';
    f << matrix(r,c);
  }
  f << ']';
}

template<class Corrs, class Points2, class Points3, class Indices, class Poses>
void PnP(const std::string& stage, const Corrs& corrs, const Points2& xy,
         const Points3& xyz, const Indices& camera_idx, const Poses& cams,
         const std::vector<char>& mask, const Rigid3d* pose) {
  if(!Enabled()) return;
  const auto path=NextFile(stage,"json");
  std::ofstream f(path); f << std::setprecision(17);
  f << "{\"stage\":\"" << stage << "\",\"trigger_image\":" << CurrentImage()
    << ",\"correspondences\":" << corrs.size() << ",\"pose\":";
  if(pose) WritePose(f,*pose); else f << "null";
  f << ",\"cams_from_rig\":[";
  for(size_t i=0;i<cams.size();++i) {if(i)f<<',';WritePose(f,cams[i]);}
  f << "],\"rows_file\":\"" << path.stem().string() << ".csv\"}\n";
  auto rows=path; rows.replace_extension("csv"); std::ofstream g(rows);
  g << std::setprecision(17)
    << "image_id,keypoint_id,point3D_id,camera_idx,u,v,x,y,z,native_inlier\n";
  for(size_t i=0;i<corrs.size();++i) {
    g << corrs[i].image_id << ',' << corrs[i].point2D_idx << ','
      << corrs[i].point3D_id << ',' << camera_idx[i] << ','
      << xy[i].x() << ',' << xy[i].y() << ','
      << xyz[i].x() << ',' << xyz[i].y() << ',' << xyz[i].z() << ','
      << (mask.empty() ? -1 : int(mask[i])) << '\n';
  }
}

#ifdef COLMAP_DIAGNOSTIC_TRACE_IMPLEMENTATION
namespace {
struct Check {image_t a,b;int ma,mb,n;double angle;Eigen::Matrix3d R;Eigen::Vector3d t;};
struct State {
  std::filesystem::path root;
  size_t completed_globals=0; std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
  size_t serial=0, snapshots=0, max_frames=442, max_snapshots=240;
  image_t image=kInvalidImageId;
  bool stop=false, initialized=false, first_global_done=false;
  size_t first_global=0;
  std::string reason;
  std::vector<Check> checks;
  std::set<std::pair<int,int>> baseline_bad;
  State() {
    const char* p=std::getenv("COLMAP_DIAGNOSTIC_TRACE");if(!p || !*p)return;
    root=p;std::filesystem::create_directories(root);
    if(const char* v=std::getenv("COLMAP_DIAGNOSTIC_MAX_FRAMES"))max_frames=std::stoull(v);
    if(const char* v=std::getenv("COLMAP_DIAGNOSTIC_MAX_SNAPSHOTS"))max_snapshots=std::stoull(v);
    if(const char* v=std::getenv("COLMAP_DIAGNOSTIC_FIRST_GLOBAL"))first_global=std::stoull(v);
    if(const char* v=std::getenv("COLMAP_DIAGNOSTIC_CHECKS")) {
      std::ifstream f(v); Check c;
      while(f >> c.a >> c.b >> c.ma >> c.mb >> c.n >> c.angle) {
        for(int r=0;r<3;++r)for(int k=0;k<3;++k)f>>c.R(r,k);
        for(int k=0;k<3;++k)f>>c.t(k);checks.push_back(c);
      }
    }
  }
};
State& Get(){static State s;return s;}
}
bool Enabled(){return !Get().root.empty();}
bool StopRequested(){return Enabled() && Get().stop;}
void SetImage(image_t image){Get().image=image;}
image_t CurrentImage(){return Get().image;}
std::filesystem::path NextFile(const std::string& kind,const std::string& ext) {
  auto& s=Get();std::ostringstream n;
  n << std::setw(6) << std::setfill('0') << s.serial++ << '_' << kind << '.' << ext;
  return s.root/n.str();
}
std::vector<std::pair<image_t,size_t>> PriorTrials() {
  std::vector<std::pair<image_t,size_t>> out;
  if(!Enabled())return out;
  if(const char* p=std::getenv("COLMAP_DIAGNOSTIC_TRIALS")) {
    std::ifstream f(p);image_t i;size_t n;while(f>>i>>n)out.emplace_back(i,n);
  }
  return out;
}
bool GlobalDue(bool native_due,size_t frames) {
  auto& s=Get();if(!Enabled() || !s.first_global)return native_due;
  if(!s.first_global_done) {
    if(frames<s.first_global)return false;
    s.first_global_done=true;
    std::ofstream f(s.root/"restored_global_schedule.json");
    f << "{\"frames\":" << frames << ",\"native_due\":" << (native_due?"true":"false")
      << ",\"source\":\"first global action observed in original journal; prior point counter not serialized\"}\n";
    return true;
  }
  return native_due;
}
void Capture(const std::string& stage,const Reconstruction& rec,bool full) {
  if(!Enabled())return;
  auto& s=Get();const auto path=NextFile(stage,"json");
  struct Row {int a,b;size_t checked=0,bad=0;};
  std::map<std::pair<int,int>,Row> groups;
  std::ofstream pairs(path.string()+".checks.csv");
  pairs << std::setprecision(17) << "image_a,image_b,moment_a,moment_b,n,parallax_deg,direction_cos,rotation_error_deg\n";
  for(const auto& c:s.checks) {
    if(!rec.ExistsImage(c.a)||!rec.ExistsImage(c.b))continue;
    const auto& a=rec.Image(c.a);const auto& b=rec.Image(c.b);
    if(!a.HasPose()||!b.HasPose())continue;
    const Rigid3d pa=a.CamFromWorld(),pb=b.CamFromWorld();
    const Eigen::Matrix3d rr=(pb.rotation()*pa.rotation().conjugate()).toRotationMatrix();
    const Eigen::Vector3d tt=pb.translation()-rr*pa.translation();
    if(tt.norm()<1e-12)continue;
    const double cosine=tt.dot(c.t)/tt.norm()/c.t.norm();
    const double angle=Eigen::AngleAxisd(rr*c.R.transpose()).angle()*180.0/3.141592653589793;
    auto& g=groups[{c.ma,c.mb}];g.a=c.ma;g.b=c.mb;++g.checked;
    if(cosine<-.8 && angle<3.0)++g.bad;
    pairs << c.a << ',' << c.b << ',' << c.ma << ',' << c.mb << ',' << c.n << ','
          << c.angle << ',' << cosine << ',' << angle << '\n';
  }
  std::set<std::pair<int,int>> strong;
  for(const auto& p:groups)if(p.second.bad>=4)strong.insert(p.first);
  if(!s.initialized) {s.baseline_bad=strong;s.initialized=true;
    if(!strong.empty()){s.stop=true;s.reason="baseline_conflict";}}
  for(const auto& p:strong)if(!s.baseline_bad.count(p)) {s.stop=true;s.reason="new_signed_translation_conflict";}
  if(rec.NumRegFrames()>=s.max_frames && stage=="after_local_refinement" && !s.stop) {s.stop=true;s.reason="bounded_frame_limit";}
  if(stage=="after_global_pipeline")++s.completed_globals;
  if(!s.stop && stage=="after_global_pipeline" && s.completed_globals>=2){s.stop=true;s.reason="global_budget";}
  if(!s.stop && (stage=="after_local_refinement" || stage=="after_global_pipeline")) {
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-s.started).count()>=3900){s.stop=true;s.reason="elapsed_checkpoint_budget";}
    std::set<std::string> found;for(auto id:rec.RegImageIds()){const auto& n=rec.Image(id).Name();for(const auto& target:std::vector<std::string>{"frame_002324.png","frame_002328.png","frame_002332.png","frame_002336.png"})if(n.find(target)!=std::string::npos)found.insert(target);}
    if(found.size()==4){s.stop=true;s.reason="target_region_registered";}
  }
  if(full && s.snapshots>=s.max_snapshots) {full=false;s.stop=true;s.reason="snapshot_budget";}
  if(std::filesystem::space(s.root).available<100ull*1024*1024*1024) {
    full=false;s.stop=true;s.reason="disk_reserve";
  }
  // Persist every pose at every hook. Full maps at ten-frame boundaries,
  // global input/filter boundaries, and a detected stop only.
  std::ofstream poses(path.string()+".poses.csv");
  poses << std::setprecision(17) << "image_id,r00,r01,r02,tx,r10,r11,r12,ty,r20,r21,r22,tz\n";
  for(const auto id:rec.RegImageIds()) {
    poses << id;
    const auto matrix=rec.Image(id).CamFromWorld().ToMatrix();
    for(int r=0;r<3;++r)for(int c=0;c<4;++c)poses << ',' << matrix(r,c);
    poses << '\n';
  }
  bool target=false;
  if(rec.ExistsImage(s.image)) {
    const auto& n=rec.Image(s.image).Name();
    target=n.find("frame_002068.png")!=std::string::npos || n.find("frame_002074.png")!=std::string::npos;
  }
  bool target_pre=false;if(rec.ExistsImage(s.image)){const auto& n=rec.Image(s.image).Name();target_pre=stage=="before_registration_trial" && (n.find("frame_002328.png")!=std::string::npos || n.find("frame_002332.png")!=std::string::npos);}
  full=full && (stage=="resumed_initial" || stage=="before_global_pipeline" || stage=="after_global_pipeline" || target_pre || s.stop);
  std::string model;
  if(full) {
    model=path.stem().string()+"_model";
    std::filesystem::create_directory(s.root/model);rec.Write(s.root/model);++s.snapshots;
  }
  std::ofstream f(path);f<<std::setprecision(17);
  f << "{\"stage\":\""<<stage<<"\",\"trigger_image\":"<<s.image
    <<",\"frames\":"<<rec.NumRegFrames()<<",\"points\":"<<rec.NumPoints3D()
    <<",\"model\":\""<<model<<"\",\"stop\":"<<(s.stop?"true":"false")
    <<",\"reason\":\""<<s.reason<<"\",\"strong_conflicts\":[";
  bool first=true;for(const auto& p:strong){if(!first)f<<',';first=false;f<<'['<<p.first<<','<<p.second<<']';}
  f<<"]}\n";f.close();
  std::ifstream event(path);std::ofstream index(s.root/"events.jsonl",std::ios::app);index<<event.rdbuf();
  if(s.stop) {std::filesystem::copy_file(path,s.root/"stop.json",std::filesystem::copy_options::overwrite_existing);}
}
#endif
}}  // namespace colmap::diagnostic_trace
