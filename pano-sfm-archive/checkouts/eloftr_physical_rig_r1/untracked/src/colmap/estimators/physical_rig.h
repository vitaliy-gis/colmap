// Experimental physical two-lens parameterization. Opt-in; PINHOLE only.
// Every observation uses A_view_from_lens * B_lens_from_rig * G_rig_from_world.
// A is immutable input. Only the second lens has a variable, shared B.
#include <cstdlib>
#include <fstream>
#include <map>

namespace colmap {
namespace {

struct PhysicalRigSlot {
  int lens = -1;
  Rigid3d view_from_lens;
};

class PhysicalRigState {
 public:
  std::map<camera_t, PhysicalRigSlot> slots;
  Rig* rig = nullptr;
  sensor_t other_sensor;
  bool refine_b = false;

  void Initialize(Reconstruction& reconstruction,
                  BundleAdjustmentConfig& config,
                  const BundleAdjustmentOptions& options) {
    const char* path = std::getenv("COLMAP_PHYSICAL_RIG_CONFIG");
    if (!path) return;
    std::ifstream input(path);
    THROW_CHECK(input.good());
    camera_t id;
    int lens;
    double x, y, z, w;
    camera_t reference = kInvalidCameraId, other = kInvalidCameraId;
    while (input >> id >> lens >> x >> y >> z >> w) {
      THROW_CHECK(lens == 0 || lens == 1);
      PhysicalRigSlot slot;
      slot.lens = lens;
      slot.view_from_lens.rotation() = Eigen::Quaterniond(w, x, y, z).normalized();
      slot.view_from_lens.translation().setZero();
      THROW_CHECK(slots.emplace(id, slot).second);
      if (slot.view_from_lens.rotation().angularDistance(Eigen::Quaterniond::Identity()) < 1e-12) {
        if (lens == 0) reference = id; else other = id;
      }
    }
    THROW_CHECK(input.eof());
    THROW_CHECK_EQ(slots.size(), 6);
    THROW_CHECK_NE(reference, kInvalidCameraId);
    THROW_CHECK_NE(other, kInvalidCameraId);
    THROW_CHECK(!config.Images().empty());
    rig = reconstruction.Image(*config.Images().begin()).FramePtr()->RigPtr();
    THROW_CHECK(rig->RefSensorId() == reconstruction.Camera(reference).SensorId());
    other_sensor = reconstruction.Camera(other).SensorId();
    refine_b = options.refine_sensor_from_rig &&
               config.Images().size() == reconstruction.NumRegImages();
    for (const auto image_id : config.Images()) {
      const Image& image = reconstruction.Image(image_id);
      THROW_CHECK_EQ(image.FramePtr()->RigId(), rig->RigId());
      THROW_CHECK(slots.count(image.CameraId()));
    }
    for (const auto& [cam_id, slot] : slots) {
      const auto sensor = reconstruction.Camera(cam_id).SensorId();
      if (slot.lens == 0 && sensor != rig->RefSensorId()) {
        config.SetConstantSensorFromRigPose(sensor);
      }
      if (slot.lens == 1 && config.HasConstantSensorFromRigPose(sensor)) refine_b = false;
      if (sensor == rig->RefSensorId()) continue;
      const Rigid3d expected = slot.lens == 0 ? slot.view_from_lens : slot.view_from_lens * B();
      const Rigid3d& actual = rig->SensorFromRig(sensor);
      THROW_CHECK_LT(expected.rotation().angularDistance(actual.rotation()), 1e-8);
      THROW_CHECK_LT((expected.translation()-actual.translation()).norm(), 1e-8 * std::max(1., B().translation().norm()));
    }
    VLOG(1) << "PHYSICAL_RIG: exact A; one shared B; refine_B=" << refine_b
            << "; images=" << config.Images().size();
  }

  Rigid3d& B() { return rig->SensorFromRig(other_sensor); }
  void Parameterize(ceres::Problem& problem) {
    if (!rig || !problem.HasParameterBlock(B().params.data())) return;
    SetManifold(&problem, B().params.data(),
                CreateProductManifold(CreateEigenQuaternionManifold(), CreateEuclideanManifold<3>()));
    if (!refine_b) problem.SetParameterBlockConstant(B().params.data());
  }
  void Synchronize(Reconstruction& reconstruction) {
    if (!rig) return;
    const Rigid3d b = B();
    for (const auto& [cam_id, slot] : slots) {
      const auto sensor = reconstruction.Camera(cam_id).SensorId();
      if (sensor == rig->RefSensorId()) continue;
      rig->SensorFromRig(sensor) = slot.lens == 0 ? slot.view_from_lens : slot.view_from_lens * b;
    }
  }
};

class PhysicalRigReprojError : public AutoDiffCostFunctor<PhysicalRigReprojError, 2, 3, 7, 7, 4> {
 public:
  PhysicalRigReprojError(const Eigen::Vector2d& xy, const Rigid3d& a)
      : xy_(xy), a_(a.rotation().toRotationMatrix()) {}
  template <typename T>
  bool operator()(const T* p, const T* b, const T* g, const T* k, T* residuals) const {
    const Eigen::Matrix<T, 3, 1> in_rig = EigenQuaternionMap<T>(g) * EigenVector3Map<T>(p) + EigenVector3Map<T>(g+4);
    const Eigen::Matrix<T, 3, 1> in_view = a_.cast<T>() * (EigenQuaternionMap<T>(b)*in_rig + EigenVector3Map<T>(b+4));
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(residuals);
    if (PinholeCameraModel::ImgFromCam(k, in_view[0], in_view[1], in_view[2], &residuals[0], &residuals[1])) r -= xy_.cast<T>();
    else r.setZero();
    return true;
  }
 private:
  Eigen::Vector2d xy_;
  Eigen::Matrix3d a_;
};

}  // namespace
}  // namespace colmap
