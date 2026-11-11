        local_bundle_image_ids.push_back(overlapping_image.ImageId());
        used_overlapping_images[overlapping_image_idx] = true;
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }

    // Check if we already collected enough images.
    if (local_bundle_image_ids.size() >= num_eff_images) {
      break;
    }
  }

  // In case there are not enough images with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping images.

  if (local_bundle_image_ids.size() < num_eff_images) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_images[overlapping_image_idx]) {
        local_bundle_image_ids.push_back(
            overlapping_images[overlapping_image_idx].first);
        used_overlapping_images[overlapping_image_idx] = true;

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }
  }

  return local_bundle_image_ids;
}

namespace {

bool EstimateInitialGeneralizedTwoViewGeometry(
    const IncrementalMapper::Options& options,
    const DatabaseCache& database_cache,
    const Image& orig_image1,
    const Image& orig_image2,
    const Frame& frame1,
    const Frame& frame2,
    const Rig& rig1,
    const Rig& rig2,
    Rigid3d& orig_cam2_from_orig_cam1) {
  std::vector<Eigen::Vector2d> points2D1;
  std::vector<Eigen::Vector2d> points2D2;
  std::vector<size_t> camera_idxs1;
  std::vector<size_t> camera_idxs2;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;

  FlatHashMap<camera_t, size_t> camera_id_to_idx;
  auto maybe_add_camera = [&cameras, &cams_from_rig, &camera_id_to_idx](
                              const Rig& rig, const Camera& camera) {
    const auto [it, inserted] =
        camera_id_to_idx.emplace(camera.camera_id, cameras.size());
    if (inserted) {
      cameras.push_back(camera);
      if (rig.IsRefSensor(camera.SensorId())) {
        cams_from_rig.push_back(Rigid3d());
      } else {
        cams_from_rig.push_back(rig.SensorFromRig(camera.SensorId()));
      }
    }
    return it->second;
  };

  FeatureMatches matches;
  for (const data_t& image_id1 : frame1.ImageIds()) {
    const Image& image1 = database_cache.Image(image_id1.id);
    const Camera& camera1 = database_cache.Camera(image1.CameraId());
    const size_t camera_idx1 = maybe_add_camera(rig1, camera1);

    for (const data_t& image_id2 : frame2.ImageIds()) {
      const Image& image2 = database_cache.Image(image_id2.id);
      const Camera& camera2 = database_cache.Camera(image2.CameraId());
      const size_t camera_idx2 = maybe_add_camera(rig2, camera2);

      database_cache.CorrespondenceGraph()->ExtractMatchesBetweenImages(
          image_id1.id, image_id2.id, matches);
      for (const auto& match : matches) {
        points2D1.push_back(image1.Point2D(match.point2D_idx1).xy);
        points2D2.push_back(image2.Point2D(match.point2D_idx2).xy);
        camera_idxs1.push_back(camera_idx1);
        camera_idxs2.push_back(camera_idx2);
      }
    }
  }

  RANSACOptions ransac_options;
  ransac_options.min_num_trials = 30;
  ransac_options.random_seed = options.random_seed;
  ransac_options.max_error = options.init_max_error;

  std::optional<Rigid3d> maybe_rig2_from_rig1;
  std::optional<Rigid3d> maybe_pano2_from_pano1;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateGeneralizedRelativePose(ransac_options,
                                       points2D1,
                                       points2D2,
                                       camera_idxs1,
                                       camera_idxs2,
                                       cams_from_rig,
                                       cameras,
                                       &maybe_rig2_from_rig1,
                                       &maybe_pano2_from_pano1,
                                       &num_inliers,
                                       &inlier_mask)) {
    return false;
  }

  VLOG(3) << "Initial general frame pair with " << num_inliers
          << " inlier matches";

  // Note that we already checked for stable geometry (i.e., non-forward
  // motion, sufficient triangulation angle) between the original image pair.
  if (static_cast<int>(num_inliers) < options.init_min_num_inliers) {
    return false;
  }

  const Rigid3d rig2_from_rig1 = maybe_rig2_from_rig1.has_value()
                                     ? maybe_rig2_from_rig1.value()
                                     : maybe_pano2_from_pano1.value();

  // Recompose the relative transformation between the original images.

  const sensor_t orig_camera_id1(SensorType::CAMERA, orig_image1.CameraId());
  Rigid3d orig_cam1_from_rig1;
  if (!rig1.IsRefSensor(orig_camera_id1)) {
    orig_cam1_from_rig1 = rig1.SensorFromRig(orig_camera_id1);
  }

  const sensor_t orig_camera_id2(SensorType::CAMERA, orig_image2.CameraId());
  Rigid3d orig_cam2_from_rig2;
  if (!rig2.IsRefSensor(orig_camera_id2)) {
    orig_cam2_from_rig2 = rig2.SensorFromRig(orig_camera_id2);
  }

  orig_cam2_from_orig_cam1 =
      orig_cam2_from_rig2 * rig2_from_rig1 * Inverse(orig_cam1_from_rig1);

  return true;
}

}  // namespace

std::optional<IncrementalMapperImpl::InitInfo>
IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
    const IncrementalMapper::Options& options,
    const DatabaseCache& database_cache,
    const image_t image_id1,
    const image_t image_id2) {
  const Image& image1 = database_cache.Image(image_id1);
  const Image& image2 = database_cache.Image(image_id2);
  const Camera& camera1 = database_cache.Camera(image1.CameraId());
  const Camera& camera2 = database_cache.Camera(image2.CameraId());

  FeatureMatches matches;
  database_cache.CorrespondenceGraph()->ExtractMatchesBetweenImages(
      image_id1, image_id2, matches);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.xy);
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.xy);
  }

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry_options.ransac_options.random_seed = options.random_seed;
  // Delegate estimator selection to the general two-view geometry estimator so
  // it lives in a single place. The relative pose is recovered separately
  // below.
  TwoViewGeometry two_view_geometry = EstimateTwoViewGeometry(
      camera1, points1, camera2, points2, matches, two_view_geometry_options);

  if (!EstimateTwoViewGeometryPose(
          camera1, points1, camera2, points2, &two_view_geometry)) {
    return std::nullopt;
  }

  VLOG(3) << "Initial image pair with config " << two_view_geometry.config
          << ", " << two_view_geometry.inlier_matches.size()
          << " inlier matches, "
          << two_view_geometry.cam2_from_cam1->translation().z()
          << " z translation, " << RadToDeg(two_view_geometry.tri_angle)
          << " deg triangulation angle";

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) <
          options.init_min_num_inliers ||
      std::abs(two_view_geometry.cam2_from_cam1->translation().z()) >=
          options.init_max_forward_motion ||
      two_view_geometry.tri_angle <= DegToRad(options.init_min_tri_angle)) {
    return std::nullopt;
  }

  const Frame& frame1 = database_cache.Frame(image1.FrameId());
  const Frame& frame2 = database_cache.Frame(image2.FrameId());
  const Rig& rig1 = database_cache.Rig(frame1.RigId());
  const Rig& rig2 = database_cache.Rig(frame2.RigId());

  InitInfo info;
  info.image_id1 = image_id1;
  info.image_id2 = image_id2;

  // If one or both of the frames are non-trivial, initialize using
  // generalized relative pose solver. Note that we intentionally do this
  // after ensuring that the given image pair has stable two-view geometry.
  if (rig1.NumSensors() > 1 || rig2.NumSensors() > 1) {
    if (!EstimateInitialGeneralizedTwoViewGeometry(options,
                                                   database_cache,
                                                   image1,
                                                   image2,
                                                   frame1,
                                                   frame2,
                                                   rig1,
                                                   rig2,
                                                   info.cam2_from_cam1)) {
      return std::nullopt;
    }
    // The generalized solver does not recover intrinsics.
    return info;
  }

  info.cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;

