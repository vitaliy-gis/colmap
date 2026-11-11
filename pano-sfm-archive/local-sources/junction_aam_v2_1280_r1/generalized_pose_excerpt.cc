  *rig_from_world = report.model;
  *num_inliers = report.support.num_unique_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return true;
}

bool EstimateGeneralizedRelativePose(
    const RANSACOptions& ransac_options,
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const std::vector<size_t>& camera_idxs1,
    const std::vector<size_t>& camera_idxs2,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    std::optional<Rigid3d>* rig2_from_rig1,
    std::optional<Rigid3d>* pano2_from_pano1,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  ThrowCheckCameras(camera_idxs1, cams_from_rig, cameras);
  ThrowCheckCameras(camera_idxs2, cams_from_rig, cameras);
  ransac_options.Check();

  const size_t num_points = points2D1.size();
  if (num_points == 0) {
    return false;
  }

  // Both branches below score with the pixel-unit tangent Sampson error, so the
  // RANSAC threshold is the plain pixel ransac_options throughout. No
  // per-camera conversion to normalized/angular units is needed.
  if (IsPanoramicRig(camera_idxs1, cams_from_rig) &&
      IsPanoramicRig(camera_idxs2, cams_from_rig)) {
    Rigid3d cam2_from_cam1;
    // EstimateRelativePose treats the panoramic rig as one central camera, so
    // each ray carries its unprojection Jacobian, rotated into the rig frame by
    // the same rotation as the ray. Unprojectable points are zeroed, which the
    // tangent Sampson residual reports as infinite (rejected).
    std::vector<CamRayWithJac> cam_rays1_with_jac(num_points);
    std::vector<CamRayWithJac> cam_rays2_with_jac(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const size_t camera_idx1 = camera_idxs1[i];
      const Eigen::Matrix3d rig_from_cam1 =
          cams_from_rig[camera_idx1].rotation().inverse().toRotationMatrix();
      if (const auto rj =
              cameras[camera_idx1].CamRayFromImgWithJac(points2D1[i])) {
        cam_rays1_with_jac[i] = {rig_from_cam1 * rj->ray,
                                 rig_from_cam1 * rj->jacobian};
      } else {
        cam_rays1_with_jac[i] = CamRayWithJac::Zero();
      }

      const size_t camera_idx2 = camera_idxs2[i];
      const Eigen::Matrix3d rig_from_cam2 =
          cams_from_rig[camera_idx2].rotation().inverse().toRotationMatrix();
      if (const auto rj =
              cameras[camera_idx2].CamRayFromImgWithJac(points2D2[i])) {
        cam_rays2_with_jac[i] = {rig_from_cam2 * rj->ray,
                                 rig_from_cam2 * rj->jacobian};
      } else {
        cam_rays2_with_jac[i] = CamRayWithJac::Zero();
      }
    }
    if (EstimateRelativePose(ransac_options,
                             cam_rays1_with_jac,
                             cam_rays2_with_jac,
                             &cam2_from_cam1,
                             num_inliers,
                             inlier_mask)) {
      *pano2_from_pano1 = cam2_from_cam1;
      return true;
    }
    return false;
  }

  std::vector<GRNPObservation> points1(num_points);
  std::vector<GRNPObservation> points2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points1[i] = {cams_from_rig[camera_idxs1[i]],
                  cameras[camera_idxs1[i]]
                      .CamRayFromImgWithJac(points2D1[i])
                      .value_or(CamRayWithJac::Zero())};
    points2[i] = {cams_from_rig[camera_idxs2[i]],
                  cameras[camera_idxs2[i]]
                      .CamRayFromImgWithJac(points2D2[i])
                      .value_or(CamRayWithJac::Zero())};
  }

  LORANSAC<GR6PEstimator, GR8PEstimator> ransac(ransac_options);
  auto report = ransac.Estimate(points1, points2);
  if (!report.success) {
    return false;
  }

  *rig2_from_rig1 = report.model;
  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return true;
}

bool RefineGeneralizedAbsolutePose(const AbsolutePoseRefinementOptions& options,
                                   const std::vector<char>& inlier_mask,
                                   const std::vector<Eigen::Vector2d>& points2D,
                                   const std::vector<Eigen::Vector3d>& points3D,
                                   const std::vector<size_t>& camera_idxs,
                                   const std::vector<Rigid3d>& cams_from_rig,
                                   Rigid3d* rig_from_world,
                                   std::vector<Camera>* cameras,
                                   Eigen::Matrix6d* rig_from_world_cov) {
  THROW_CHECK_EQ(points2D.size(), inlier_mask.size());
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras->size());
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras->size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  std::vector<double*> cameras_params_data(cameras->size());
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data[i] = cameras->at(i).params.data();
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);

  // Cost function assumes unit quaternion.
  rig_from_world->rotation().normalize();

  std::vector<Eigen::Vector3d> point3D_params = points3D;
  std::vector<Eigen::Vector7d> cam_from_rig_params(cams_from_rig.size());
  for (size_t i = 0; i < cams_from_rig.size(); ++i) {
    cam_from_rig_params[i] = cams_from_rig[i].params;
  }

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    const size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    problem.AddResidualBlock(
        CreateCameraCostFunction<RigReprojErrorCostFunctor>(
            cameras->at(camera_idx).model_id, points2D[i]),
        loss_function.get(),
        point3D_params[i].data(),
        cam_from_rig_params[camera_idx].data(),
        rig_from_world->params.data(),
        cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(point3D_params[i].data());
  }

  if (options.use_position_prior) {
    problem.AddResidualBlock(
        CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
            Create(options.position_prior_covariance,
                   options.position_prior_in_world),
        nullptr,
        rig_from_world->params.data());
  }

  if (problem.NumResiduals() > 0) {
    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

