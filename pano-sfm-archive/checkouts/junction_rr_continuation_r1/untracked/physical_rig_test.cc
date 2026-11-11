#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/physical_rig.h"
#include <iostream>

int main() {
  using namespace colmap;
  Rigid3d truth, guess, g;
  truth.rotation() = Eigen::AngleAxisd(3.12, Eigen::Vector3d::UnitY());
  truth.translation() = Eigen::Vector3d(.004, -.001, .002);
  guess = truth;
  guess.rotation() = Eigen::AngleAxisd(.03, Eigen::Vector3d::UnitX()) * truth.rotation();
  guess.translation() += Eigen::Vector3d(.01, -.02, .01);
  g.translation() = Eigen::Vector3d(.1, -.2, .3);
  double k[4] = {875.259422762, 875.259422762, 1250, 1250};
  std::vector<Eigen::Vector3d> points;
  points.reserve(150);
  ceres::Problem problem;
  double maximum_jacobian_error = 0;
  for (int view = -1; view <= 1; ++view) {
    Rigid3d a;
    a.rotation() = Eigen::AngleAxisd(view * .7853981633974483, Eigen::Vector3d::UnitY());
    const Rigid3d camera = a * truth * g;
    for (int i = 0; i < 50; ++i) {
      Eigen::Vector3d in_view((i%7-3)*.18, (i%11-5)*.09, 2.+.11*i);
      points.push_back(Inverse(camera) * in_view);
      Eigen::Vector2d xy(k[0]*in_view.x()/in_view.z()+k[2], k[1]*in_view.y()/in_view.z()+k[3]);
      auto* cost = PhysicalRigReprojError::Create(xy, a);
      if (i == 0) {
        double const* blocks[4] = {points.back().data(), guess.params.data(), g.params.data(), k};
        double residual[2], j0[6], j1[14], j2[14], j3[8];
        double* jac[4] = {j0,j1,j2,j3};
        THROW_CHECK(cost->Evaluate(blocks, residual, jac));
        for (int block=0; block<4; ++block) {
          const int dim = cost->parameter_block_sizes()[block];
          for (int column=0; column<dim; ++column) {
            double* x = const_cast<double*>(blocks[block]);
            double old=x[column], plus[2], minus[2];
            x[column]=old+1e-6;cost->Evaluate(blocks,plus,nullptr);
            x[column]=old-1e-6;cost->Evaluate(blocks,minus,nullptr);x[column]=old;
            for (int row=0; row<2; ++row) maximum_jacobian_error=std::max(maximum_jacobian_error,std::abs((plus[row]-minus[row])/2e-6-jac[block][row*dim+column]));
          }
        }
      }
      problem.AddResidualBlock(cost,nullptr,points.back().data(),guess.params.data(),g.params.data(),k);
      problem.SetParameterBlockConstant(points.back().data());
    }
  }
  THROW_CHECK_LT(maximum_jacobian_error, 1e-4);
  problem.SetParameterBlockConstant(g.params.data());problem.SetParameterBlockConstant(k);
  SetManifold(&problem,guess.params.data(),CreateProductManifold(CreateEigenQuaternionManifold(),CreateEuclideanManifold<3>()));
  ceres::Solver::Options options;options.linear_solver_type=ceres::DENSE_QR;options.max_num_iterations=100;
  options.function_tolerance=1e-13;options.gradient_tolerance=1e-13;options.parameter_tolerance=1e-13;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  THROW_CHECK(summary.IsSolutionUsable());
  THROW_CHECK_LT(guess.rotation().angularDistance(truth.rotation()),1e-8);
  THROW_CHECK_LT((guess.translation()-truth.translation()).norm(),1e-8);
  std::cout << "SYNTHETIC_SHARED_B_PASS jacobian_max=" << maximum_jacobian_error << " final_cost=" << summary.final_cost << "\n";
}
