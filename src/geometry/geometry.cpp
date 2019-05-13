
// STL
#include <algorithm>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>

// Original
#include <common/math.h>
#include <common/debug_helper.h>
#include <common/container_util.h>
#include <geometry/geometry.h>

namespace {

const double INLIER_THRESH_FOR_PNP_REPROJ = 10.0;
const double INLIER_RATIO_THRESH_FOR_PNP_REPROJ = 0.2;
const double INLIER_RATIO_FOR_PNP_RANSAC = 0.7;

}

namespace {

bool calc_fundamental_matrix(
      const simple_sfm::common::vec1d<cv::Point2f>& img_point_set1,
      const simple_sfm::common::vec1d<cv::Point2f>& img_point_set2,
      cv::Matx33d& F,
      simple_sfm::common::vec1d<cv::Point2f>& inlier_set1,
      simple_sfm::common::vec1d<cv::Point2f>& inlier_set2,
      double reproj_err_scaling, 
      double ransac_confidence,
      int inlier_thresh) {

  simple_sfm::common::vec1d<uint8_t> status;
  double minVal, maxVal;
  cv::minMaxIdx(img_point_set1, &minVal, &maxVal);
  F = cv::findFundamentalMat(
    img_point_set1, img_point_set2, cv::FM_RANSAC, 
    reproj_err_scaling * maxVal, ransac_confidence, status);

  for(int i=0; i<status.size(); i++) {
    if (status[i]) {
      inlier_set1.push_back(img_point_set1[i]);
      inlier_set2.push_back(img_point_set2[i]);
    }
  }
  // If num of inlier is less than thresh, return false.
  return cv::countNonZero(status) > inlier_thresh;
}

bool calc_essential_matrix(
    const cv::Matx33d& F, 
    const simple_sfm::common::CamIntrinsics& cam_intr,
    cv::Matx33d& E) {
  
  E = cam_intr.K.t() * F * cam_intr.K;

  return simple_sfm::common::math::check_E_validity(E);
}

cv::Matx34d create_camera_mat_from_RT(
      const cv::Matx33d R,
      const cv::Matx31d T) {
  return cv::Matx34d(R(0,0), R(0,1), R(0,2), T(0,0),
                     R(1,0), R(1,1), R(1,2), T(1,0),
                     R(2,0), R(2,1), R(2,2), T(2,0));
}

bool decompose_P_to_R_T(
      const cv::Matx34d& P,
      cv::Matx33d& R,
      cv::Matx31d& T) {

  R = {P(0,0), P(0,1), P(0,2),
       P(1,0), P(1,1), P(1,2),
       P(2,0), P(2,1), P(2,2)};
  T = {P(0,3), P(1,3), P(2,3)};

  return true;
}

bool compose_R_T_to_P(
      const cv::Matx33d& R,
      const cv::Matx31d& T,
      cv::Matx34d& P
) {

  P(0,0) = R(0,0); P(0,1) = R(0,1); P(0,2) = R(0,2); P(0,3) = T(0,0); 
  P(1,0) = R(1,0); P(1,1) = R(1,1); P(1,2) = R(1,2); P(1,3) = T(1,0); 
  P(2,0) = R(2,0); P(2,1) = R(2,1); P(2,2) = R(2,2); P(2,3) = T(2,0); 

  return true;
}

bool decompose_E_to_R_T_internal(
      const cv::Matx33d& E,
      cv::Matx33d& R1,
      cv::Matx33d& R2,
      cv::Matx31d& T1,
      cv::Matx31d& T2) {

  // Use OpenCV SVD.
  cv::SVD svd(E, cv::SVD::MODIFY_A);

  // Result check.
  double singular_value_ratio = 
      std::abs(svd.w.at<double>(0) / svd.w.at<double>(1));
  // If two singular values are too far apart, this caluclation fails.
  if (singular_value_ratio < 0.7 || 1.5 < singular_value_ratio) { 
    return false;
  }

  cv::Matx33d  W(0,-1,0,1,0,0,0,0,1);
  R1 = cv::Mat(svd.u * cv::Mat(W) * svd.vt);
  T1 = cv::Mat(svd.u.col(2));

  cv::Matx33d Wt(0,1,0,-1,0,0,0,0,1);
  R2 = cv::Mat(svd.u * cv::Mat(Wt) * svd.vt);
  T2 = cv::Mat(-svd.u.col(2));
  return true;
}

cv::Point3d normalize_point_in_pix_coord(
  const cv::Point2f& pix_pnt,
  const cv::Matx33d& Kinv) {
  
  // Homogeneous & Normalize
  cv::Matx31d normed_pnt = 
    Kinv * cv::Matx31d(pix_pnt.x, pix_pnt.y, 1.0);
    
  return cv::Point3d(normed_pnt(0), normed_pnt(1), normed_pnt(2));
}

void build_homogeneous_eqn_system_for_triangulation(
  const cv::Point3d& norm_p1,
  const cv::Matx34d& P1,
  const cv::Point3d& norm_p2,
  const cv::Matx34d& P2,
  double w1,
  double w2,
  cv::Matx43d& A,
  cv::Matx41d& B) {

	cv::Matx43d A_(
    (norm_p1.x*P1(2,0)-P1(0,0))/w1, (norm_p1.x*P1(2,1)-P1(0,1))/w1,	(norm_p1.x*P1(2,2)-P1(0,2))/w1,
		(norm_p1.y*P1(2,0)-P1(1,0))/w1, (norm_p1.y*P1(2,1)-P1(1,1))/w1,	(norm_p1.y*P1(2,2)-P1(1,2))/w1,		
		(norm_p2.x*P2(2,0)-P2(0,0))/w2, (norm_p2.x*P2(2,1)-P2(0,1))/w2,	(norm_p2.x*P2(2,2)-P2(0,2))/w2,	
		(norm_p2.y*P2(2,0)-P2(1,0))/w2, (norm_p2.y*P2(2,1)-P2(1,1))/w2,	(norm_p2.y*P2(2,2)-P2(1,2))/w2);

	cv::Matx41d B_(
    -(norm_p1.x*P1(2,3)-P1(0,3))/w1,
		-(norm_p1.y*P1(2,3)-P1(1,3))/w1,
		-(norm_p2.x*P2(2,3)-P2(0,3))/w2,
		-(norm_p2.y*P2(2,3)-P2(1,3))/w2);
  
  A = A_;
  B = B_;
}

void solve_linear_eqns(
  const cv::Matx43d& A, 
  const cv::Matx41d& B,
  cv::Matx41d& X) {

    cv::Matx31d tmp_X;
    tmp_X(0) = X(0);
    tmp_X(1) = X(1);
    tmp_X(2) = X(2);
    cv::solve(A, B, tmp_X, cv::DECOMP_SVD);
    X(0) = tmp_X(0);
    X(1) = tmp_X(1);
    X(2) = tmp_X(2);
    X(3) = 1.0;
}

}


namespace simple_sfm {
namespace geometry {

bool decompose_E_to_R_T(
      const cv::Matx33d& E,
      cv::Matx33d& R1,
      cv::Matx33d& R2,
      cv::Matx31d& T1,
      cv::Matx31d& T2) {
 
  cv::Matx33d tmpE = E;
  bool result = decompose_E_to_R_T_internal(tmpE, R1, R2, T1, T2);
  // TODO Remove magic number and extract to const.
  if (cv::determinant(R1) + 1.0 < 1e-09) {
    tmpE = -tmpE;
    result = decompose_E_to_R_T_internal(tmpE, R1, R2, T1, T2);
  }

  if (!result) {
    common::debug_helper::print_debug_info(
      __FILE__, __LINE__, __FUNCTION__, "Decompose E failed.");
    return false;
  }

  return true;
}

double triangulate_points(
      const common::vec1d<cv::Point2f>& img_point_set,
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& Porigin,
      const cv::Matx34d& P1,
      common::vec1d<common::Point3dWithRepError>& point3d_w_err) {

  assert(img_point_set.size() == img_point_set1.size());
  point3d_w_err.clear();

  cv::Matx34d KP1 = cam_intr.K * P1;
  common::vec1d<double> reproj_errors;
  
  // Triangulation.
  for (int i = 0; i < img_point_set1.size(); i++) {

    cv::Point3d norm_pnt 
      = normalize_point_in_pix_coord(img_point_set[i], cam_intr.Kinv);
    cv::Point3d norm_pnt1 
      = normalize_point_in_pix_coord(img_point_set1[i], cam_intr.Kinv);
    cv::Matx41d X = 
      iterative_linear_ls_triangulation(norm_pnt, Porigin, norm_pnt1, P1);

    // Reproject
    {
      double reproj_err(0.0);
      cv::Matx31d rep_img_coord = KP1 * X;
      cv::Point2f rep_img_normalized(
        rep_img_coord(0) / rep_img_coord(2),
        rep_img_coord(1) / rep_img_coord(2));
      reproj_err = cv::norm(rep_img_normalized - img_point_set1[i]);
      reproj_errors.push_back(reproj_err);
      common::Point3dWithRepError pt_w_err;
      pt_w_err.coord = cv::Point3d(X(0), X(1), X(2));
      pt_w_err.reprojection_err = reproj_err;
      point3d_w_err.push_back(pt_w_err);
    }
  }
 
  double sum = std::accumulate(reproj_errors.begin(), reproj_errors.end(), 0.0); 
  double mean = sum / reproj_errors.size();
  return mean;
}

cv::Matx41d iterative_linear_ls_triangulation(
      const cv::Point3d& norm_pnt1,
      const cv::Matx34d& P1,
      const cv::Point3d& norm_pnt2,
      const cv::Matx34d& P2) {

  // Do once for initial value.
  double w1(1.0), w2(1.0);
  cv::Matx43d A;
  cv::Matx41d B, X;

  build_homogeneous_eqn_system_for_triangulation(
    norm_pnt1, P1, norm_pnt2, P2, w1, w2, A, B);

  solve_linear_eqns(A, B, X);

  // Iteratively refine triangulation.
  for (int i = 0; i < 10; i++) {

    // Calculate weight.
    double p2x1 = (P1.row(2) * X)(0);
    double p2x2 = (P2.row(2) * X)(0);

    if (std::abs(w1 - p2x1) < TRI_ITERATIVE_TERM &&
        std::abs(w2 - p2x2) < TRI_ITERATIVE_TERM) {
      break;
    }

    w1 = p2x1;
    w2 = p2x2;

    build_homogeneous_eqn_system_for_triangulation(
      norm_pnt1, P1, norm_pnt2, P2, w1, w2, A, B);

    solve_linear_eqns(A, B, X);

  }

  return X;
}

bool validate_triangulated_points_via_reprojection(
      const common::vec1d<common::Point3dWithRepError>& point3d_w_err_list,
      const cv::Matx34d& P,
      std::vector<uint8_t>& status,
      double point_ratio_in_front_of_cam) {

  common::vec1d<cv::Point3d> point3d_projected(point3d_w_err_list.size());
  {
    cv::Matx44d P4x4 = cv::Matx44f::eye();
    for (int i = 0; i < 12; i++) {
      P4x4.val[i] = P.val[i];
    }
    common::vec1d<cv::Point3d> point3d_list;
    common::container_util::convert_point3d_w_reperr_list_to_point3d_list(
      point3d_w_err_list,
      point3d_list);

    cv::perspectiveTransform(point3d_list, point3d_projected, P4x4);
  }

  // If reprojected point is in front of camera, valid.
  status.resize(point3d_w_err_list.size());
  for (int i = 0; i < point3d_projected.size(); i++) {
    status[i] = (point3d_projected[i].z > 0) ? 1 : 0;
  }

  int valid_count = cv::countNonZero(status);
  double front_point_ratio = ((double)valid_count / point3d_w_err_list.size());
  return front_point_ratio > point_ratio_in_front_of_cam;
}

bool triangulate_points_and_validate(
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& Porigin,
      const cv::Matx34d& P,
      common::vec1d<common::Point3dWithRepError>& point3d_w_err,
      double reproj_error_thresh,
      double point_ratio_in_front_of_cam) { 

  // Rotaion Matrix Validity Check.
  {
    cv::Matx33d R;  
    cv::Matx31d T;
    decompose_P_to_R_T(P, R, T);
    bool result = common::math::check_R_validity(R);
    if (!result) {
      std::cout << "Failed at check_R_validity" << std::endl;
      std::cout << "R Mat : " << std::endl;
      std::cout << R << std::endl;
      return false;
    }
  }

  double reproj_error1 
    = triangulate_points(img_point_set1,
                         img_point_set2,
                         cam_intr,
                         Porigin,
                         P,
                         point3d_w_err);

  common::vec1d<common::Point3dWithRepError> dummy_point3d_w_err;
  double reproj_error2
    = triangulate_points(img_point_set2,
                         img_point_set1,
                         cam_intr,
                         P,
                         Porigin,
                         dummy_point3d_w_err);    

  common::vec1d<  uint8_t> status1, status2;
  bool validity1 = validate_triangulated_points_via_reprojection(
                        point3d_w_err, Porigin, status1, 
                        point_ratio_in_front_of_cam);
  
  bool validity2 = validate_triangulated_points_via_reprojection(
                        dummy_point3d_w_err, P, status2, 
                        point_ratio_in_front_of_cam);

  for (size_t idx = 0; idx < status1.size(); idx++) {
    point3d_w_err[idx].valid = (status1[idx] != 0 && status2[idx] != 0) ? true : false;
  }

  if (!validity1 || !validity2 ||
      reproj_error1 > reproj_error_thresh ||
      reproj_error2 > reproj_error_thresh) {

      std::cout << "Validity 1 : " << validity1 << std::endl;
      std::cout << "Validity 2 : " << validity2 << std::endl;

      std::cout << "Reprojection Error Thresh : " << reproj_error_thresh << std::endl;
      std::cout << "Reprojection Error 1 : " << reproj_error1 << std::endl;
      std::cout << "Reprojection Error 2 : " << reproj_error2 << std::endl;

    return false;
  }
  return true;
}

bool find_camera_matrix_via_pnp(
      const common::CamIntrinsics& cam_intr,  
      const common::vec1d<cv::Point2f>& point2d_list,
      const common::vec1d<cv::Point3f>& point3d_list,
      cv::Matx34d& P) {

  assert(point2d_list.size() == point3d_list.size());
  assert(7 < point2d_list.size() && 7 < point3d_list.size());

  cv::Matx33d R;
  cv::Matx31d T, rvec;

  // Solve PNP
  {
    // Vector to cv::Mat
    cv::Mat point3d_mat = cv::Mat(point3d_list).t();
    cv::Mat point2d_mat = cv::Mat(point2d_list).t();

    decompose_P_to_R_T(P, R, T);

    // Double to Float conversion.
    cv::Matx33f K32f = cam_intr.K;
    cv::Matx33f Kinv32f = cam_intr.Kinv;
    cv::Mat_<float> distortion_coeff32f = cam_intr.distortion_coeff;
    cv::Rodrigues(R, rvec);

    cv::solvePnPRansac(
      point3d_mat, point2d_mat, cam_intr.K, cam_intr.distortion_coeff,
      rvec, T, false, 100, 3.0f);

    cv::Rodrigues(rvec, R);
    compose_R_T_to_P(R, T, P);
  }

  // Inlier & Outlier Check
  common::vec1d<size_t> inliers;
  {
    // Project 3D points with calculated camera pose.
    common::vec1d<cv::Point2f> projected_point2d_list;
    cv::projectPoints(point3d_list, rvec, T, cam_intr.K, cam_intr.distortion_coeff, projected_point2d_list);
    for (size_t idx = 0; idx < projected_point2d_list.size(); idx++) {
      
      double reperr = cv::norm(projected_point2d_list[idx] - point2d_list[idx]);
      std::cout << reperr << std::endl;

      if (cv::norm(projected_point2d_list[idx] - point2d_list[idx]) < INLIER_THRESH_FOR_PNP_REPROJ) {
        inliers.push_back(idx);
      }
    }
  }

  if (inliers.size() < point2d_list.size() * INLIER_RATIO_THRESH_FOR_PNP_REPROJ) {
    common::debug_helper::print_debug_info(
      __FILE__, __LINE__, __FUNCTION__, "Not enough inliers to consider goot pose.");
    return false;
  }

  if (200.0 < cv::norm(T)) {
    common::debug_helper::print_debug_info(
      __FILE__, __LINE__, __FUNCTION__, "Estimated camera movement is too big.");
    return false;
  }

  if (!common::math::check_R_validity(R)) {
    common::debug_helper::print_debug_info(
      __FILE__, __LINE__, __FUNCTION__, "R it not valid.");
  }

  return true;
}

bool find_camera_matrix(
      const common::CamIntrinsics& cam_intr,
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const cv::Matx34d& Porigin,
      cv::Matx34d& P,
      common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list) {

  // Initialize Output.
  P = 0;
  point3d_w_reperr_list.clear();

  // Fundamental Matrix Calculation.
  cv::Matx33d F;
  common::vec1d<cv::Point2f> inlier_set1, inlier_set2;
  {
    bool fmat_result = calc_fundamental_matrix(
                          img_point_set1, 
                          img_point_set2, 
                          F,
                          inlier_set1, 
                          inlier_set2, 
                          FMAT_REPROJ_ERR_SCALING, 
                          FMAT_RANSAC_CONFIDENCE,
                          FMAT_MINIMUM_INLIERS);

    std::cout << std::endl;
    std::cout << "Calculated F : " << std::endl;
    std::cout << F << std::endl;

    if (!fmat_result) {
      std::cout << "Failed at calc_fundamental_matrix" << std::endl;
      return false;
    }
  }

  // Essential Matrix Calculation.
  cv::Matx33d E;
  {
    bool emat_result = calc_essential_matrix(
                          F, cam_intr, E);

    std::cout << std::endl;
    std::cout << "Calculated E : " << std::endl;
    std::cout << E << std::endl;

    if (!emat_result) {
      std::cout << "Failed at calc_essential_matrix" << std::endl;
      return false;
    }
  }

  // Decompose E matrix to R and T
  cv::Matx33d R1, R2;
  cv::Matx31d T1, T2;
  {
    bool e_decomp = decompose_E_to_R_T(E, R1, R2, T1, T2);
    if (!e_decomp) {
      std::cout << "Failed at decompose_E_to_R_T" << std::endl;
      return false;
    }
  }

  // Try triangulate with 4 Possible Configurations.
  common::vec1d<cv::Matx34d> configs;
  configs.push_back(create_camera_mat_from_RT(R1, T1));
  configs.push_back(create_camera_mat_from_RT(R1, T2));
  configs.push_back(create_camera_mat_from_RT(R2, T1));
  configs.push_back(create_camera_mat_from_RT(R2, T2));
  for (common::vec1d<cv::Matx34d>::const_iterator citr = configs.cbegin();
       citr != configs.cend();
       citr++) {
    common::vec1d<common::Point3dWithRepError> tmp_point3d_w_reperr;
    bool result = triangulate_points_and_validate(
                    img_point_set1, img_point_set2, cam_intr, 
                    Porigin, *citr, tmp_point3d_w_reperr);


    if (result) {
      P = *citr;
      point3d_w_reperr_list.resize(tmp_point3d_w_reperr.size());
      std::copy(tmp_point3d_w_reperr.begin(), tmp_point3d_w_reperr.end(), point3d_w_reperr_list.begin());
      std::cout << std::endl << "Found valid configuration!" << std::endl;
      return true;
    } else {
      std::cout << "Failed at triangulate_points_and_validate" << std::endl;
    }
  }
  return false;
}

}
}