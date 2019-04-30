
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// Original
#include <common/math.h>
#include <common/debug_helper.h>
#include <geometry/Geometry.h>

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
  cv::findFundamentalMat(
    img_point_set1, img_point_set2, cv::FM_RANSAC, 
    reproj_err_scaling * maxVal, ransac_confidence, status);

  for(int i=0; i<status.size(); i++) {
    if (status[i]) {
      inlier_set1.push_back(img_point_set1[i]);
      inlier_set2.push_back(img_point_set2[i]);
    }
  }

  // If num of inlier is less than thresh, return false.
  return status.size() > inlier_thresh;
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

  cv::Matx33d  W(0,-1,0,1,0.0,0,0,1);
  R1 = cv::Mat(svd.u * cv::Mat(W) * svd.vt);
  T1 = cv::Mat(svd.u.col(2));

  cv::Matx33d Wt(0,1,0,-1,0.0,0,0,1);
  R2 = cv::Mat(svd.u * cv::Mat(Wt) * svd.vt);
  T2 = cv::Mat(-svd.u.col(2));
  return true;
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
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& P1,
      const cv::Matx34d& P2,
      common::vec1d<cv::Point3f>& point_cloud) {

  return 0.0;
}

bool validate_triangulated_points_via_reprojection(
      const common::vec1d<cv::Point3f>& point3d,
      const cv::Matx34d& P,
      std::vector<uint8_t>& status,
      double point_ratio_in_front_of_cam) {

  cv::Matx44d P4x4 = cv::Matx44d::eye();
  common::vec1d<cv::Point3f> point3d_projected;
  cv::perspectiveTransform(point3d, point3d_projected, P4x4);

  // If reprojected point is in front of camera, valid.
  status.resize(point3d.size());
  for (int i = 0; i < point3d_projected.size(); i++) {
    status[i] = (point3d_projected[i].z > 0) ? 1 : 0;
  }

  int valid_count = cv::countNonZero(status);
  double front_point_ratio = ((double)valid_count / point3d.size());
  return front_point_ratio > point_ratio_in_front_of_cam;
}

bool triangulate_points_and_validate(
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& Porigin,
      const cv::Matx34d& P,
      common::vec1d<cv::Point3f>& point_cloud,
      double reproj_error_thresh,
      double point_ratio_in_front_of_cam) { 

  // Rotaion Matrix Validity Check.
  {
    cv::Matx33d R;
    cv::Matx31d T;
    decompose_P_to_R_T(P, R, T);
    bool result = common::math::check_R_validity(R);
    if (!result) {
      return false;
    }
  }

  double reproj_error1 
    = triangulate_points(img_point_set1,
                         img_point_set2,
                         cam_intr,
                         Porigin,
                         P,
                         point_cloud);

  common::vec1d<cv::Point3f> dummy_point_cloud;
  double reproj_error2
    = triangulate_points(img_point_set2,
                         img_point_set1,
                         cam_intr,
                         P,
                         Porigin,
                         dummy_point_cloud);    

  common::vec1d<uint8_t> status1, status2;
  bool validity1 = validate_triangulated_points_via_reprojection(
                        point_cloud, Porigin, status1, 
                        point_ratio_in_front_of_cam);
  
  bool validity2 = validate_triangulated_points_via_reprojection(
                        dummy_point_cloud, P, status2, 
                        point_ratio_in_front_of_cam);

  if (!validity1 || !validity2 ||
      reproj_error1 > reproj_error_thresh ||
      reproj_error2 > reproj_error_thresh) {
    return false;
  }
  return true;
}

bool find_camera_matrix(
      const common::CamIntrinsics& cam_intr,
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const cv::Matx34d& Porigin,
      cv::Matx34d& P,
      common::vec1d<cv::Point3f>& point3d) {

  // Initialize Output.
  P = 0;
  point3d.clear();

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
    if (!fmat_result) {
      return false;
    }
  }

  // Essential Matrix Calculation.
  cv::Matx33d E;
  {
    bool emat_result = calc_essential_matrix(
                          F, cam_intr, E);
    if (!emat_result) {
      return false;
    }
  }

  // Decompose E matrix to R and T
  cv::Matx33d R1, R2;
  cv::Matx31d T1, T2;
  {
    bool e_decomp = decompose_E_to_R_T(E, R1, R2, T1, T2);
    if (!e_decomp) {
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
    common::vec1d<cv::Point3f> tmp_point3d;
    bool result = triangulate_points_and_validate(
                    img_point_set1, img_point_set2, cam_intr, 
                    Porigin, *citr, tmp_point3d);

    if (result) {
      P = *citr;
      point3d.resize(tmp_point3d.size());
      std::copy(tmp_point3d.begin(), tmp_point3d.end(), point3d.begin());
      return true;
    }
  }
  return false;
}

}
}