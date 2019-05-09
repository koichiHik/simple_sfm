
#ifndef GEOMETRY_H
#define GEOMETRY_H

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace geometry {

// Fundamental Matrix Calculation
static const float FMAT_REPROJ_ERR_SCALING = 0.00006;
static const float FMAT_RANSAC_CONFIDENCE = 0.99;
static const int FMAT_MINIMUM_INLIERS = 100;

// Triangulation Validation Check.
static const float TRI_REPROJ_ERR_THRESH = 100.0;
static const float POINT_RATIO_IN_FRONT_CAM = 0.75;

// Triangulation Iteration Termination
static const float TRI_ITERATIVE_TERM =0.0001;


bool decompose_E_to_R_T(
      const cv::Matx33d& E,
      cv::Matx33d& R1,
      cv::Matx33d& R2,
      cv::Matx31d& T1,
      cv::Matx31d& T2);

double triangulate_points(
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& P1,
      const cv::Matx34d& P2,
      common::vec1d<common::Point3dWithRepError>& point3d_w_err);

cv::Matx41d iterative_linear_ls_triangulation(
      const cv::Point3d& norm_pnt1,
      const cv::Matx34d& P1,
      const cv::Point3d& norm_pnt2,
      const cv::Matx34d& P2);

bool validate_triangulated_points_via_reprojection(
      const common::vec1d<common::CloudPoint>& cloud_point,
      const cv::Matx34d& P,
      std::vector<uint8_t>& status,
      double point_ratio_in_front_of_cam = POINT_RATIO_IN_FRONT_CAM);

bool triangulate_points_and_validate(
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& Porigin,
      const cv::Matx34d& P,
      common::vec1d<common::Point3dWithRepError>& point3d_w_err,
      double reproj_error_thresh = TRI_REPROJ_ERR_THRESH,
      double point_ratio_in_front_of_cam = POINT_RATIO_IN_FRONT_CAM);

bool find_camera_matrix_via_pnp(
      const common::CamIntrinsics& cam_intr,  
      const common::vec1d<cv::Point2f>& point2d_list,
      const common::vec1d<cv::Point3f>& point3d_list,
      cv::Matx34d& P);

bool find_camera_matrix(
      const common::CamIntrinsics& cam_intr,
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const cv::Matx34d& Porigin,
      cv::Matx34d& P,
      common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list);

}
}

#endif // GEOMETRY_H