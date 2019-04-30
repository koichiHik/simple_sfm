
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
      common::vec1d<cv::Point3f>& point3d);

bool validate_triangulated_points_via_reprojection(
      const common::vec1d<cv::Point3f>& point3d,
      const cv::Matx34d& P,
      std::vector<uint8_t>& status,
      double point_ratio_in_front_of_cam = POINT_RATIO_IN_FRONT_CAM);

bool triangulate_points_and_validate(
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const common::CamIntrinsics& cam_intr,
      const cv::Matx34d& Porigin,
      const cv::Matx34d& P,
      common::vec1d<cv::Point3f>& point3d,
      double reproj_error_thresh = TRI_REPROJ_ERR_THRESH,
      double point_ratio_in_front_of_cam = POINT_RATIO_IN_FRONT_CAM);

bool find_camera_matrix(
      const common::CamIntrinsics& cam_intr,
      const common::vec1d<cv::Point2f>& img_point_set1,
      const common::vec1d<cv::Point2f>& img_point_set2,
      const cv::Matx34d& Porigin,
      cv::Matx34d& P,
      common::vec1d<cv::Point3f>& point3d);

}
}

#endif // GEOMETRY_H