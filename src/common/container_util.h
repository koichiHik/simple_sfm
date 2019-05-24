
#ifndef CONTAINER_UTIL_H
#define CONTAINER_UTIL_H

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace common {
namespace container_util {

void convert_key_point_list_to_point2f_list(
    const common::vec1d<cv::KeyPoint>& key_point_list,
    common::vec1d<cv::Point2f>& point2f_list);

void convert_cloud_point_list_to_point3f_list(
    const common::vec1d<common::CloudPoint>& cloud_point_list,
    common::vec1d<cv::Point3f>& point3f_list);

void convert_cloud_point_list_to_point3d_list(
    const common::vec1d<common::CloudPoint>& cloud_point_list,
    common::vec1d<cv::Point3d>& point3d_list);

void convert_point3d_w_reperr_list_to_point3f_list(
    const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
    common::vec1d<cv::Point3f>& point3f_list);

void convert_point3d_w_reperr_list_to_point3d_list(
    const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
    common::vec1d<cv::Point3d>& point3d_list);

void convert_point3d_w_reperr_list_to_cloud_point_list(
    int image_num,
    const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
    common::vec1d<common::CloudPoint>& cloud_point_list);

void create_key_point_list_aligned_with_matches(
    const common::vec1d<cv::DMatch>& matches,
    const common::vec1d<cv::KeyPoint>& key_point_list_train,
    const common::vec1d<cv::KeyPoint>& key_point_list_query,
    common::vec1d<cv::KeyPoint>& aligned_key_point_list_train,
    common::vec1d<cv::KeyPoint>& aligned_key_point_list_query);

void create_point2f_list_aligned_with_matches(
    const common::vec1d<cv::DMatch>& matches,
    const common::vec1d<cv::KeyPoint>& key_point_list_train,
    const common::vec1d<cv::KeyPoint>& key_point_list_query,
    common::vec1d<cv::Point2f>& aligned_point2f_list_train,
    common::vec1d<cv::Point2f>& aligned_point2f_list_query);

void create_point2f_list_aligned_with_matches(
    const common::vec1d<cv::DMatch>& matches,
    const common::vec1d<cv::Point2f>& point2f_list_train,
    const common::vec1d<cv::Point2f>& point2f_list_query,
    common::vec1d<cv::Point2f>& aligned_point2f_list_train,
    common::vec1d<cv::Point2f>& aligned_point2f_list_query);

void ConvertCloudPointListToPCLPointXYZRGBAList(
    const common::vec2d<cv::KeyPoint>& key_points,
    const common::vec1d<cv::Mat>& original_imgs,
    const common::vec1d<CloudPoint>& cloud_points,
    common::vec1d<pcl::PointXYZRGBA>& pcl_points_xyz_rgb);

}  // namespace container_util
}  // namespace common
}  // namespace simple_sfm

#endif  // CONTAINER_UTIL_H
