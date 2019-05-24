
#ifndef PCL_DRAWER_H
#define PCL_DRAWER_H

// Eigen
#include <Eigen/Eigen>

// PCL
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace visualization {

void draw_point_clouds(const common::vec1d<cv::Point3d>& point_cloud_dists,
                       const common::vec1d<cv::Vec3b>& point_cloud_clrs,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr& point_cloud);

void draw_camera_poses(
    const common::vec1d<cv::Matx34d>& cam_poses,
    common::vec1d<std::pair<std::string, pcl::PolygonMesh> >& cam_meshes,
    common::vec1d<
        std::pair<std::string, std::vector<Eigen::Matrix<float, 6, 1> > > >&
        lines_to_draw);

void create_rgb_vector_from_point_cloud(
    const common::vec2d<cv::KeyPoint>& key_points,
    const common::vec1d<cv::Mat>& original_imgs,
    const common::vec1d<common::CloudPoint>& point_cloud,
    common::vec1d<cv::Vec3b>& rgb_vecs);

}  // namespace visualization
}  // namespace simple_sfm

#endif  // PCL_DRAWER_H