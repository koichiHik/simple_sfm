
// System
#include <assert.h>

// STL
#include <deque>

// Eigen
#include <Eigen/Eigen>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/ros/conversions.h>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>
#include <visualization/pcl_drawer.h>

// Constants
namespace {

float CAM_SCALE_COEFF = 0.25;
float VIEWING_RAY_SCALE_COEFF = 3.0;
int	CAM_POLYGON[18] = {0,1,2, 0,3,1, 0,4,3, 0,2,4, 3,1,4, 2,4,1};
cv::Scalar CAM_COLOR(255, 0, 0);

}

namespace {


pcl::PointXYZRGB eigen_to_PointXYZRGB(
  Eigen::Vector3d v, 
  Eigen::Vector3d rgb) { 
  pcl::PointXYZRGB p(rgb[0],rgb[1],rgb[2]); 
  p.x = v[0]; p.y = v[1]; p.z = v[2]; 
  return p; 
}

Eigen::Matrix<float,6,1> eigen_2_eigen(
  Eigen::Vector3d v, 
  Eigen::Vector3d rgb) { 
  return (Eigen::Matrix<float,6,1>() << v[0],v[1],v[2],rgb[0],rgb[1],rgb[2]).finished(); 
}

simple_sfm::common::vec1d<Eigen::Matrix<float,6,1> > to_vector(
  const Eigen::Matrix<float,6,1>& p1, 
  const Eigen::Matrix<float,6,1>& p2) {
  std::vector<Eigen::Matrix<float,6,1> > v(2);
  v[0] = p1; v[1] = p2;
  return v; 
}

void populate_pcl_point_clouds(
  const simple_sfm::common::vec1d<cv::Point3d>& point_cloud_dists,
  const simple_sfm::common::vec1d<cv::Vec3b>& point_cloud_clrs,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& point_cloud) {

  assert(point_cloud_dists.size() == point_cloud_clrs.size());

  for (size_t idx = 0; idx < point_cloud_dists.size(); idx++) {
    // Color
    cv::Vec3b rgbv = point_cloud_clrs[idx];

    //std::cout << "RGB : " << (int)rgbv[2] << ", " << (int)rgbv[1] << ", " << (int)rgbv[0] << std::endl;

    // Position
    pcl::PointXYZRGB pclp(rgbv[2], rgbv[1], rgbv[0]);
    pclp.x = point_cloud_dists[idx].x;
    pclp.y = point_cloud_dists[idx].y;
    pclp.z = point_cloud_dists[idx].z;
    point_cloud->push_back(pclp);
  }
}

void compose_camera_element(
    const cv::Matx34d& pose, 
    const std::string& name,
    float r, float g, float b, 
    simple_sfm::common::vec1d<std::pair<std::string,pcl::PolygonMesh> >& cam_meshes,
    simple_sfm::common::vec1d<std::pair<std::string,std::vector<Eigen::Matrix<float,6,1> > > >& lines_to_draw,    
    double s) {

  // Confirm name is not empty.
  assert(name.length() > 0);

  Eigen::Matrix3d R;
  Eigen::Vector3d T, T_trans;
  R << pose(0,0), pose(0,1), pose(0,2), 
       pose(1,0), pose(1,1), pose(1,2), 
       pose(2,0), pose(2,1), pose(2,2);
  T << pose(0,3), pose(1,3), pose(2,3);

  T_trans = -R.transpose() * T;
  Eigen::Vector3d v_right, v_up, v_forward;
  v_right = R.row(0).normalized() * s;
  v_up = -R.row(1).normalized() * s;
  v_forward = R.row(2).normalized() * s;
  Eigen::Vector3d rgb(r, g, b);

  // Polygon Mesh
  {
    pcl::PointCloud<pcl::PointXYZRGB> mesh_cld;
    mesh_cld.push_back(eigen_to_PointXYZRGB(T_trans, rgb));
    mesh_cld.push_back(eigen_to_PointXYZRGB(T_trans + v_forward + v_right / 2.0 + v_up / 2.0, rgb));
    mesh_cld.push_back(eigen_to_PointXYZRGB(T_trans + v_forward + v_right / 2.0 - v_up / 2.0, rgb));
    mesh_cld.push_back(eigen_to_PointXYZRGB(T_trans + v_forward - v_right / 2.0 + v_up / 2.0, rgb));
    mesh_cld.push_back(eigen_to_PointXYZRGB(T_trans + v_forward - v_right / 2.0 - v_up / 2.0, rgb));

    pcl::PolygonMesh pm;
    pm.polygons.resize(6);
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 3; j++) {
        pm.polygons[i].vertices.push_back(CAM_POLYGON[i*3 + j]);
      }
    }
    pcl::toROSMsg(mesh_cld, pm.cloud);
    cam_meshes.push_back(std::make_pair(name, pm));
  }

  // Viewing Rays.
  {
    std::string line_name = name + "line";
    lines_to_draw.push_back(
      std::make_pair(
        line_name, 
        to_vector(eigen_2_eigen(T_trans, rgb), eigen_2_eigen(T_trans + v_forward * VIEWING_RAY_SCALE_COEFF, rgb)))
    );
  }
}

}

namespace simple_sfm {
namespace visualization {

void draw_point_clouds(
  const common::vec1d<cv::Point3d>& point_cloud_dists,
  const common::vec1d<cv::Vec3b>& point_cloud_clrs,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& point_cloud) {

  point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  populate_pcl_point_clouds(point_cloud_dists, point_cloud_clrs, point_cloud);

}

void draw_camera_poses(
  const common::vec1d<cv::Matx34d>& cam_poses,
  simple_sfm::common::vec1d<std::pair<std::string,pcl::PolygonMesh> >& cam_meshes,
  simple_sfm::common::vec1d<std::pair<std::string,std::vector<Eigen::Matrix<float,6,1> > > >& lines_to_draw) {
  
  for (size_t idx = 0; idx < cam_poses.size(); idx++) {
    std::stringstream ss;
    ss << "Camera" << idx;
    compose_camera_element(
      cam_poses[idx], ss.str(), 
      CAM_COLOR(0), CAM_COLOR(1), CAM_COLOR(2), 
      cam_meshes, lines_to_draw, CAM_SCALE_COEFF);
  }
}

void create_rgb_vector_from_point_cloud(
  const common::vec2d<cv::KeyPoint>& key_points,
  const common::vec1d<cv::Mat>& original_imgs,
  const common::vec1d<common::CloudPoint>& point_cloud,
  common::vec1d<cv::Vec3b>& rgb_vecs){

  rgb_vecs.resize(point_cloud.size());

  using c_itr = common::vec1d<common::CloudPoint>::const_iterator;

  // Loop : Point Cloud.
  for (size_t cp_idx = 0; cp_idx < point_cloud.size(); cp_idx++) {   
    
    common::CloudPoint cp = point_cloud[cp_idx];
    common::vec1d<cv::Vec3b> point_clrs;
    // Loop : Image in which the ponit is visible.
    for (size_t img_idx = 0; img_idx < original_imgs.size(); img_idx++) {
      
      size_t pt_idx_in_img = cp.idx_in_img[img_idx];
      // This point is visible in "img_idx"
      if (pt_idx_in_img != -1) {
        assert(pt_idx_in_img < key_points[img_idx].size());
        assert(img_idx < original_imgs.size());
        cv::Point pt = key_points[img_idx][pt_idx_in_img].pt;
        assert(pt.x < original_imgs[img_idx].cols && pt.y < original_imgs[img_idx].rows);

        //cv::Vec3b rgbv = original_imgs[img_idx].at<cv::Vec3b>(pt);
        point_clrs.push_back(original_imgs[img_idx].at<cv::Vec3b>(pt));

        //std::cout << "RGB : " << (int)rgbv[2] << ", " << (int)rgbv[1] << ", " << (int)rgbv[0] << std::endl;

      }
    }

    cv::Scalar color = cv::mean(point_clrs);
    rgb_vecs[cp_idx] = (cv::Vec3b(color[0], color[1], color[2]));
  }
}

}
}