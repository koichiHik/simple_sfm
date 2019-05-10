
// STL
#include <thread>
#include <deque>

// Eigen
#include <Eigen/Eigen>

// PCL
#include <pcl/visualization/pcl_visualizer.h>

// Original
#include <visualization/pcl_viewer.h>
#include <visualization/pcl_drawer.h>
#include <common/container.h>
#include <common/container_util.h>

namespace {

void convert_eigen_matrix_to_PointXYZRGB(
  const Eigen::Matrix<float, 6, 1>& pnt_org,
  pcl::PointXYZRGB& pnt_xyzrgb) {
  pnt_xyzrgb.x = pnt_org[0];
  pnt_xyzrgb.y = pnt_org[1];
  pnt_xyzrgb.z = pnt_org[2];
  pnt_xyzrgb.r = pnt_org[3];
  pnt_xyzrgb.g = pnt_org[4];
  pnt_xyzrgb.b = pnt_org[5];
}

}

namespace simple_sfm {
namespace visualization {

struct PCLViewerInternalStorage {
  PCLViewerInternalStorage(const std::string& window_name, 
                           const common::vec1d<cv::Mat>& org_img_list,
                           const common::vec2d<cv::KeyPoint>& key_point_lists) :
    m_window_name(window_name), m_cloud_name(""), m_org_img_list(org_img_list),
    m_key_point_lists(key_point_lists),
    m_quit(false), m_cloud_updated(false), m_camera_updated(false),
    point_cloud(nullptr), m_p_vis_thread(nullptr) 
  {}

  bool m_quit, m_cloud_updated, m_camera_updated;
  std::string m_window_name, m_cloud_name;
  std::unique_ptr<std::thread> m_p_vis_thread;

  const common::vec1d<cv::Mat>& m_org_img_list;
  const common::vec2d<cv::KeyPoint>& m_key_point_lists;

  // Drawing elements.
  common::vec1d<std::pair<std::string,pcl::PolygonMesh> > cam_meshes;
  common::vec1d<std::pair<std::string,std::vector<Eigen::Matrix<float,6,1> > > > lines_to_draw;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud;
};

struct PCLViewerHandler {

  static void keyboard_event_handler(
    const pcl::visualization::KeyboardEvent& event,
    void* PCLViewer_void) {
      PCLViewer* p_pcl_viewer = static_cast<PCLViewer *>(PCLViewer_void);
      std::string key = event.getKeySym();
      if (key == "Escape" && event.keyDown()) {
        p_pcl_viewer->m_intl->m_quit = true;
      }
  }
};

PCLViewer::PCLViewer(const std::string& window_name, 
                     const common::vec1d<cv::Mat>& org_img_list,
                     const common::vec2d<cv::KeyPoint>& key_point_lists) :
  m_intl(new PCLViewerInternalStorage(window_name, org_img_list, key_point_lists))
{}

PCLViewer::~PCLViewer()
{}

void PCLViewer::Update(const common::vec1d<common::CloudPoint>& cloud,
                       const common::vec1d<cv::Matx34d>& poses) {

  common::vec1d<cv::Vec3b> rgb_clrs;
  common::vec1d<cv::Point3d> point3d_list;

  create_rgb_vector_from_point_cloud(
    m_intl->m_key_point_lists, m_intl->m_org_img_list, cloud, rgb_clrs);

  common::container_util::convert_cloud_point_list_to_point3d_list(
    cloud, point3d_list);

  this->update(point3d_list, rgb_clrs, poses);
  
}

void PCLViewer::run_visualization_async() {
  m_intl->m_p_vis_thread.reset(new std::thread(&PCLViewer::run_visualization, this));
}

void PCLViewer::wait_vis_thread() {
  m_intl->m_p_vis_thread->join();
}

void PCLViewer::run_visualization() {

  pcl::visualization::PCLVisualizer viewer(m_intl->m_window_name);

  viewer.registerKeyboardCallback(
    PCLViewerHandler::keyboard_event_handler,
    static_cast<void *>(this));

  // Loop till viewer gets stopped.
  while(!viewer.wasStopped()) {
    if (m_intl->m_quit) {
      break;
    }

    if (m_intl->m_cloud_updated) {
      viewer.removePointCloud("orig");
      viewer.addPointCloud(m_intl->point_cloud, "orig");
      m_intl->m_cloud_updated = false;
    }

    if (m_intl->m_camera_updated) {
      for (size_t i = 0; i < m_intl->cam_meshes.size(); i++) {
        viewer.removeShape(m_intl->cam_meshes[i].first);
        viewer.addPolygonMesh(m_intl->cam_meshes[i].second, m_intl->cam_meshes[i].first);
      }
      m_intl->cam_meshes.clear();

      for (size_t i = 0; i < m_intl->lines_to_draw.size(); i++) {
        pcl::PointXYZRGB A, B;
        convert_eigen_matrix_to_PointXYZRGB(m_intl->lines_to_draw[i].second[0], A);
        convert_eigen_matrix_to_PointXYZRGB(m_intl->lines_to_draw[i].second[1], B);
        viewer.removeShape(m_intl->lines_to_draw[i].first);
        viewer.addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(A, B, m_intl->lines_to_draw[i].first);
      }
      m_intl->m_camera_updated = false;
    }    

    viewer.spinOnce();
  }
}

void PCLViewer::update(
  const common::vec1d<cv::Point3d>& point_cloud_dists,
  const common::vec1d<cv::Vec3b>& point_cloud_clrs,
  const common::vec1d<cv::Matx34d>& cameras) {

  draw_point_clouds(point_cloud_dists, point_cloud_clrs, m_intl->point_cloud);
  m_intl->m_cloud_updated = true;

  draw_camera_poses(cameras, m_intl->cam_meshes, m_intl->lines_to_draw);
  m_intl->m_camera_updated = true;
}

} // namespace vis3d
} // namespace simple_sfm

