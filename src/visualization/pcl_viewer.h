
#ifndef PCL_VIEWER_H
#define PCL_VIEWER_H

// System

// STL
#include <string>
#include <memory>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>
#include <app/i_data_update_listener.h>

namespace simple_sfm {
namespace visualization {

struct PCLViewerInternalStorage;
struct PCLViewerHandler;
class PCLViewer : public app::SfmResultUpdateListener {
public:
  PCLViewer(const std::string& window_name,
            const common::vec1d<cv::Mat>& org_img_list,
            const common::vec2d<cv::KeyPoint>& key_point_lists);

  ~PCLViewer();

  virtual void Update(const common::vec1d<common::CloudPoint>& cloud,
                      const common::vec1d<cv::Matx34d>& poses);

  void run_visualization();

  void run_visualization_async();

  void wait_vis_thread();

  void update(
    const common::vec1d<cv::Point3d>& point_cloud_dists,
    const common::vec1d<cv::Vec3b>& point_cloud_clrs,
    const common::vec1d<cv::Matx34d>& cameras);

private:
  std::unique_ptr<PCLViewerInternalStorage> m_intl;

friend struct PCLViewerHandler;
};

}
}

#endif // PCL_VIEWER_H