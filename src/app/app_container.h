
#ifndef APP_CONTAINER_H
#define APP_CONTAINER_H

// STL
#include <string>
#include <set>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>
#include <app/if_app_viewer.h>

namespace simple_sfm {
namespace app {


struct SfmConfig {
  std::string img_dir;
  common::vec1d<std::string> img_path_list;
  common::CamIntrinsics cam_intr;
};

struct Images {
  common::vec1d<cv::Mat> org_imgs;
  common::vec1d<cv::Mat> gray_imgs;
  common::vec1d<cv::Mat> key_pnt_imgs;
  common::matching_map<cv::Mat> matched_imgs;
  common::matching_map<cv::Mat> refined_matched_imgs;
};

struct FeatureMatch {
  common::vec2d<cv::KeyPoint> key_points;
  common::vec1d<cv::Mat> descriptors;
  common::match_matrix matrix;
  common::match_matrix f_ref_matrix;
  common::match_matrix homo_ref_matrix;
};

struct AlgoStatus {
  std::set<size_t> processed_view;
  std::set<size_t> pose_recovered_view;
  std::set<size_t> adopted_view;
};

class SfmResult {
private:
  common::vec1d<common::CloudPoint> point_cloud;
  common::vec1d<cv::Matx34d> cam_poses;
  common::vec1d<SfmResultUpdateListener *> listeners;
public:
  SfmResult(size_t img_num) {
    cam_poses.resize(img_num);
  }

  const common::vec1d<common::CloudPoint>& GetPointCloud() {
    return point_cloud;
  }
  const common::vec1d<cv::Matx34d>& GetCamPoses() {
    return cam_poses;
  }
  void AddPointCloud(common::vec1d<common::CloudPoint>& new_cloud_point) {
    point_cloud.reserve(point_cloud.size() + new_cloud_point.size());
    std::copy(new_cloud_point.begin(), new_cloud_point.end(), std::back_inserter(point_cloud));

    this->NotifyAll();
  }
  void AddCamPoses(size_t idx, cv::Matx34d& pose) {
    cam_poses[idx] = pose;

    this->NotifyAll();
  }
  void AddListener(SfmResultUpdateListener* l) {
    listeners.push_back(l);
  }
  void NotifyAll() {
    std::for_each(listeners.begin(), listeners.end(), 
        [&](SfmResultUpdateListener* l) { l->Update(point_cloud, cam_poses); } );
  }
};

struct SfmDB {
  SfmDB(const SfmConfig& config) :
    config(config), sfm_result(config.img_path_list.size())
  {}
  const SfmConfig config;
  Images images;
  FeatureMatch feature_match;
  AlgoStatus algo_status;
  SfmResult sfm_result;
};

struct AlgoIF {
public:
  virtual ~AlgoIF()
  {}
};

}
}

#endif // APP_CONTAINER_H
