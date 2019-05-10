
// Self Header
#include <app/fmat_point3d_gen_runner.h>

// Original
#include <common/container.h>
#include <common/container_util.h>
#include <geometry/geometry.h>


using namespace simple_sfm::common;
using namespace simple_sfm::geometry;

namespace simple_sfm {
namespace app {

struct FMatPoint3DGenRunnerInternalStorage {

};

FMatPoint3DGenRunner::FMatPoint3DGenRunner() :
  m_intl(nullptr)
{}

FMatPoint3DGenRunner::~FMatPoint3DGenRunner()
{}

bool FMatPoint3DGenRunner::Initialize() {

  m_intl.reset(new FMatPoint3DGenRunnerInternalStorage());


  return true;
}

bool FMatPoint3DGenRunner::Run(
      const AlgoIF& constState,
      AlgoIF& state) {

  const FMatPoint3DGenConst& c_interface = static_cast<const FMatPoint3DGenConst &>(constState);
  FMatPoint3DGen& interface = static_cast<FMatPoint3DGen &>(state);

  std::vector<std::pair<int, std::pair<size_t, size_t> > > sorted_match_list
    = SortMatchListWrtHomographyInliers(c_interface.f_ref_matrix, c_interface.homo_ref_matrix);

  // 13. Calculate 1st Camera Matrix.
  //interface.cam_poses.resize(c_interface.img_path_list.size());
  {
    std::cout << std::endl << "13. Calculate 1st Camera Matrix." << std::endl;    
    cv::Matx34d Porigin(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0), P;
    size_t train_img, query_img;
    using sorted_pair = vec1d<std::pair<int, std::pair<size_t, size_t> > >;
    bool cam_mat_result = false;
    for (sorted_pair::const_iterator citr = sorted_match_list.cbegin();
         citr != sorted_match_list.cend();
         citr++) {
      size_t train_img_idx = citr->second.first;
      size_t query_img_idx = citr->second.second;

      std::cout << std::endl << "Try to find camera matrix : " 
      << train_img_idx << " and " << query_img_idx << std::endl;

      const common::vec1d<cv::DMatch>& matches = common::getMapValue(c_interface.f_ref_matrix, citr->second);
      common::vec1d<cv::Point2f> aligned_point2f_list_train, aligned_point2f_list_query;
      common::container_util::create_point2f_list_aligned_with_matches(
        matches,
        c_interface.key_points[train_img_idx], c_interface.key_points[query_img_idx],
        aligned_point2f_list_train, aligned_point2f_list_query);

      // Initialize P everytime.
      common::vec1d<common::Point3dWithRepError> tmp_point3d_w_reperr_list;
      cv::Matx34d tmp_P(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
      cam_mat_result = cam_mat_result || find_camera_matrix(
        c_interface.cam_intr, 
        aligned_point2f_list_train, 
        aligned_point2f_list_query,
        Porigin, tmp_P, tmp_point3d_w_reperr_list);

      if(cam_mat_result) {
        P = tmp_P;
        train_img = train_img_idx;
        query_img = query_img_idx;
        interface.processed_view.insert(train_img_idx);
        interface.pose_recovered_view.insert(train_img_idx);
        interface.adopted_view.insert(train_img_idx);
        interface.processed_view.insert(query_img_idx);
        interface.pose_recovered_view.insert(query_img_idx);
        interface.adopted_view.insert(query_img_idx);

        std::cout << std::endl << "Camera Matrix Found. P : " << std::endl;
        std::cout << P << std::endl;

        common::vec1d<common::CloudPoint> new_cloud_point;
        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          c_interface.img_path_list.size(), tmp_point3d_w_reperr_list, new_cloud_point
        );

        for (size_t idx = 0; idx < new_cloud_point.size(); idx++) {
          if (new_cloud_point[idx].pt.valid) {
            new_cloud_point[idx].idx_in_img[train_img_idx] = matches[idx].trainIdx;
            new_cloud_point[idx].idx_in_img[query_img_idx] = matches[idx].queryIdx;
          }
        }
        interface.sfm_result.AddPointCloud(new_cloud_point);
        break;
      }
    }

    if (!cam_mat_result) {
      std::cout << "Failed to find Camera Matrix.... Exit here." << std::endl;
      std::exit(0);
    }
    interface.sfm_result.AddCamPoses(train_img, Porigin);
    interface.sfm_result.AddCamPoses(query_img, P);
  }

  return true;
}

bool FMatPoint3DGenRunner::Terminate() {


  m_intl.reset(nullptr);

  return true;
}

std::vector<std::pair<int, std::pair<size_t, size_t> > >
    FMatPoint3DGenRunner::SortMatchListWrtHomographyInliers(
      const common::match_matrix& f_ref_matrix,
      const common::match_matrix& homo_ref_matrix) {

      // 12. Sort img pair in the order of the number of inliers from homography calc.
  std::vector<std::pair<int, std::pair<size_t, size_t> > > sorted_match_list;
  {
    
    std::cout << std::endl << "12. Sort img pair in the order of the number of inliers from homography calc." << std::endl;
    for (match_matrix::const_iterator citr = homo_ref_matrix.cbegin();
         citr != homo_ref_matrix.cend();
         citr++) {
      std::pair<size_t, size_t> key = citr->first;
      int h_inliers = citr->second.size();
      int original = common::getMapValue(f_ref_matrix, key).size();

      std::pair<int, std::pair<size_t, size_t> > elem;
      elem.first = (int)(100.0 * (double)h_inliers / (double)original);
      elem.second = key;
      sorted_match_list.push_back(elem);
    }
    // Sort from low to high.
    std::sort(
      sorted_match_list.begin(),
      sorted_match_list.end(),
      [](const std::pair<int, std::pair<size_t, size_t>>& a,
         const std::pair<int, std::pair<size_t, size_t>>& b) {
           return a.first < b.first;
         }
    );
  }
  return sorted_match_list;
}

}
}