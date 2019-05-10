
// Self Header
#include <app/pnp_point3d_gen_runner.h>

// Original
#include <common/container.h>
#include <common/container_util.h>
#include <common/point_cloud_util.h>
#include <geometry/geometry.h>

using namespace simple_sfm::common;
using namespace simple_sfm::geometry;

namespace simple_sfm {
namespace app {

struct PNPPoint3DGenRunnerInternalStorage {

};

PNPPoint3DGenRunner::PNPPoint3DGenRunner() :
  m_intl(nullptr)
{}

PNPPoint3DGenRunner::~PNPPoint3DGenRunner()
{}

bool PNPPoint3DGenRunner::Initialize() {

  m_intl.reset(new PNPPoint3DGenRunnerInternalStorage());


  return true;
}

bool PNPPoint3DGenRunner::Run(
      const AlgoIF& constState,
      AlgoIF& state) {

  const PNPPoint3DGenGenConst& c_interface = static_cast<const PNPPoint3DGenGenConst &>(constState);
  PNPPoint3DGenGen& interface = static_cast<PNPPoint3DGenGen &>(state);

    // Accumulation Phase.
  {
    size_t img_num = c_interface.img_path_list.size();
    common::vec2d<cv::Point2f> point2d_list(img_num);
    for (size_t idx = 0; idx < c_interface.img_path_list.size(); idx++) {
      container_util::convert_key_point_list_to_point2f_list(
        c_interface.key_points[idx], point2d_list[idx]);
    }

    // Triangulation with the all processed views.
    while (interface.processed_view.size() < c_interface.img_path_list.size()) {

      common::vec1d<cv::Point2f> corresp_2d_pnts;
      common::vec1d<cv::Point3f> corresp_3d_pnts;

      int query_img_idx = -1;
      {
        const common::vec1d<CloudPoint>& current_cloud = interface.sfm_result.GetPointCloud();
        query_img_idx = find_best_matching_img_with_current_cloud(
          interface.processed_view,
          interface.adopted_view,
          c_interface.f_ref_matrix,
          point2d_list,
          current_cloud,
          //interface.point_cloud,
          corresp_2d_pnts,
          corresp_3d_pnts);
      }

      if (query_img_idx == -1) {
        break;
      }

      // Register this img as "Processed"
      interface.processed_view.insert(query_img_idx);

      cv::Matx34d Pnew;
      bool pose_estimated = find_camera_matrix_via_pnp(
        c_interface.cam_intr, corresp_2d_pnts, corresp_3d_pnts, Pnew);
      if (!pose_estimated) {
        continue;
      }
      interface.pose_recovered_view.insert(query_img_idx);
      interface.sfm_result.AddCamPoses(query_img_idx, Pnew);

      std::cout << std::endl << "Found camera matrices ";
    

      for (std::set<size_t>::const_iterator citr = interface.pose_recovered_view.cbegin();
           citr != interface.pose_recovered_view.cend();
           citr++) {
        const common::vec1d<cv::Matx34d>& current_pose = interface.sfm_result.GetCamPoses();
        std::cout << std::endl << "P : " << std::endl << current_pose[*citr];
      }

      // Triangulation with already adopted views.
      for (std::set<size_t>::const_iterator citr = interface.adopted_view.cbegin();
           citr != interface.adopted_view.cend();
           citr++) {
        
        size_t train_img_idx = *citr;
        if (query_img_idx == train_img_idx) {
          continue;
        }

        common::vec1d<cv::Point2f> aligned_point2d_list_query, aligned_point2d_list_train;
        
        std::pair<size_t, size_t> key = std::make_pair(train_img_idx, query_img_idx);

        common::container_util::create_point2f_list_aligned_with_matches(
          common::getMapValue(c_interface.f_ref_matrix, key),
          c_interface.key_points[train_img_idx], c_interface.key_points[query_img_idx],
          aligned_point2d_list_train, aligned_point2d_list_query);

        common::vec1d<Point3dWithRepError> tmp_point3d_w_reperr;
        bool tri_result = false;
        {
          const common::vec1d<cv::Matx34d>& current_pose = interface.sfm_result.GetCamPoses();
          tri_result 
            = triangulate_points_and_validate(
                aligned_point2d_list_train,
                aligned_point2d_list_query,
                c_interface.cam_intr,
                current_pose[train_img_idx],
                current_pose[query_img_idx],
                tmp_point3d_w_reperr);
        }

        common::vec1d<CloudPoint> cp_triangulated;
        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          img_num, tmp_point3d_w_reperr, cp_triangulated
        );

        if (!tri_result) {
          continue;
        }

        const common::vec1d<cv::DMatch>& match = common::getMapValue(c_interface.f_ref_matrix, key);
        common::vec1d<CloudPoint> add_to_cloud;
        std::set<size_t> added_point;

        // Loop : New point cloud.
        for (size_t new_cp_idx = 0; new_cp_idx < cp_triangulated.size(); new_cp_idx++) {
          if (!cp_triangulated[new_cp_idx].pt.valid) {
            continue;
          }

          int new_cp_train_idx = match[new_cp_idx].trainIdx;
          int new_cp_query_idx = match[new_cp_idx].queryIdx;

          // Loop : Already existing point cloud.
          const common::vec1d<CloudPoint>& current_cloud = interface.sfm_result.GetPointCloud();
          for (size_t old_cp_idx = 0; old_cp_idx < current_cloud.size(); old_cp_idx++) {
            
            int old_cp_query_idx = current_cloud[old_cp_idx].idx_in_img[train_img_idx];

            // If this cp exists already.
            if (old_cp_query_idx == new_cp_query_idx) {
              // TODO FIX ME.
              const_cast<common::vec1d<CloudPoint>&>(current_cloud)[old_cp_idx].idx_in_img[query_img_idx] = new_cp_query_idx;
              //interface.point_cloud[old_cp_idx].idx_in_img[query_img_idx] = new_cp_query_idx;
              continue;
            }
            
            if (added_point.find(new_cp_train_idx) != added_point.end()) {
              continue;
            }

            // This point new and has to be added.
            CloudPoint cp(img_num);
            cp.idx_in_img[train_img_idx] = new_cp_train_idx;
            cp.idx_in_img[query_img_idx] = new_cp_query_idx;
            cp.pt = cp_triangulated[new_cp_idx].pt;
            add_to_cloud.push_back(cp);
            added_point.insert(new_cp_train_idx);
          }
        }
        const common::vec1d<CloudPoint>& current_cloud = interface.sfm_result.GetPointCloud();
        std::cout << std::endl << "Image Pair (" << query_img_idx << ", " << *citr << ")" << std::endl;
        std::cout << "Original : " << current_cloud.size() << ", Added : " << add_to_cloud.size() << std::endl;
        interface.sfm_result.AddPointCloud(add_to_cloud);
        //interface.point_cloud.reserve(current_cloud.size() + add_to_cloud.size());
        //interface.point_cloud.insert(interface.point_cloud.end(), add_to_cloud.begin(), add_to_cloud.end());
        add_to_cloud.clear();
        added_point.clear();
      }

      interface.adopted_view.insert(query_img_idx);

    }
  }


  return true;
}

bool PNPPoint3DGenRunner::Terminate() {

  m_intl.reset(nullptr);

  return true;
}

}
}