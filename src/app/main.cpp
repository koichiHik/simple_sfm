
// System Library
#include <iostream>
#include <string>

// STL
#include <vector>
#include <thread>
#include <set>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Original
#include <common/container.h>
#include <common/container_util.h>
#include <common/math.h>
#include <common/point_cloud_util.h>
#include <utility/fileutil.h>
#include <utility/vis2dutil.h>
#include <feature_extractor/IFeatureExtractorFactory.h>
#include <descriptor_matcher/IDescriptorMatcherFactory.h>
#include <key_point_filter/IKeyPointFilterFactory.h>
#include <geometry/Geometry.h>
#include <vis3d/PCLViewer.h>
#include <vis3d/PCLDrawer.h>
#include <app/app_container.h>
#include <app/app_util.h>

// Runner
#include <app/sfm_runner.h>

// Original namespace
using namespace simple_sfm;
using namespace simple_sfm::app;
using namespace simple_sfm::common;
using namespace simple_sfm::utility;
using namespace simple_sfm::feature_extractor;
using namespace simple_sfm::descriptor_matcher;
using namespace simple_sfm::key_point_filter;
using namespace simple_sfm::geometry;
using namespace simple_sfm::vis3d;


namespace {

int sfm_run(int argc, char** argv) {



  // 1. Parse User Specified Argument.
  SfmConfig config;
  if (!parseArgment(argc, argv, config)) {
    return 0;
  }

  // 2. Collect all paths for image.
  {
    std::cout << std::endl << "2. Collect all paths for image." << std::endl;  
    std::vector<std::string> img_list{}, extensions{".jpg", ".png"};
    file::raise_all_img_files_in_directory(config.img_dir, 
                                config.img_path_list, 
                                extensions);
  }

  // 3. Read Calib Information and Load Images.
  //Images images;
  {
    std::cout << std::endl << "3. Read Calib Information and Load Images." << std::endl;
    file::read_calib_file(config.img_dir + "/K.txt", config.cam_intr);
    //file::load_images(config.img_path_list, images.org_imgs, images.gray_imgs);
  }

  SfMRunner runner;
  runner.Initialize(config);
  //SfmDB& db = runner.GetSfmDB();

  #if 0
  // 4. Display Image and Thread Start for later visualziation.
  vis3d::PCLViewer viewer;
  {
    std::cout << std::endl << "4. Display Image." << std::endl;
    vis2d::resize_and_show(db.images.org_imgs, "Original Image", 0.25, 100);
    viewer.run_visualization_async();
  }
#endif

# if 1

  runner.Run();

#elif
  // 5. Feature detection and Descriptor extraction.
  //FeatureMatching feature_match;
  {
    std::cout << std::endl << "5. Feature detection and Descriptor extraction." << std::endl;  
    cv::Ptr<IFeatureExtractor> feature_extractor 
        = IFeatureExtractorFactory::createFeatureExtractor(FeatureType::GPU_SURF);
    feature_extractor->detectAndCompute(
      db.images.gray_imgs, db.feature_match.key_points, db.feature_match.descriptors);
  }

  // 6. Draw Feature and Display.
  {
    std::cout << std::endl << "6. Draw Feature and Display." << std::endl;
    vis2d::draw_key_points(db.images.org_imgs, db.feature_match.key_points, db.images.key_pnt_imgs);
  }

  // 7. Match key points of each image pairs.
  {
    std::cout << std::endl << "7. Match key points of each image pairs." << std::endl;
    cv::Ptr<IDescriptorMatcher> descriptor_matcher
        = IDescriptorMatcherFactory::createDescriptorMatcher(MatcherType::GPU_BF_RATIO_CHECK);
    descriptor_matcher->createMatchingMatrix(db.feature_match.descriptors, db.feature_match.matrix);
  }

#endif

#if 0
  // 8. Draw and show matched images.
  {
    std::cout << std::endl << "8. Draw and show matched images." << std::endl;
    vis2d::draw_matched_imgs(
      db.images.org_imgs, db.feature_match.key_points, 
      db.feature_match.matrix, db.images.matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }
#endif

#if 1

#elif
  // 9. Filtering by F matrix calculation.
  {
    std::cout << std::endl << "9. Filtering by F matrix calculation." << std::endl;
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    fmat_point_filter->run(
      db.feature_match.key_points, db.feature_match.matrix, db.feature_match.f_ref_matrix);

    std::cout << std::endl << "F Match Result : " << std::endl;
    using match_map = std::map<std::pair<size_t, size_t>, common::vec1d<cv::DMatch> >;
    for (match_map::const_iterator citr = db.feature_match.f_ref_matrix.cbegin();
         citr != db.feature_match.f_ref_matrix.cend();
         citr++) {
      std::cout << citr->second.size() << std::endl;
    }    

  }
#endif

#if 0
  // 10. Draw and show matched images.
  {
    std::cout << std::endl << "10. Draw and show matched images." << std::endl;
    vis2d::draw_matched_imgs(
      db.images.org_imgs, db.feature_match.key_points,
      db.feature_match.f_ref_matrix, db.images.refined_matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }

  // 11. Apply homography check for removing coplanar point set.
  {
    std::cout << std::endl << "11. Apply homography check for removing coplanar point set." << std::endl;
    cv::Ptr<IKeyPointFilter> homo_point_filter 
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::HOMOGRAPHY);
    homo_point_filter->run(
      db.feature_match.key_points, db.feature_match.f_ref_matrix, db.feature_match.homo_ref_matrix);
  }

#endif

#if 1

#elif
  // 12. Sort img pair in the order of the number of inliers from homography calc.
  std::vector<std::pair<int, std::pair<size_t, size_t> > > sorted_match_list;
  {
    std::cout << std::endl << "12. Sort img pair in the order of the number of inliers from homography calc." << std::endl;
    for (match_matrix::const_iterator citr = db.feature_match.homo_ref_matrix.cbegin();
         citr != db.feature_match.homo_ref_matrix.cend();
         citr++) {
      std::pair<size_t, size_t> key = citr->first;
      int h_inliers = citr->second.size();
      int original = db.feature_match.f_ref_matrix[key].size();

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

  // 13. Calculate 1st Camera Matrix.
  db.sfm_result.cam_poses.resize(db.images.org_imgs.size());
  {
    std::cout << std::endl << "13. Calculate 1st Camera Matrix." << std::endl;    
    cv::Matx34d Porigin(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0), P;
    size_t train_img, query_img;
    using sorted_pair = vec1d<std::pair<int, std::pair<size_t, size_t> > >;
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    bool cam_mat_result = false;
    for (sorted_pair::const_iterator citr = sorted_match_list.cbegin();
         citr != sorted_match_list.cend();
         citr++) {
      size_t train_img_idx = citr->second.first;
      size_t query_img_idx = citr->second.second;

      std::cout << std::endl << "Try to find camera matrix : " 
      << train_img_idx << " and " << query_img_idx << std::endl;

      common::vec1d<cv::DMatch> matches = db.feature_match.f_ref_matrix[citr->second];
      common::vec1d<cv::Point2f> aligned_point2f_list_train, aligned_point2f_list_query;
      common::container_util::create_point2f_list_aligned_with_matches(
        matches, 
        db.feature_match.key_points[train_img_idx], db.feature_match.key_points[query_img_idx],
        aligned_point2f_list_train, aligned_point2f_list_query);

      // Initialize P everytime.
      common::vec1d<common::Point3dWithRepError> tmp_point3d_w_reperr_list;
      cv::Matx34d tmp_P(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
      cam_mat_result = cam_mat_result || geometry::find_camera_matrix(
        db.config.cam_intr, 
        aligned_point2f_list_train, 
        aligned_point2f_list_query,
        Porigin, tmp_P, tmp_point3d_w_reperr_list);

      if(cam_mat_result) {
        P = tmp_P;
        train_img = train_img_idx;
        query_img = query_img_idx;
        db.algo_status.adopted_view.insert(train_img_idx);
        db.algo_status.adopted_view.insert(query_img_idx);
        db.algo_status.pose_recovered_view.insert(train_img_idx);
        db.algo_status.pose_recovered_view.insert(query_img_idx);
        db.algo_status.processed_view.insert(train_img_idx);
        db.algo_status.processed_view.insert(query_img_idx);

        std::cout << std::endl << "Camera Matrix Found. P : " << std::endl;
        std::cout << P << std::endl;

        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          db.images.org_imgs.size(), tmp_point3d_w_reperr_list, db.sfm_result.point_cloud
        );

        for (size_t idx = 0; idx < db.sfm_result.point_cloud.size(); idx++) {
          db.sfm_result.point_cloud[idx].idx_in_img.resize(db.images.org_imgs.size(), -1);
          if (db.sfm_result.point_cloud[idx].pt.valid) {
            db.sfm_result.point_cloud[idx].idx_in_img[train_img_idx] = matches[idx].trainIdx;
            db.sfm_result.point_cloud[idx].idx_in_img[query_img_idx] = matches[idx].queryIdx;
          }
        }
        break;
      }
    }

    if (!cam_mat_result) {
      std::cout << "Failed to find Camera Matrix.... Exit here." << std::endl;
      std::exit(0);
    }
    db.sfm_result.cam_poses[train_img] = Porigin;
    db.sfm_result.cam_poses[query_img] = P;
  }

  // Draw Result via PCL Viewer.
  {
    vec1d<cv::Vec3b> rgb_clrs;
    vec1d<cv::Point3d> point3d_list;
    create_rgb_vector_from_point_cloud(
      db.feature_match.key_points, db.images.org_imgs, db.sfm_result.point_cloud, rgb_clrs);
    container_util::convert_cloud_point_list_to_point3d_list(
      db.sfm_result.point_cloud, point3d_list);
    viewer.update(point3d_list, rgb_clrs, db.sfm_result.cam_poses);

    cv::Mat dummy(100, 100, CV_8UC1);
    cv::imshow("dummy", dummy);
    cv::waitKey(0);

  }

#endif

#if 0
  // Accumulation Phase.
  {
    common::vec2d<cv::Point2f> point2d_list(db.images.org_imgs.size());
    for (size_t idx = 0; idx < db.images.org_imgs.size(); idx++) {
      container_util::convert_key_point_list_to_point2f_list(
        db.feature_match.key_points[idx], point2d_list[idx]);
    }

    // Triangulation with the all processed views.
    while (db.algo_status.processed_view.size() < db.images.org_imgs.size()) {

      common::vec1d<cv::Point2f> corresp_2d_pnts;
      common::vec1d<cv::Point3f> corresp_3d_pnts;

      int query_img_idx = find_best_matching_img_with_current_cloud(
        db.algo_status.processed_view,
        db.algo_status.adopted_view,
        db.feature_match.f_ref_matrix,
        point2d_list,
        db.sfm_result.point_cloud,
        corresp_2d_pnts,
        corresp_3d_pnts);

      if (query_img_idx == -1) {
        break;
      }

      // Register this img as "Processed"
      db.algo_status.processed_view.insert(query_img_idx);

      cv::Matx34d Pnew;
      bool pose_estimated = find_camera_matrix_via_pnp(
        db.config.cam_intr, corresp_2d_pnts, corresp_3d_pnts, Pnew);
      if (!pose_estimated) {
        continue;
      }
      db.algo_status.pose_recovered_view.insert(query_img_idx);
      db.sfm_result.cam_poses[query_img_idx] = Pnew;

      std::cout << std::endl << "Found camera matrices ";
      for (std::set<size_t>::const_iterator citr = db.algo_status.pose_recovered_view.cbegin();
           citr != db.algo_status.pose_recovered_view.cend();
           citr++) {
        std::cout << std::endl << "P : " << std::endl << db.sfm_result.cam_poses[*citr];
      }

      // Triangulation with already adopted views.
      for (std::set<size_t>::const_iterator citr = db.algo_status.adopted_view.cbegin();
           citr != db.algo_status.adopted_view.cend();
           citr++) {
        
        size_t train_img_idx = *citr;
        if (query_img_idx == train_img_idx) {
          continue;
        }

        common::vec1d<cv::Point2f> aligned_point2d_list_query, aligned_point2d_list_train;
        
        common::container_util::create_point2f_list_aligned_with_matches(
          db.feature_match.f_ref_matrix[std::make_pair(train_img_idx, query_img_idx)], 
          db.feature_match.key_points[train_img_idx], db.feature_match.key_points[query_img_idx],
          aligned_point2d_list_train, aligned_point2d_list_query);

        common::vec1d<Point3dWithRepError> tmp_point3d_w_reperr;
        bool tri_result 
          = triangulate_points_and_validate(
              aligned_point2d_list_train,
              aligned_point2d_list_query,
              db.config.cam_intr,
              db.sfm_result.cam_poses[train_img_idx],
              db.sfm_result.cam_poses[query_img_idx],
              tmp_point3d_w_reperr);

        common::vec1d<CloudPoint> cp_triangulated;
        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          db.images.org_imgs.size(), tmp_point3d_w_reperr, cp_triangulated
        );

        if (!tri_result) {
          continue;
        }

        std::pair<size_t, size_t> key = std::make_pair(train_img_idx, query_img_idx);
        common::vec1d<cv::DMatch> match = db.feature_match.f_ref_matrix[key];
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
          for (size_t old_cp_idx = 0; old_cp_idx < db.sfm_result.point_cloud.size(); old_cp_idx++) {
            
            int old_cp_query_idx = db.sfm_result.point_cloud[old_cp_idx].idx_in_img[train_img_idx];

            // If this cp exists already.
            if (old_cp_query_idx == new_cp_query_idx) {
              db.sfm_result.point_cloud[old_cp_idx].idx_in_img[query_img_idx] = new_cp_query_idx;
              continue;
            }
            
            if (added_point.find(new_cp_train_idx) != added_point.end()) {
              continue;
            }

            // This point new and has to be added.
            CloudPoint cp(db.images.org_imgs.size());
            cp.idx_in_img[train_img_idx] = new_cp_train_idx;
            cp.idx_in_img[query_img_idx] = new_cp_query_idx;
            cp.pt = cp_triangulated[new_cp_idx].pt;
            add_to_cloud.push_back(cp);
            added_point.insert(new_cp_train_idx);
          }
        }
        std::cout << std::endl << "Image Pair (" << query_img_idx << ", " << *citr << ")" << std::endl;
        std::cout << "Original : " << db.sfm_result.point_cloud.size() << ", Added : " << add_to_cloud.size() << std::endl;
        db.sfm_result.point_cloud.reserve(db.sfm_result.point_cloud.size() + add_to_cloud.size());
        db.sfm_result.point_cloud.insert(db.sfm_result.point_cloud.end(), add_to_cloud.begin(), add_to_cloud.end());
        add_to_cloud.clear();
        added_point.clear();
      }

      db.algo_status.adopted_view.insert(query_img_idx);

      // Draw Result via PCL Viewer.
      {
        vec1d<cv::Vec3b> rgb_clrs;
        vec1d<cv::Point3d> point3d_list;
        create_rgb_vector_from_point_cloud(
          db.feature_match.key_points, db.images.org_imgs, db.sfm_result.point_cloud, rgb_clrs);
        container_util::convert_cloud_point_list_to_point3d_list(
          db.sfm_result.point_cloud, point3d_list);
        viewer.update(point3d_list, rgb_clrs, db.sfm_result.cam_poses);

        cv::Mat dummy(100, 100, CV_8UC1);
        cv::imshow("dummy", dummy);
        cv::waitKey(0);


      }

    }
  }

#endif

#if 0
  // Draw Result via PCL Viewer.
  {
    vec1d<cv::Vec3b> rgb_clrs;
    vec1d<cv::Point3d> point3d_list;
    create_rgb_vector_from_point_cloud(
      db.feature_match.key_points, db.images.org_imgs, db.sfm_result.point_cloud, rgb_clrs);
    container_util::convert_cloud_point_list_to_point3d_list(
      db.sfm_result.point_cloud, point3d_list);
    viewer.update(point3d_list, rgb_clrs, db.sfm_result.cam_poses);

    cv::Mat dummy(100, 100, CV_8UC1);
    cv::imshow("dummy", dummy);
    cv::waitKey(0);

  }
#endif

  // 14. Wait till visualization thread gets joined.
  //viewer.wait_vis_thread();
  runner.Terminate();

  return 0;
}

}

int main(int argc, char** argv) {
  std::cout << __FILE__ << " starts!" << std::endl;
  sfm_run(argc, argv);
  return 0;
}
