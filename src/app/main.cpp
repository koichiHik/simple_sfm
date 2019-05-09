
// System Library
#include <iostream>
#include <string>

// STL
#include <vector>
#include <thread>
#include <set>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

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

// Original namespace
using namespace simple_sfm;
using namespace simple_sfm::common;
using namespace simple_sfm::utility;
using namespace simple_sfm::feature_extractor;
using namespace simple_sfm::descriptor_matcher;
using namespace simple_sfm::key_point_filter;
using namespace simple_sfm::geometry;
using namespace simple_sfm::vis3d;

struct SfmConfig {
  std::string img_dir;
  vec1d<std::string> img_path_list;
  vec1d<std::string> img_filename_list;
  common::CamIntrinsics cam_intr;
};

struct Images {
  vec1d<cv::Mat> org_imgs;
  vec1d<cv::Mat> gray_imgs;
};

struct VisImages {
  vec1d<cv::Mat> key_pnt_imgs;
  matching_map<cv::Mat> matched_imgs;
  matching_map<cv::Mat> refined_matched_imgs;
};

struct Features {
  vec2d<cv::KeyPoint> key_points;
  vec1d<cv::Mat> descriptors;
};

struct MatchingResult {
  match_matrix matrix;
  match_matrix f_ref_matrix;
  match_matrix homo_ref_matrix;
};

struct PointCloud {
  common::vec1d<common::CloudPoint> point_cloud;
};

struct CameraPose {
  common::vec1d<cv::Matx34d> poses;
};

struct SfmState {
  std::set<size_t> processed_view;
  std::set<size_t> pose_recovered_view;
  std::set<size_t> adopted_view;
};

namespace {

common::vec1d<cv::Matx34d> get_valid_pose_vector(
  std::set<size_t>& processed_imgs, common::vec1d<cv::Matx34d>& poses) {

  common::vec1d<cv::Matx34d> valid_poses;
  for (std::set<size_t>::const_iterator citr = processed_imgs.cbegin();
       citr != processed_imgs.cend();
       citr++) {
    valid_poses.push_back(poses[*citr]);
  }
  return valid_poses;
}

bool parseArgment(int argc, char** argv, SfmConfig &config) {
  namespace po = boost::program_options;
  po::options_description opt("Option");
  po::variables_map map;

  opt.add_options()
      ("help,h", "Display help")
      ("directory,d", po::value<std::string>(), "Directory path containing images.");

  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt), map);
    boost::program_options::notify(map);

    // Cast element for value aquisition.
    if (map.count("help") || !map.count("directory")) {
      std::cout << opt << std::endl;
      return false;
    } else {
      config.img_dir = map["directory"].as<std::string>();
    }

  } catch(const boost::program_options::error_with_option_name& e) {
    std::cout << e.what() << std::endl;
    return false;
  } catch (boost::bad_any_cast& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}


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
                                config.img_filename_list,
                                extensions);
  }

  // 3. Read Calib Information and Load Images.
  Images images;
  {
    std::cout << std::endl << "3. Read Calib Information and Load Images." << std::endl;
    file::read_calib_file(config.img_dir + "/K.txt", config.cam_intr);
    file::load_images(config.img_path_list, images.org_imgs, images.gray_imgs);
  }

  // 4. Display Image and Thread Start for later visualziation.
  vis3d::PCLViewer viewer;
  {
    std::cout << std::endl << "4. Display Image." << std::endl;
    vis2d::resize_and_show(images.org_imgs, "Original Image", 0.25, 100);
    viewer.run_visualization_async();
  }

  // 5. Feature detection and Descriptor extraction.
  Features features;
  {
    std::cout << std::endl << "5. Feature detection and Descriptor extraction." << std::endl;  
    cv::Ptr<IFeatureExtractor> feature_extractor 
        = IFeatureExtractorFactory::createFeatureExtractor(FeatureType::GPU_SURF);
    feature_extractor->detectAndCompute(
      images.gray_imgs, features.key_points, features.descriptors);
  }

  // 6. Draw Feature and Display.
  VisImages vis_imgs;
  {
    std::cout << std::endl << "6. Draw Feature and Display." << std::endl;
    vis2d::draw_key_points(images.org_imgs, features.key_points, vis_imgs.key_pnt_imgs);
  }

  // 7. Match key points of each image pairs.
  MatchingResult match_result;
  {
    std::cout << std::endl << "7. Match key points of each image pairs." << std::endl;
    cv::Ptr<IDescriptorMatcher> descriptor_matcher
        = IDescriptorMatcherFactory::createDescriptorMatcher(MatcherType::GPU_BF_RATIO_CHECK);
    descriptor_matcher->createMatchingMatrix(features.descriptors, match_result.matrix);
  }

  // 8. Draw and show matched images.
  {
    std::cout << std::endl << "8. Draw and show matched images." << std::endl;
    vis2d::draw_matched_imgs(
      images.org_imgs, features.key_points, 
      match_result.matrix, vis_imgs.matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }

  // 9. Filtering by F matrix calculation.
  {
    std::cout << std::endl << "9. Filtering by F matrix calculation." << std::endl;
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    fmat_point_filter->run(
      features.key_points, match_result.matrix, match_result.f_ref_matrix);

    std::cout << std::endl << "F Match Result : " << std::endl;
    using match_map = std::map<std::pair<size_t, size_t>, common::vec1d<cv::DMatch> >;
    for (match_map::const_iterator citr = match_result.f_ref_matrix.cbegin();
         citr != match_result.f_ref_matrix.cend();
         citr++) {
      std::cout << citr->second.size() << std::endl;
    }    

  }

  // 10. Draw and show matched images.
  {
    std::cout << std::endl << "10. Draw and show matched images." << std::endl;
    vis2d::draw_matched_imgs(
      images.org_imgs, features.key_points,
      match_result.f_ref_matrix, vis_imgs.refined_matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }

  // 11. Apply homography check for removing coplanar point set.
  {
    std::cout << std::endl << "11. Apply homography check for removing coplanar point set." << std::endl;
    cv::Ptr<IKeyPointFilter> homo_point_filter 
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::HOMOGRAPHY);
    homo_point_filter->run(
      features.key_points, match_result.f_ref_matrix, match_result.homo_ref_matrix);
  }

  // 12. Sort img pair in the order of the number of inliers from homography calc.
  std::vector<std::pair<int, std::pair<size_t, size_t> > > sorted_match_list;
  {
    std::cout << std::endl << "12. Sort img pair in the order of the number of inliers from homography calc." << std::endl;
    for (match_matrix::const_iterator citr = match_result.homo_ref_matrix.cbegin();
         citr != match_result.homo_ref_matrix.cend();
         citr++) {
      std::pair<size_t, size_t> key = citr->first;
      int h_inliers = citr->second.size();
      int original = match_result.f_ref_matrix[key].size();

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
  PointCloud cloud;
  CameraPose cam_poses;
  cam_poses.poses.resize(images.org_imgs.size());
  SfmState state;
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

      common::vec1d<cv::DMatch> matches = match_result.f_ref_matrix[citr->second];
      common::vec1d<cv::Point2f> aligned_point2f_list_train, aligned_point2f_list_query;
      common::container_util::create_point2f_list_aligned_with_matches(
        matches, 
        features.key_points[train_img_idx], features.key_points[query_img_idx],
        aligned_point2f_list_train, aligned_point2f_list_query);

      // Initialize P everytime.
      common::vec1d<common::Point3dWithRepError> tmp_point3d_w_reperr_list;
      cv::Matx34d tmp_P(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
      cam_mat_result = cam_mat_result || geometry::find_camera_matrix(
        config.cam_intr, 
        aligned_point2f_list_train, 
        aligned_point2f_list_query,
        Porigin, tmp_P, tmp_point3d_w_reperr_list);

      if(cam_mat_result) {
        P = tmp_P;
        train_img = train_img_idx;
        query_img = query_img_idx;
        state.adopted_view.insert(train_img_idx);
        state.adopted_view.insert(query_img_idx);
        state.pose_recovered_view.insert(train_img_idx);
        state.pose_recovered_view.insert(query_img_idx);
        state.processed_view.insert(train_img_idx);
        state.processed_view.insert(query_img_idx);

        std::cout << std::endl << "Camera Matrix Found. P : " << std::endl;
        std::cout << P << std::endl;

        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          images.org_imgs.size(), tmp_point3d_w_reperr_list, cloud.point_cloud
        );

        for (size_t idx = 0; idx < cloud.point_cloud.size(); idx++) {
          cloud.point_cloud[idx].idx_in_img.resize(images.org_imgs.size(), -1);
          if (cloud.point_cloud[idx].pt.valid) {
            cloud.point_cloud[idx].idx_in_img[train_img_idx] = matches[idx].trainIdx;
            cloud.point_cloud[idx].idx_in_img[query_img_idx] = matches[idx].queryIdx;
          }
        }
        break;
      }
    }

    if (!cam_mat_result) {
      std::cout << "Failed to find Camera Matrix.... Exit here." << std::endl;
      std::exit(0);
    }
    cam_poses.poses[train_img] = Porigin;
    cam_poses.poses[query_img] = P;
  }

  // Draw Result via PCL Viewer.
  {
    vec1d<cv::Vec3b> rgb_clrs;
    vec1d<cv::Point3d> point3d_list;
    create_rgb_vector_from_point_cloud(
      features.key_points, images.org_imgs, cloud.point_cloud, rgb_clrs);
    container_util::convert_cloud_point_list_to_point3d_list(
      cloud.point_cloud, point3d_list);
    viewer.update(point3d_list, rgb_clrs, cam_poses.poses);

    cv::Mat dummy(100, 100, CV_8UC1);
    cv::imshow("dummy", dummy);
    cv::waitKey(0);

  }



  // Accumulation Phase.
  {
    common::vec2d<cv::Point2f> point2d_list(images.org_imgs.size());
    for (size_t idx = 0; idx < images.org_imgs.size(); idx++) {
      container_util::convert_key_point_list_to_point2f_list(
        features.key_points[idx], point2d_list[idx]);
    }

    // Triangulation with the all processed views.
    while (state.processed_view.size() < images.org_imgs.size()) {

      common::vec1d<cv::Point2f> corresp_2d_pnts;
      common::vec1d<cv::Point3f> corresp_3d_pnts;

      int query_img_idx = find_best_matching_img_with_current_cloud(
        state.processed_view,
        state.adopted_view,
        match_result.f_ref_matrix,
        point2d_list,
        cloud.point_cloud,
        corresp_2d_pnts,
        corresp_3d_pnts);

      if (query_img_idx == -1) {
        break;
      }

      // Register this img as "Processed"
      state.processed_view.insert(query_img_idx);

      cv::Matx34d Pnew;
      bool pose_estimated = find_camera_matrix_via_pnp(
        config.cam_intr, corresp_2d_pnts, corresp_3d_pnts, Pnew);
      if (!pose_estimated) {
        continue;
      }
      state.pose_recovered_view.insert(query_img_idx);
      cam_poses.poses[query_img_idx] = Pnew;

      std::cout << std::endl << "Found camera matrices ";
      for (std::set<size_t>::const_iterator citr = state.pose_recovered_view.cbegin();
           citr != state.pose_recovered_view.cend();
           citr++) {
        std::cout << std::endl << "P : " << std::endl << cam_poses.poses[*citr];
      }

      // Triangulation with already adopted views.
      for (std::set<size_t>::const_iterator citr = state.adopted_view.cbegin();
           citr != state.adopted_view.cend();
           citr++) {
        
        size_t train_img_idx = *citr;
        if (query_img_idx == train_img_idx) {
          continue;
        }

        common::vec1d<cv::Point2f> aligned_point2d_list_query, aligned_point2d_list_train;
        
        common::container_util::create_point2f_list_aligned_with_matches(
          match_result.f_ref_matrix[std::make_pair(train_img_idx, query_img_idx)], 
          features.key_points[train_img_idx], features.key_points[query_img_idx],
          aligned_point2d_list_train, aligned_point2d_list_query);

        common::vec1d<Point3dWithRepError> tmp_point3d_w_reperr;
        bool tri_result 
          = triangulate_points_and_validate(
              aligned_point2d_list_train,
              aligned_point2d_list_query,
              config.cam_intr,
              cam_poses.poses[train_img_idx],
              cam_poses.poses[query_img_idx],
              tmp_point3d_w_reperr);

        common::vec1d<CloudPoint> cp_triangulated;
        common::container_util::convert_point3d_w_reperr_list_to_cloud_point_list(
          images.org_imgs.size(), tmp_point3d_w_reperr, cp_triangulated
        );

        if (!tri_result) {
          continue;
        }

        std::pair<size_t, size_t> key = std::make_pair(train_img_idx, query_img_idx);
        common::vec1d<cv::DMatch> match = match_result.f_ref_matrix[key];
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
          for (size_t old_cp_idx = 0; old_cp_idx < cloud.point_cloud.size(); old_cp_idx++) {
            
            int old_cp_query_idx = cloud.point_cloud[old_cp_idx].idx_in_img[train_img_idx];

            // If this cp exists already.
            if (old_cp_query_idx == new_cp_query_idx) {
              cloud.point_cloud[old_cp_idx].idx_in_img[query_img_idx] = new_cp_query_idx;
              continue;
            }
            
            if (added_point.find(new_cp_train_idx) != added_point.end()) {
              continue;
            }

            // This point new and has to be added.
            CloudPoint cp(images.org_imgs.size());
            cp.idx_in_img[train_img_idx] = new_cp_train_idx;
            cp.idx_in_img[query_img_idx] = new_cp_query_idx;
            cp.pt = cp_triangulated[new_cp_idx].pt;
            add_to_cloud.push_back(cp);
            added_point.insert(new_cp_train_idx);
          }
        }
        std::cout << std::endl << "Image Pair (" << query_img_idx << ", " << *citr << ")" << std::endl;
        std::cout << "Original : " << cloud.point_cloud.size() << ", Added : " << add_to_cloud.size() << std::endl;
        cloud.point_cloud.reserve(cloud.point_cloud.size() + add_to_cloud.size());
        cloud.point_cloud.insert(cloud.point_cloud.end(), add_to_cloud.begin(), add_to_cloud.end());
        add_to_cloud.clear();
        added_point.clear();
      }

      state.adopted_view.insert(query_img_idx);


      // Draw Result via PCL Viewer.
      {
        vec1d<cv::Vec3b> rgb_clrs;
        vec1d<cv::Point3d> point3d_list;
        create_rgb_vector_from_point_cloud(
          features.key_points, images.org_imgs, cloud.point_cloud, rgb_clrs);
        container_util::convert_cloud_point_list_to_point3d_list(
          cloud.point_cloud, point3d_list);
        viewer.update(point3d_list, rgb_clrs, cam_poses.poses);

        cv::Mat dummy(100, 100, CV_8UC1);
        cv::imshow("dummy", dummy);
        cv::waitKey(0);


      }

    }
  }


  // 14. Wait till visualization thread gets joined.
  viewer.wait_vis_thread();

  return 0;
}

}

int main(int argc, char** argv) {
  std::cout << __FILE__ << " starts!" << std::endl;
  sfm_run(argc, argv);
  return 0;
}
