
// System Library
#include <iostream>
#include <string>

// STL
#include <vector>

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
#include <utility/fileutil.h>
#include <utility/vis2dutil.h>
#include <feature_extractor/IFeatureExtractorFactory.h>
#include <descriptor_matcher/IDescriptorMatcherFactory.h>
#include <key_point_filter/IKeyPointFilterFactory.h>
#include <geometry/Geometry.h>

// Original namespace
using namespace simple_sfm;
using namespace simple_sfm::common;
using namespace simple_sfm::utility;
using namespace simple_sfm::feature_extractor;
using namespace simple_sfm::descriptor_matcher;
using namespace simple_sfm::key_point_filter;
using namespace simple_sfm::geometry;

struct ActivationConfig {
  std::string img_dir;
  vec1d<std::string> img_path_list;
  vec1d<std::string> img_filename_list;
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

bool parseArgment(int argc, char** argv, ActivationConfig &config) {
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

int main(int argc, char** argv) {
  std::cout << __FILE__ << " starts!" << std::endl;

  // 1. Parse User Specified Argument.
  ActivationConfig config;
  if (!parseArgment(argc, argv, config)) {
    return 0;
  }

  // 2. Collect all paths for image.
  {
    std::cout << "2. Collect all paths for image." << std::endl;  
    std::vector<std::string> img_list{}, extensions{".jpg", ".png"};
    file::raise_all_files_in_directory(config.img_dir, 
                                config.img_path_list, 
                                config.img_filename_list,
                                extensions);
  }

  // 3. Load Images.
  Images images;
  {
    std::cout << "3. Load Images." << std::endl;
    file::load_images(config.img_path_list, images.org_imgs, images.gray_imgs);
  }

  // 4. Display Image.
  {
    std::cout << "4. Display Image." << std::endl;
    vis2d::resize_and_show(images.org_imgs, "Original Image", 0.25, 100);
  }

  // 5. Feature detection and Descriptor extraction.
  Features features;
  {
    std::cout << "5. Feature detection and Descriptor extraction." << std::endl;  
    cv::Ptr<IFeatureExtractor> feature_extractor 
        = IFeatureExtractorFactory::createFeatureExtractor(FeatureType::GPU_SURF);
    feature_extractor->detectAndCompute(
      images.gray_imgs, features.key_points, features.descriptors);
  }

  // 6. Draw Feature and Display.
  VisImages vis_imgs;
  {
    std::cout << "6. Draw Feature and Display." << std::endl;
    vis2d::draw_key_points(images.org_imgs, features.key_points, vis_imgs.key_pnt_imgs);
  }

  // 7. Match key points of each image pairs.
  MatchingResult match_result;
  {
    cv::Ptr<IDescriptorMatcher> descriptor_matcher
        = IDescriptorMatcherFactory::createDescriptorMatcher(MatcherType::GPU_BF_RATIO_CHECK);
    descriptor_matcher->createMatchingMatrix(features.descriptors, match_result.matrix);
  }

  // 8. Draw and show matched images.
  {
    vis2d::draw_matched_imgs(
      images.org_imgs, features.key_points, 
      match_result.matrix, vis_imgs.matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }

  // 9. Filtering by F matrix calculation.
  {
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    fmat_point_filter->run(
      features.key_points, match_result.matrix, match_result.f_ref_matrix);
  }

  // 10. Draw and show matched images.
  {
    vis2d::draw_matched_imgs(
      images.org_imgs, features.key_points,
      match_result.f_ref_matrix, vis_imgs.refined_matched_imgs,
      cv::Scalar(255, 0, 0), cv::Scalar::all(0));
  }

  // 11. Apply homography check for removing coplanar point set.
  {
    cv::Ptr<IKeyPointFilter> homo_point_filter 
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::HOMOGRAPHY);
    homo_point_filter->run(
      features.key_points, match_result.f_ref_matrix, match_result.homo_ref_matrix);
  }

  // 12. Sort img pair in the order of the number of inliers from homography calc.
  std::vector<std::pair<int, std::pair<size_t, size_t> > > sorted_match_list;
  {
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

  // 13. Start baseline calculation.
  {
    cv::Matx34d P, Porigin;
    common::vec1d<cv::Point3f> point3d;
    common::CamIntrinsics cam_intr;
    using sorted_pair = vec1d<std::pair<int, std::pair<size_t, size_t> > >;
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    for (sorted_pair::const_iterator citr = sorted_match_list.cbegin();
         citr != sorted_match_list.cend();
         citr++) {
      size_t query_idx = citr->second.first;
      size_t train_idx = citr->second.second;
      common::vec1d<cv::DMatch> matches = match_result.f_ref_matrix[citr->second];
      common::vec1d<cv::Point2f> aligned_point2f_list1, aligned_point2f_list2;
      common::container_util::create_point2f_list_aligned_with_matches(
        matches, 
        features.key_points[query_idx], features.key_points[train_idx],
        aligned_point2f_list1, aligned_point2f_list2);

      geometry::find_camera_matrix(
        cam_intr, 
        aligned_point2f_list1, 
        aligned_point2f_list2,
        Porigin, P, point3d);
    }
  }

  return 0;
}
