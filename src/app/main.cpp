
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
#include <utility/fileutil.h>
#include <utility/vis2dutil.h>
#include <feature/IFeatureCalculatorFactory.h>

// Original namespace
using namespace simple_sfm;
using namespace simple_sfm::utility;
using namespace simple_sfm::feature;
using namespace simple_sfm::common;

struct ActivationConfig {
  std::string img_dir;
  std::vector<std::string> img_path_list;
  std::vector<std::string> img_filename_list;
};

struct Images {
  std::vector<cv::Mat> org_imgs;
  std::vector<cv::Mat> gray_imgs;
};

struct Features {
  vec2d<cv::KeyPoint> key_points;
  vec1d<cv::Mat> descriptors;
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
  std::cout << "2. Collect all paths for image." << std::endl;  
  std::vector<std::string> img_list{}, extensions{".jpg", ".png"};
  file::raise_all_files_in_directory(config.img_dir, 
                               config.img_path_list, 
                               config.img_filename_list,
                               extensions);

  // 3. Load Images.
  std::cout << "3. Load Images." << std::endl;
  Images images;
  file::load_images(config.img_path_list, images.org_imgs, images.gray_imgs);

  // 4. Display Image.
  std::cout << "4. Display Image." << std::endl;
  vis2d::resize_and_show(images.org_imgs, "Original Image", 0.25, 0);

  // 5. Feature detection and Descriptor extraction.
  std::cout << "5. Feature detection and Descriptor extraction." << std::endl;  
  Features features;
  cv::Ptr<IFeatureCalculator> feature_calculator 
      = IFeatureCalculatorFactory::createFeatureCalculator(FeatureType::GPU_SURF);
  feature_calculator->detectAndComputeForMultipleImgs(
    images.gray_imgs, features.key_points, features.descriptors);

  // 6. Draw Feature and Display.
  std::cout << "6. Draw Feature and Display." << std::endl;



  return 0;
}