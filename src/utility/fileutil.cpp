
// System
#include <iostream>
#include <fstream>

// Boost
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

// Original
#include <utility/fileutil.h>
#include <common/container.h>

// Static Functions.
namespace {

void raise_all_files_in_directory_internal(
      const std::string& dirpath, 
      std::vector<std::string>& img_path_list, 
      std::vector<std::string>& img_filename_list) {

  namespace fs = boost::filesystem;
  const fs::path path(dirpath);

  BOOST_FOREACH(const fs::path& p, 
    std::make_pair(fs::directory_iterator(path), fs::directory_iterator())) {
    if (!fs::is_directory(p)) {
      img_filename_list.push_back(p.filename().string());
      img_path_list.push_back(fs::absolute(p).string());
    }
  }

  std::sort(img_filename_list.begin(), img_filename_list.end());
  std::sort(img_path_list.begin(), img_path_list.end());
}

}

namespace simple_sfm {
namespace utility {
namespace file {

void resize_and_show(const std::vector<cv::Mat>& images, 
                     const std::string window_name, 
                     double scale, int delay) {

  for (auto img : images) {
    cv::Mat scaled_img;
    cv::resize(img, scaled_img, cv::Size(), scale, scale);
    cv::imshow(window_name, scaled_img);
    cv::waitKey(delay);
  }
}

void load_images(const std::vector<std::string>& img_path_list, 
                 std::vector<cv::Mat>& org_imgs, 
                 std::vector<cv::Mat>& gray_imgs) {

  for (auto path : img_path_list) {
    cv::Mat img, gray_img;
    img = cv::imread(path, cv::IMREAD_COLOR);
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    org_imgs.push_back(img);
    gray_imgs.push_back(gray_img);
  }
}

void raise_all_img_files_in_directory(
      const std::string& dirpath, 
      std::vector<std::string>& img_path_list, 
      std::vector<std::string>& img_filename_list,
      const std::vector<std::string>& exts) {

  std::vector<std::string> tmp_img_path_list, tmp_file_name_list;
  raise_all_files_in_directory_internal(dirpath, tmp_img_path_list, tmp_file_name_list);

  for (int i = 0; i < tmp_img_path_list.size(); i++) {
    std::string abs_path = tmp_img_path_list[i];
    std::string filename = tmp_file_name_list[i];
    for (auto ext : exts) {
      if (abs_path.find(ext) == abs_path.size() - ext.size()) {
        img_path_list.push_back(abs_path);
        img_filename_list.push_back(filename);
        break;
      }
    }
  }
}

bool read_calib_file(
      const std::string& filepath,
      common::CamIntrinsics& cam_intr) {
  
  std::ifstream ifs(filepath);
  if (ifs.fail()) {
    std::cout << "Failed at read_calib_file" << std::endl;
    return false;
    
  }

  {
    std::string line;
    // 1st line
    getline(ifs, line);
    sscanf(line.c_str(), "%lf %lf %lf", &cam_intr.K(0,0), &cam_intr.K(0,1), &cam_intr.K(0,2));

    // 2nd line
    getline(ifs, line);
    sscanf(line.c_str(), "%lf %lf %lf", &cam_intr.K(1,0), &cam_intr.K(1,1), &cam_intr.K(1,2));

    // 2nd line
    getline(ifs, line);
    sscanf(line.c_str(), "%lf %lf %lf", &cam_intr.K(2,0), &cam_intr.K(2,1), &cam_intr.K(2,2));
  }
  cam_intr.Kinv = cam_intr.K.inv();

  std::cout << "Read CamIntrinsics is .... " << std::endl;
  std::cout << cam_intr.K << std::endl;

  return true;
}

}
}
}