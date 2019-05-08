
#ifndef FILEUTIL_H
#define FILEUTIL_H

// System Library
#include <string>

// STL
#include <vector>

// OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace utility {
namespace file {

void load_images(const std::vector<std::string>& img_path_list, 
                 std::vector<cv::Mat>& org_imgs, 
                 std::vector<cv::Mat>& gray_imgs);

void raise_all_img_files_in_directory(
      const std::string& dirpath, 
      std::vector<std::string>& img_path_list, 
      std::vector<std::string>& img_filename_list,
      const std::vector<std::string>& exts = std::vector<std::string>());

bool read_calib_file(
      const std::string& filepath,
      common::CamIntrinsics& cam_intr);

}
}
}

#endif // FILEUTIL_H
