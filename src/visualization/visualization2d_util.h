
#ifndef VIS2DUTIL_H
#define VIS2DUTIL_H

// System Library
#include <string>

// STL
#include <vector>

// OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace visualization {

void resize_and_show(const common::vec1d<cv::Mat>& images, 
                     const std::string window_name, 
                     double scale, int delay);

void draw_key_points(const common::vec1d<cv::Mat>& images,
                     const common::vec2d<cv::KeyPoint>& key_points,
                     common::vec1d<cv::Mat>& result_images,
                     double scale = 0.25, int delay = 100, 
                     bool show = true);

void draw_matched_imgs(const common::vec1d<cv::Mat>& img_list,
                       const common::vec2d<cv::KeyPoint>& key_point_list,
                       const common::match_matrix& match_mat,
                       common::matching_map<cv::Mat>& matched_imgs,
                       const cv::Scalar& match_clr = cv::Scalar::all(-1),
                       const cv::Scalar& single_clr = cv::Scalar::all(-1),
                       int flags = cv::DrawMatchesFlags::DEFAULT,
                       double scale = 0.25, int delay = 100,
                       bool show = true);

}
}


#endif // VIS2DUTIL_H
