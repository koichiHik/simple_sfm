

// Original
#include <utility/vis2dutil.h>

namespace simple_sfm {
namespace utility {
namespace vis2d { 

void resize_and_show(const cv::Mat& image, 
                     const std::string window_name, 
                     double scale, int delay, bool dest_window) {

  if (dest_window) {
    cv::namedWindow(window_name);
  }

  cv::Mat scaled_img;
  cv::resize(image, scaled_img, cv::Size(), scale, scale);
  cv::imshow(window_name, scaled_img);
  cv::waitKey(delay);

  if (dest_window) {
    cv::destroyWindow(window_name);
  }
}

void resize_and_show(const common::vec1d<cv::Mat>& images, 
                     const std::string window_name, 
                     double scale, int delay) {

  for (auto img : images) {
    resize_and_show(img, window_name, scale, delay, false);
  }
  cv::destroyWindow(window_name);
}

void draw_key_points(const common::vec1d<cv::Mat>& images,
                     const common::vec2d<cv::KeyPoint>& key_points,
                     common::vec1d<cv::Mat>& result_images,
                     double scale, int delay, bool show) {

  if (images.size() != result_images.size()) {
    result_images.resize(images.size());
  }

  std::string window_name = "Key Points";
  if (show) {
    cv::namedWindow(window_name);
  }
  for (int i = 0; i < images.size(); i++) {
    cv::drawKeypoints(images[i], 
                      key_points[i], 
                      result_images[i], 
                      cv::Scalar(0, 180, 0), 
                      cv::DrawMatchesFlags::DEFAULT);

    if (show) {
      std::string text = 
        "Key points for image " + std::to_string(i);
      resize_and_show(result_images[i], window_name, scale, delay, false);
    }
  }
  if (show) {
    cv::destroyWindow(window_name);
  }
}

void draw_matched_imgs(const common::vec1d<cv::Mat>& img_list,
                       const common::vec2d<cv::KeyPoint>& key_point_list,
                       const common::match_matrix& match_mat,
                       common::matching_map<cv::Mat>& matched_imgs,
                       const cv::Scalar& match_clr,
                       const cv::Scalar& single_clr,
                       int flags, double scale, int delay, bool show) {

  size_t img_num = img_list.size();
  matched_imgs.clear();

  std::string window_name = "Matched Image";
  if (show) {  
    cv::namedWindow(window_name);
  }

  for (size_t i = 0; i < img_num - 1; i++) {
    for (size_t j = i + 1; j < img_num; j++) {
      const std::pair<size_t, size_t> key = std::make_pair(i, j);
      matched_imgs[key] = cv::Mat();
      const common::vec1d<cv::DMatch>& match 
          = common::getMapValue(match_mat, key);
      //std::cout << "(i, j)" << i << ", " << j << std::endl;
      //std::cout << "before drawMatches" << std::endl;
      cv::drawMatches(img_list[i], key_point_list[i],
                      img_list[j], key_point_list[j],
                      match, matched_imgs[key],
                      match_clr, single_clr, 
                      std::vector<char>(), flags);
      //std::cout << "after drawMatches" << std::endl;     
      if (show) {
        std::string text = 
          "Match between " + std::to_string(i) + " vs " + std::to_string(j);
        resize_and_show(matched_imgs[key], window_name, scale, delay, false);
      }
      //std::cout << "after show" << std::endl;     
    }
  }

  if (show) {
    cv::destroyWindow(window_name);
  }

}

} // vis2d
} // utility
} // simple_sfm
