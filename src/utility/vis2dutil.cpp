
#include <utility/vis2dutil.h>

namespace simple_sfm {
namespace utility {
namespace vis2d { 

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

}
}
}
