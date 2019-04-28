
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

namespace simple_sfm {
namespace utility {
namespace vis2d {

void resize_and_show(const std::vector<cv::Mat>& images, 
                     const std::string window_name, 
                     double scale, int delay);

}
}
}


#endif // VIS2DUTIL_H
