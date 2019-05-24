

#ifndef RESULT_WRITER_H_
#define RESULT_WRITER_H_

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace fileio {

bool SavePointCloudToPCDFile(const std::string& filepath,
                             const common::vec2d<cv::KeyPoint>& key_points,
                             const common::vec1d<cv::Mat>& original_imgs,
                             const common::vec1d<common::CloudPoint>& points);

bool SaveCameraPosesToPLYFile(const std::string& ply_filepath,
                              const common::vec1d<cv::Matx34d>& poses);

}  // namespace fileio
}  // namespace simple_sfm

#endif  // RESULT_WRITER_H_
