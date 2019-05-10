
#ifndef IF_APP_VIEWER_H
#define IF_APP_VIEWER_H

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace app {

class SfmResultUpdateListener {
public:

  virtual ~SfmResultUpdateListener()
  {}

  virtual void Update(const common::vec1d<common::CloudPoint>& cloud,
                      const common::vec1d<cv::Matx34d>& poses) = 0;
};

}
}

#endif // IF_APP_VIEWER_H
