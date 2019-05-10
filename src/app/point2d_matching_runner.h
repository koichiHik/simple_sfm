
#ifndef POINT2D_MATCHING_RUNNER_H
#define POINT2D_MATCHING_RUNNER_H

// STL
#include <memory>



// Original
#include <app/app_container.h>

namespace simple_sfm {
namespace app {

struct Point2DMatchingConst : public AlgoIF {

  Point2DMatchingConst (const SfmDB& db) :
    gray_imgs(db.images.gray_imgs)
  {}
  const common::vec1d<cv::Mat>& gray_imgs;
};

struct Point2DMatching : public AlgoIF {

  Point2DMatching (SfmDB& db) :
    key_points(db.feature_match.key_points),
    descriptors(db.feature_match.descriptors),
    matrix(db.feature_match.matrix)
  {}
  common::vec2d<cv::KeyPoint>& key_points;
  common::vec1d<cv::Mat>& descriptors;
  common::match_matrix& matrix;
};

struct Point2DMatchingRunnerInternalStorage;
class Point2DMatchingRunner {
public:
  Point2DMatchingRunner();

  ~Point2DMatchingRunner();

  bool Initialize();

  bool Run(
        const AlgoIF& constState,
        AlgoIF& state);

  bool Terminate();

private:
  std::unique_ptr<Point2DMatchingRunnerInternalStorage> m_intl;
};

} // app
} // simple_sfm

#endif // POINT2D_MATCHING_RUNNER_H
