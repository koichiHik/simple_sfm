
// Self Header
#include <app/point2d_matching_runner.h>

// Original
#include <feature_extractor/IFeatureExtractor.h>
#include <feature_extractor/IFeatureExtractorFactory.h>
#include <feature_extractor/ConcreteFeatureExtractor.h>
#include <descriptor_matcher/IDescriptorMatcher.h>
#include <descriptor_matcher/IDescriptorMatcherFactory.h>
#include <descriptor_matcher/ConcreteDescriptorMatcher.h>


using namespace simple_sfm::feature_extractor;
using namespace simple_sfm::descriptor_matcher;

namespace simple_sfm {
namespace app {

struct Point2DMatchingRunnerInternalStorage {

};

Point2DMatchingRunner::Point2DMatchingRunner() :
  m_intl(nullptr)
{}

Point2DMatchingRunner::~Point2DMatchingRunner()
{}

bool Point2DMatchingRunner::Initialize() {

  m_intl.reset(new Point2DMatchingRunnerInternalStorage());

  return true;
}

bool Point2DMatchingRunner::Run(
      const AlgoIF& constState,
      AlgoIF& state) {

  const Point2DMatchingConst& c_interface = static_cast<const Point2DMatchingConst &>(constState);
  Point2DMatching interface = static_cast<Point2DMatching &>(state);

  // 5. Feature detection and Descriptor extraction.
  //FeatureMatching feature_match;
  {
    std::cout << std::endl << "5. Feature detection and Descriptor extraction." << std::endl;  
    cv::Ptr<feature_extractor::IFeatureExtractor> feature_extractor 
        = IFeatureExtractorFactory::createFeatureExtractor(FeatureType::GPU_SURF);
    feature_extractor->detectAndCompute(
      c_interface.gray_imgs, interface.key_points, interface.descriptors);
  }

  #if 0
  // 6. Draw Feature and Display.
  {
    std::cout << std::endl << "6. Draw Feature and Display." << std::endl;
    vis2d::draw_key_points(db.images.org_imgs, db.feature_match.key_points, db.images.key_pnt_imgs);
  }
  #endif

  // 7. Match key points of each image pairs.
  {
    std::cout << std::endl << "7. Match key points of each image pairs." << std::endl;
    cv::Ptr<IDescriptorMatcher> descriptor_matcher
        = IDescriptorMatcherFactory::createDescriptorMatcher(MatcherType::GPU_BF_RATIO_CHECK);
    descriptor_matcher->createMatchingMatrix(interface.descriptors, interface.matrix);
  }

  return true;
}

bool Point2DMatchingRunner::Terminate() {

  m_intl.reset(nullptr);

  return true;
}

}
}