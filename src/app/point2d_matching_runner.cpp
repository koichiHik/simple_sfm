
// Self Header
#include <app/point2d_matching_runner.h>

// Original
#include <common/container_util.h>
#include <feature_extractor/i_feature_extractor.h>
#include <feature_extractor/i_feature_extractor_factory.h>
#include <feature_extractor/concrete_feature_extractor.h>
#include <descriptor_matcher/i_descriptor_matcher.h>
#include <descriptor_matcher/i_descriptor_matcher_factory.h>
#include <descriptor_matcher/concrete_descriptor_matcher.h>

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

  {
    interface.point2f_lists.resize(c_interface.gray_imgs.size());
    for (size_t idx = 0; idx < c_interface.gray_imgs.size(); idx++) {
      common::container_util::convert_key_point_list_to_point2f_list(
        interface.key_points[idx], interface.point2f_lists[idx]);
    }
  } 

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