
// OpenCV
#include <opencv2/gpu/gpu.hpp>

// Original
#include <descriptor_matcher/ConcreteDescriptorMatcher.h>

namespace simple_sfm {
namespace descriptor_matcher {

struct GPUBruteForceMatcherInternalStorage {
cv::gpu::BruteForceMatcher_GPU<cv::L2<float> > m_matcher;
};

GPUBruteForceMatcher::GPUBruteForceMatcher() :
  m_intl(new GPUBruteForceMatcherInternalStorage())
{}

GPUBruteForceMatcher::~GPUBruteForceMatcher() 
{}

void GPUBruteForceMatcher::createMatchingMatrix(
                  const vec1d<cv::Mat>& descriptor_list,
                  common::match_matrix& match_matrix) {
  
  match_matrix.clear();
  size_t img_num = descriptor_list.size();

  for (int query_idx = 0; query_idx < img_num - 1; query_idx++) {
    for (int train_idx = query_idx + 1; train_idx < img_num; train_idx++) {
      vec1d<cv::DMatch> valid_match;
      vec2d<cv::DMatch> raw_match;
      match(descriptor_list[query_idx],
            descriptor_list[train_idx],
            raw_match);
      match_matrix[std::make_pair(query_idx, train_idx)] = valid_match;
      match_matrix[std::make_pair(train_idx, query_idx)] = flipMatches(valid_match); 
    }
  }
} 

void GPUBruteForceMatcher::match(
                const cv::Mat& query_descriptor,
                const cv::Mat& train_descriptor,
                vec2d<cv::DMatch>& matches) {

  matches.clear();
  matches.resize(1);

  cv::gpu::GpuMat gpu_query_descriptor, gpu_train_descriptor;
  cv::gpu::GpuMat pairIdx, distance, all_dist;

  m_intl->m_matcher.match(
        gpu_query_descriptor,
        gpu_train_descriptor,
        matches[0]);

}

}
}