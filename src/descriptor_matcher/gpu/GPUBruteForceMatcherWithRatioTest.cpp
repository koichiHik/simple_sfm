
// System
#include <iostream>

// OpenCV
#include <opencv2/gpu/gpu.hpp>

// Original
#include <descriptor_matcher/ConcreteDescriptorMatcher.h>

namespace simple_sfm {
namespace descriptor_matcher {

static const float DIST_THRESH = 0.7;
static const int K = 2;

struct GPUBruteForceMatcherWithRatioCheckInternalStorage {
cv::gpu::BruteForceMatcher_GPU<cv::L2<float> > m_matcher;
};

//const float GPUBruteForceMatcherWithRatioCheckInternalStorage::dist_thresh = 0.7;
//const int GPUBruteForceMatcherWithRatioCheckInternalStorage::k = 2;

GPUBruteForceMatcherWithRatioCheck::GPUBruteForceMatcherWithRatioCheck() :
  m_intl(new GPUBruteForceMatcherWithRatioCheckInternalStorage())
{}

GPUBruteForceMatcherWithRatioCheck::~GPUBruteForceMatcherWithRatioCheck() 
{}

void GPUBruteForceMatcherWithRatioCheck::createMatchingMatrix(
                  const vec1d<cv::Mat>& descriptor_list,
                  common::match_matrix& match_matrix) {
  
  match_matrix.clear();
  size_t img_num = descriptor_list.size();

  for (int query_idx = 0; query_idx < img_num - 1; query_idx++) {

    std::cout << std::endl;
    std::cout << "Creating matching matrix : " << query_idx << " / " << img_num << std::endl;

    for (int train_idx = query_idx + 1; train_idx < img_num; train_idx++) {
      vec1d<cv::DMatch> valid_match;
      vec2d<cv::DMatch> knn_match;
      match(descriptor_list[query_idx],
            descriptor_list[train_idx],
            knn_match);

      for (int i = 0; i < knn_match.size(); i++) {
        // Distance ratio check.
        if (knn_match[i][0].distance / knn_match[i][1].distance < DIST_THRESH) {
          valid_match.push_back(knn_match[i][0]);
        }
      }
      match_matrix[std::make_pair(query_idx, train_idx)] = valid_match;
      match_matrix[std::make_pair(train_idx, query_idx)] = flipMatches(valid_match); 
      std::cout << "Match between " << query_idx << " vs " << train_idx << " : ";
      std::cout << "VALID / TOTAL : " << valid_match.size() << " / " << knn_match.size() << std::endl;
    }
  }
} 

void GPUBruteForceMatcherWithRatioCheck::match(
                const cv::Mat& query_descriptor,
                const cv::Mat& train_descriptor,
                vec2d<cv::DMatch>& matches) {

  matches.clear();

  cv::gpu::GpuMat gpu_query_descriptor, gpu_train_descriptor;
  cv::gpu::GpuMat pairIdx, distance, all_dist;

  gpu_query_descriptor.upload(query_descriptor);
  gpu_train_descriptor.upload(train_descriptor);

  m_intl->m_matcher.knnMatchSingle(
        gpu_query_descriptor, 
        gpu_train_descriptor,
        pairIdx, distance, all_dist, K);
  
  m_intl->m_matcher.knnMatchDownload(
        pairIdx, distance, matches); 
}

}
}