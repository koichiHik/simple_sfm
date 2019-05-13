
// System
#include <iostream>

// STL
#include <memory>

// OpenCV
#include <opencv2/xfeatures2d/cuda.hpp>

// Original
#include <feature_extractor/concrete_feature_extractor.h>

namespace simple_sfm {
namespace feature_extractor {

struct GPUSurfFeatureExtractorInternalStorage {
  cv::cuda::SURF_CUDA m_extractor;
};

GPUSurfFeatureExtractor::GPUSurfFeatureExtractor() :
  m_intl(new GPUSurfFeatureExtractorInternalStorage())
{}

GPUSurfFeatureExtractor::~GPUSurfFeatureExtractor() 
{}

void GPUSurfFeatureExtractor::detectAndCompute(
          const cv::Mat& image,
          vec1d<cv::KeyPoint>& key_points,
          cv::Mat& descriptors
          ) {

    std::cout << "Feature Extraction for Single Image" << std::endl;
    cv::cuda::GpuMat g_img;
    cv::cuda::GpuMat g_descriptor;
    g_img.upload(image);
    m_intl->m_extractor(g_img, cv::cuda::GpuMat(), key_points, g_descriptor);
    g_descriptor.download(descriptors);

}

void GPUSurfFeatureExtractor::detectAndCompute(
          const vec1d<cv::Mat>& images,
          vec2d<cv::KeyPoint>& key_points,
          vec1d<cv::Mat>& descriptors
          ) {

  std::cout << "GPUSurfFeatureExtractor::detectAndComputeForMultipleImgs" << std::endl;

  // Initialize and resize
  key_points.clear();
  descriptors.clear();
  key_points.resize(images.size());
  descriptors.resize(images.size());

  // Feature extraction.
  for (size_t i = 0; i < images.size(); i++) {
    std::cout << "Feature Extraction : " << i << "/" << images.size() << std::endl;
    cv::cuda::GpuMat g_img;
    cv::cuda::GpuMat g_descriptor;
    g_img.upload(images[i]);
    m_intl->m_extractor(g_img, cv::cuda::GpuMat(), key_points[i], g_descriptor);
    g_descriptor.download(descriptors[i]);
  }
}

}
}