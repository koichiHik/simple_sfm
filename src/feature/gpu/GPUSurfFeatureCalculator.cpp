
// System
#include <iostream>

// STL
#include <memory>

// OpenCV
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

// Original
#include <feature/ConcreteFeatureCalculator.h>

namespace simple_sfm {
namespace feature {

struct GPUSurfFeatureCalculatorInternalStorage {
  cv::gpu::SURF_GPU m_extractor;
  //std::vector<cv::gpu::GpuMat> m_gpu_imgs;
  //std::vector<cv::gpu::GpuMat> m_gpu_descriptors;
};

GPUSurfFeatureCalculator::GPUSurfFeatureCalculator() {
  m_intl.reset(new GPUSurfFeatureCalculatorInternalStorage());
}

GPUSurfFeatureCalculator::~GPUSurfFeatureCalculator() 
{}

void GPUSurfFeatureCalculator::detectAndComputeForSingleImg (
          const cv::Mat& image,
          vec1d<cv::KeyPoint>& key_points,
          cv::Mat& descriptors
          ) {


}

void GPUSurfFeatureCalculator::detectAndComputeForMultipleImgs (
          const vec1d<cv::Mat>& images,
          vec2d<cv::KeyPoint>& key_points,
          vec1d<cv::Mat>& descriptors
          ) {

  std::cout << "GPUSurfFeatureCalculator::detectAndComputeForMultipleImgs" << std::endl;

  // Initialize and resize
  key_points.clear();
  descriptors.clear();
  key_points.resize(images.size());
  descriptors.resize(images.size());

  // Feature extraction.
  for (size_t i = 0; i < images.size(); i++) {
    std::cout << "Feature Extraction : " << i << "/" << images.size() << std::endl;
    cv::gpu::GpuMat g_img;
    cv::gpu::GpuMat g_descriptor;
    g_img.upload(images[i]);
    m_intl->m_extractor(g_img, cv::gpu::GpuMat(), key_points[i], g_descriptor);
    g_descriptor.download(descriptors[i]);
  }
}

}
}