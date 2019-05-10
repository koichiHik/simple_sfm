
// Self Header
#include <app/match_filtering_runner.h>

// Original
#include <key_point_filter/i_key_point_filter.h>
#include <key_point_filter/i_key_point_filter_factory.h>
#include <key_point_filter/concrete_key_point_filter.h>

using namespace simple_sfm::key_point_filter;

namespace simple_sfm {
namespace app {

struct MatchFilteringRunnerInternalStorage {

};

MatchFilteringRunner::MatchFilteringRunner() :
  m_intl(nullptr)
{}

MatchFilteringRunner::~MatchFilteringRunner()
{}

bool MatchFilteringRunner::Initialize() {

  m_intl.reset(new MatchFilteringRunnerInternalStorage());


  return true;
}

bool MatchFilteringRunner::Run(
      const AlgoIF& constState,
      AlgoIF& state) {

  const MatchFilteringConst& c_interface = static_cast<const MatchFilteringConst &>(constState);
  MatchFiltering& interface = static_cast<MatchFiltering &>(state);

  // 9. Filtering by F matrix calculation.
  {
    std::cout << std::endl << "9. Filtering by F matrix calculation." << std::endl;
    cv::Ptr<IKeyPointFilter> fmat_point_filter
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::F_MAT);
    fmat_point_filter->run(
      c_interface.key_points, c_interface.matrix, interface.f_ref_matrix);

    std::cout << std::endl << "F Match Result : " << std::endl;
    using match_map = std::map<std::pair<size_t, size_t>, common::vec1d<cv::DMatch> >;
    for (match_map::const_iterator citr = interface.f_ref_matrix.cbegin();
         citr != interface.f_ref_matrix.cend();
         citr++) {
      std::cout << citr->second.size() << std::endl;
    }    

  }

  // 11. Apply homography check for removing coplanar point set.
  {
    std::cout << std::endl << "11. Apply homography check for removing coplanar point set." << std::endl;
    cv::Ptr<IKeyPointFilter> homo_point_filter 
        = IKeyPointFilterFactory::createKeyPointFilter(FilterType::HOMOGRAPHY);
    homo_point_filter->run(
      c_interface.key_points, interface.f_ref_matrix, interface.homo_ref_matrix);
  }

  return true;
}

bool MatchFilteringRunner::Terminate() {


  m_intl.reset(nullptr);

  return true;
}

}
}