
#ifndef MATCH_FILTERING_RUNNER_H
#define MATCH_FILTERING_RUNNER_H

// STL
#include <memory>

// Original
#include <app/app_container.h>

namespace simple_sfm {
namespace app {

struct MatchFilteringConst : public AlgoIF {

  MatchFilteringConst(const SfmDB& db) :
    key_points(db.feature_match.key_points),
    matrix(db.feature_match.matrix)
  {}

  const common::vec2d<cv::KeyPoint>& key_points;
  const common::match_matrix& matrix;
};

struct MatchFiltering : public AlgoIF {

  MatchFiltering(SfmDB& db) :
    f_ref_matrix(db.feature_match.f_ref_matrix),
    homo_ref_matrix(db.feature_match.homo_ref_matrix)
  {}
  
  common::match_matrix& f_ref_matrix;
  common::match_matrix& homo_ref_matrix;
};

struct MatchFilteringRunnerInternalStorage;
class MatchFilteringRunner {
public:
  MatchFilteringRunner();

  ~MatchFilteringRunner();

  bool Initialize();

  bool Run(
        const AlgoIF& constState,
        AlgoIF& state);

  bool Terminate();

private:
  std::unique_ptr<MatchFilteringRunnerInternalStorage> m_intl;
};

} // app
} // simple_sfm

#endif // MATCH_FILTERING_RUNNER_H
