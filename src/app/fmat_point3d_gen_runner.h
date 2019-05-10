
#ifndef FMAT_POINT3D_GEN_RUNNER_H
#define FMAT_POINT3D_GEN_RUNNER_H

// STL
#include <memory>


// Original
#include <app/app_container.h>

namespace simple_sfm {
namespace app {

struct FMatPoint3DGenConst : public AlgoIF {

  FMatPoint3DGenConst(const SfmDB& db) :
    cam_intr(db.config.cam_intr),
    img_path_list(db.config.img_path_list),
    key_points(db.feature_match.key_points),
    f_ref_matrix(db.feature_match.f_ref_matrix),
    homo_ref_matrix(db.feature_match.homo_ref_matrix) 
  {}

  const common::CamIntrinsics& cam_intr;  
  const common::vec1d<std::string>& img_path_list;
  const common::vec2d<cv::KeyPoint>& key_points;
  const common::match_matrix& f_ref_matrix;
  const common::match_matrix& homo_ref_matrix;
};

struct FMatPoint3DGen : public AlgoIF {

  FMatPoint3DGen(SfmDB& db) :
    sfm_result(db.sfm_result),
    processed_view(db.algo_status.processed_view),
    pose_recovered_view(db.algo_status.pose_recovered_view),
    adopted_view(db.algo_status.adopted_view)
  {}  

  SfmResult& sfm_result;
  std::set<size_t>& processed_view;
  std::set<size_t>& pose_recovered_view;
  std::set<size_t>& adopted_view;
};


struct FMatPoint3DGenRunnerInternalStorage;
class FMatPoint3DGenRunner {
public:

  FMatPoint3DGenRunner();

  ~FMatPoint3DGenRunner();

  bool Initialize();

  bool Run(
        const AlgoIF& constState,
        AlgoIF& state);

  bool Terminate();

private:

  std::vector<std::pair<int, std::pair<size_t, size_t> > >
    SortMatchListWrtHomographyInliers(
      const common::match_matrix& f_ref_matrix,
      const common::match_matrix& homo_ref_matrix);

private:
  std::unique_ptr<FMatPoint3DGenRunnerInternalStorage> m_intl;
};

} // app
} // simple_sfm

#endif // FMAT_POINT3D_GEN_RUNNER_H
