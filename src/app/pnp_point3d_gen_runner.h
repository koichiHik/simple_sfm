
#ifndef PNP_POINT3D_GEN_RUNNER_H
#define PNP_POINT3D_GEN_RUNNER_H

// STL
#include <memory>


// Original
#include <app/app_container.h>

namespace simple_sfm {
namespace app {

struct PNPPoint3DGenGenConst : public AlgoIF {

  PNPPoint3DGenGenConst(const SfmDB& db) :
    cam_intr(db.config.cam_intr),
    img_path_list(db.config.img_path_list),
    point2f_lists(db.feature_match.point2f_lists),
    f_ref_matrix(db.feature_match.f_ref_matrix)
  {}

  const common::CamIntrinsics& cam_intr;  
  const common::vec1d<std::string>& img_path_list;
  const common::vec2d<cv::Point2f>& point2f_lists;
  const common::match_matrix& f_ref_matrix;
};

struct PNPPoint3DGenGen : public AlgoIF {

  PNPPoint3DGenGen(SfmDB& db) :
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

struct PNPPoint3DGenRunnerInternalStorage; 
class PNPPoint3DGenRunner {
public:
  PNPPoint3DGenRunner();

  ~PNPPoint3DGenRunner();

  bool Initialize();

  bool Run(
        const AlgoIF& constState,
        AlgoIF& state);

  bool Terminate();

private:
  std::unique_ptr<PNPPoint3DGenRunnerInternalStorage> m_intl;
};

} // app
} // simple_sfm

#endif // PNP_POINT3D_GEN_RUNNER_H
