
// Self Header
#include <app/sfm_runner.h>





// Original
#include <vis3d/PCLViewer.h>
#include <app/point2d_matching_runner.h>
#include <app/match_filtering_runner.h>
#include <app/fmat_point3d_gen_runner.h>
#include <app/pnp_point3d_gen_runner.h>
#include <utility/fileutil.h>

namespace simple_sfm {
namespace app {

struct SfMRunnerInternalStorage {

  SfMRunnerInternalStorage(const SfmConfig& config) :
    db(config),
    viewer("Simple SFM Viewer", db.images.org_imgs, db.feature_match.key_points),
    point2d_matching_const_if(db),
    point2d_matching_if(db),
    match_filtering_const_if(db),
    match_filtering_if(db),
    fmat_point3d_gen_const_if(db),
    fmat_point3d_gen_if(db),
    pnp_point3d_gen_const_if(db),
    pnp_point3d_gen_if(db)
  {}

  // DB
  SfmDB db;

  // Viewer
  vis3d::PCLViewer viewer;

  // Runner
  Point2DMatchingRunner point2d_matching_runner;
  MatchFilteringRunner match_filtering_runner;
  FMatPoint3DGenRunner fmat_point3d_gen_runner;
  PNPPoint3DGenRunner pnp_point3d_gen_runner;

  // Interface Object.
  Point2DMatchingConst point2d_matching_const_if;
  Point2DMatching point2d_matching_if;
  MatchFilteringConst match_filtering_const_if;
  MatchFiltering match_filtering_if;
  FMatPoint3DGenConst fmat_point3d_gen_const_if;
  FMatPoint3DGen fmat_point3d_gen_if;
  PNPPoint3DGenGenConst pnp_point3d_gen_const_if;
  PNPPoint3DGenGen pnp_point3d_gen_if;
};

SfMRunner::SfMRunner() :
  m_intl(nullptr)
{}

SfMRunner::~SfMRunner()
{}

bool SfMRunner::Initialize(SfmConfig& config) {
  m_intl.reset(new SfMRunnerInternalStorage(config));

  m_intl->db.sfm_result.AddListener(&m_intl->viewer);

  utility::file::load_images(
    m_intl->db.config.img_path_list, m_intl->db.images.org_imgs, m_intl->db.images.gray_imgs);

  m_intl->point2d_matching_runner.Initialize();

  m_intl->match_filtering_runner.Initialize();

  m_intl->fmat_point3d_gen_runner.Initialize();

  m_intl->pnp_point3d_gen_runner.Initialize();

  m_intl->viewer.run_visualization_async();

  return true;
}

bool SfMRunner::Run() {

  bool result = false;

  result = m_intl->point2d_matching_runner.Run(
    m_intl->point2d_matching_const_if,
    m_intl->point2d_matching_if);

  if (!result) {
    return false;
  }

  result = m_intl->match_filtering_runner.Run(
    m_intl->match_filtering_const_if,
    m_intl->match_filtering_if);

  if (!result) {
    return false;
  }

  result = m_intl->fmat_point3d_gen_runner.Run(
    m_intl->fmat_point3d_gen_const_if,
    m_intl->fmat_point3d_gen_if);

  if (!result) {
    return false;
  }

  result = m_intl->pnp_point3d_gen_runner.Run(
    m_intl->pnp_point3d_gen_const_if,
    m_intl->pnp_point3d_gen_if);

  if (!result) {
    return false;
  }

  return true;
}

bool SfMRunner::Terminate() {

  m_intl->viewer.wait_vis_thread();

  m_intl->pnp_point3d_gen_runner.Terminate();

  m_intl->fmat_point3d_gen_runner.Terminate();

  m_intl->match_filtering_runner.Terminate();

  m_intl->point2d_matching_runner.Terminate();

  m_intl.reset(nullptr);
  return true;
}

SfmDB& SfMRunner::GetSfmDB() {
  return m_intl->db;
}

}
}