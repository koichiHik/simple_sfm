
#ifndef SFM_RUNNER_H
#define SFM_RUNNER_H

// STL
#include <memory>

// Original
#include <app/app_container.h>
#include <app/app_util.h>

namespace simple_sfm {
namespace app {

struct SfMRunnerInternalStorage;
class SfMRunner {
public:
  SfMRunner();

  ~SfMRunner();

  bool Initialize(SfmConfig& config);

  bool Run();

  bool Terminate();

  SfmDB& GetSfmDB();

private:
  std::unique_ptr<SfMRunnerInternalStorage> m_intl;
};

} // app
} // simple_sfm

#endif // SFM_RUNNER_H
