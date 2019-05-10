
// System Library
#include <iostream>
#include <string>

// STL
#include <vector>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// Original
#include <common/container.h>
#include <utility/fileutil.h>
#include <app/sfm_runner.h>

// Original namespace
using namespace simple_sfm::app;
using namespace simple_sfm::utility;

namespace {

bool parseArgment(int argc, char** argv, SfmConfig &config) {
  namespace po = boost::program_options;
  po::options_description opt("Option");
  po::variables_map map;

  opt.add_options()
      ("help,h", "Display help")
      ("directory,d", po::value<std::string>(), "Directory path containing images.");

  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt), map);
    boost::program_options::notify(map);

    // Cast element for value aquisition.
    if (map.count("help") || !map.count("directory")) {
      std::cout << opt << std::endl;
      return false;
    } else {
      config.img_dir = map["directory"].as<std::string>();
    }

  } catch(const boost::program_options::error_with_option_name& e) {
    std::cout << e.what() << std::endl;
    return false;
  } catch (boost::bad_any_cast& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

int SimpleSfMRun(int argc, char** argv) {

  // 1. Parse User Specified Argument.
  SfmConfig config;
  if (!parseArgment(argc, argv, config)) {
    return 0;
  }

  // 2. Collect all paths for image and read calib info.
  {
    std::cout << std::endl << "Collect all paths for image and read calibration file." << std::endl;  
    std::vector<std::string> img_list{}, extensions{".jpg", ".png"};
    file::raise_all_img_files_in_directory(config.img_dir, 
                                config.img_path_list, 
                                extensions);
    file::read_calib_file(config.img_dir + "/K.txt", config.cam_intr);
  }

  SfMRunner runner;

  runner.Initialize(config);
  runner.Run();
  runner.Terminate();

  return 0;
}

}

int main(int argc, char** argv) {
  std::cout << __FILE__ << " starts!" << std::endl;
  SimpleSfMRun(argc, argv);
  return 0;
}
