

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// STL
#include <string>

// Original
#include <app/app_container.h>

namespace simple_sfm {
namespace app {

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

}
}
