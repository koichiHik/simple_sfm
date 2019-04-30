
#ifndef DEBUG_HELPER_H
#define DEBUG_HELPER_H

// System
#include <string>
#include <iostream>

namespace simple_sfm {
namespace common {
namespace debug_helper {

inline void print_debug_info(
              const std::string& filename,
              int line_no,
              const std::string& func_name,
              const std::string& comment) {

  std::cout << "[DEBUG] filename : " << filename << ", line ; "
  << std::to_string(line_no) << ", function : " << func_name << "  " << comment
  << std::endl;
}

}
}  
}

#endif // DEBUG_HELPER_H


