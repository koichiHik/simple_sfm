
#ifndef CONTAINER_H
#define CONTAINER_H

// STL
#include <vector>

namespace simple_sfm {
namespace common {

template <typename Value>
using vec1d = std::vector<Value>;

template <typename Value>
using vec2d = std::vector<std::vector<Value> >;

}
}

#endif //CONTAINER_H
