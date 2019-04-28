project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

# Use C++ 14
set(CMAKE_CXX_STANDARD 14)

# Necessary Libraries.
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
find_package(VTK REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS filesystem program_options)

# All header file inclusion from top level directory.
include_directories(${CMAKE_SOURCE_DIR}/src)

# Include Directories
add_subdirectory(src/common)
add_subdirectory(src/utility)
add_subdirectory(src/feature)
add_subdirectory(src/app)
