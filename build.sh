#!/bin/bash

ROOT_3RD=~/workspace/3rdParty
CMAKE_BUILD_TYPE=Debug
OpenCV_DIR_RELEASE=${ROOT_3RD}/opencv331/install/share/
OpenCV_DIR_DEBUG=${ROOT_3RD}/opencv331/installd/share/
Eigen3_DIR=${ROOT_3RD}/eigen334/install/share/eigen3/cmake/

if [ CMAKE_BUILD_TYPE="Release" ]; then
  OpenCV_DIR=${OpenCV_DIR_RELEASE}
else
  OpenCV_DIR=${OpenCV_DIR_DEBUG}
fi

if [ ! -e ./build ]; then
  mkdir build
fi
cd build

cmake \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -D OpenCV_DIR=${OpenCV_DIR} \
  -D Eigen3_DIR=${Eigen3_DIR} \
  ../

make

cd ../
