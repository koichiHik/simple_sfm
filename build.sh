#!/bin/bash

ROOT_3RD=~/workspace/3rdParty
CMAKE_BUILD_TYPE=Debug
OpenCV_DIR_RELEASE=${ROOT_3RD}/opencv24137/install/share/
OpenCV_DIR_DEBUG=${ROOT_3RD}/opencv24137/installd/share/

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
  ../

make

cd ../
