cmake_minimum_required(VERSION 2.8)
project(optical_flow)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

include_directories("/usr/include/eigen3")



find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(cv_of cv_of.cpp)
target_link_libraries(cv_of ${OpenCV_LIBS})