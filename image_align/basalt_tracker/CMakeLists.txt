cmake_minimum_required(VERSION 3.2)

project(
basalt_tracker   
VERSION         20.08
LANGUAGES       CXX
DESCRIPTION     "C++ Implementation of basalt tracker."
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(USE_OPENMP_DEFAULT ON)
#set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release)
endif()


### eigen
if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "/home/sky/Documents/thirdparty/eigen")
endif()

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()




### OPENCV
find_package(OpenCV 3 QUIET)
message(STATUS "Found OpenCV version is ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include is ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV libraries is ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

### TBB
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
message(STATUS "TBB_INTERFACE_VERSION: ${TBB_INTERFACE_VERSION}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()


#include_directories(/home/sky/Documents/cereal/include)
include_directories(/home/sky/Documents/thirdparty/Sophus)
include_directories(/home/sky/Documents/thirdparty/cereal/include)

#add_library(io SHARED
#video_io.cpp
#vio_config.cpp
#)
#target_link_libraries(io PUBLIC
#${OpenCV_LIBS}
#)


add_library(tracker SHARED
video_io.cpp
vio_config.cpp
feature_tracker.cpp
)
target_link_libraries(tracker PUBLIC
${OpenCV_LIBS}
${TBB_LIBRARIES}
)

#add_executable(test main.cpp)
#target_link_libraries(test PUBLIC 
#io
#)

add_executable(test_tracker test_tracker.cpp)
target_link_libraries(test_tracker PUBLIC 
tracker
)

