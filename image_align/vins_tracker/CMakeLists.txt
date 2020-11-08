cmake_minimum_required( VERSION 2.8 )
project( vins_tracker )

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -03")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()



################# dependencies ####################
# opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)
################ SOURCE AND TEST ###################
add_executable(vins_tracker
    main.cpp
    feature_tracker.cpp
    optical_flow.cpp
    )

target_link_libraries(vins_tracker ${OpenCV_LIBS})

add_executable(video
    video.cpp
    feature_tracker.cpp
    optical_flow.cpp
    )

target_link_libraries(video ${OpenCV_LIBS})

add_executable(test_of
    test_of.cpp
    optical_flow.cpp
    )

target_link_libraries(test_of ${OpenCV_LIBS})