cmake_minimum_required(VERSION 3.2)

project(
SPCPP   
VERSION         20.08
LANGUAGES       CXX
DESCRIPTION     "C++ Implementation of Superpoint."
)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

### LIBTORCH
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/sky/Documents/libtorch")
find_package(Torch REQUIRED QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
###

### OPENCV
find_package(OpenCV 3 QUIET)
message(STATUS "Found OpenCV version is ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include is ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV libraries is ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
###

add_library(Super SHARED
SuperPoint.cpp
SPDetector.cpp
#SPExtractor.cpp
Tools.cpp
)
target_compile_features(Super PUBLIC cxx_std_14)
target_link_libraries(Super PUBLIC
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)



### Runfile to test new Framework.
add_executable(Test main.cpp)
target_compile_features(Test PUBLIC cxx_std_14)
target_link_libraries(Test PUBLIC 
Super
# ${TORCH_LIBRARIES}
# ${OpenCV_LIBS}
)

