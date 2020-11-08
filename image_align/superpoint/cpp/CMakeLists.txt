cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ORB_SLAM2)

SET(CMAKE_BUILD_TYPE Debug)
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})



LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories("/usr/include/eigen3")

#SET(TORCH_PATH /home/sky/Documents/torch/1-0-1/pytorch/torch/lib/tmp_install/share/cmake/Torch)
SET(TORCH_PATH /home/sky/Documents/torch/1-2-0/pytorch/torch/share/cmake/Torch)
#SET(TORCH_PATH /home/sky/workspace/myrepo/my_front/image_align/spcpp/libtorch-cxx11-abi-shared-with-deps-1.2.0/libtorch/share/cmake/Torch)
#SET(TORCH_PATH /home/sky/Documents/torch/pytorch/torch/share/cmake/Torch)

if( TORCH_PATH ) 
   message(" set to: ${TORCH_PATH}")
   set(Torch_DIR ${TORCH_PATH})
else()
   message(FATAL_ERROR "Need to specify Torch path, e.g., pytorch/torch/share/cmake/Torch ")
endif()

find_package(Torch REQUIRED)
message(STATUS "Torch version is: ${Torch_VERSION}")
if(Torch_VERSION GREATER 1.0.1)
   message(STATUS "Torch version is newer than v1.0.1, will use new api")
   add_definitions(-DTORCH_NEW_API)
endif()

add_library( sp_lib
         SuperPoint.cc
)

target_link_libraries(sp_lib ${OpenCV_LIBS} ${TORCH_LIBRARIES})

add_executable(sp test.cpp SuperPoint.cc)
target_link_libraries(sp ${OpenCV_LIBS} sp_lib)

#add_executable(sp test.cpp SPextractor.cpp SuperPoint.cpp)
#target_link_libraries(sp ${OpenCV_LIBS} ${TORCH_LIBRARIES})

