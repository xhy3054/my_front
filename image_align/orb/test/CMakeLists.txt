
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/orb)
add_executable(orb orb/orb.cpp)
target_link_libraries(orb ORBextractor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/orb)
add_executable(orb-slam orb/orb-slam.cpp)
target_link_libraries(orb-slam ORBextractor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/orb)
add_executable(video orb/video.cpp)
target_link_libraries(video ORBextractor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/harris)
add_executable(test_harris harris/test_harris.cpp)
target_link_libraries(test_harris myharris)


