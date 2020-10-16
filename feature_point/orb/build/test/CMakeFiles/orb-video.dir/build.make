# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sky/workspace/myrepo/my_front/feature_point/orb

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sky/workspace/myrepo/my_front/feature_point/orb/build

# Include any dependencies generated for this target.
include test/CMakeFiles/orb-video.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/orb-video.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/orb-video.dir/flags.make

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o: test/CMakeFiles/orb-video.dir/flags.make
test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o: ../test/orb/orb-video.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sky/workspace/myrepo/my_front/feature_point/orb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb-video.dir/orb/orb-video.cpp.o -c /home/sky/workspace/myrepo/my_front/feature_point/orb/test/orb/orb-video.cpp

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb-video.dir/orb/orb-video.cpp.i"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sky/workspace/myrepo/my_front/feature_point/orb/test/orb/orb-video.cpp > CMakeFiles/orb-video.dir/orb/orb-video.cpp.i

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb-video.dir/orb/orb-video.cpp.s"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sky/workspace/myrepo/my_front/feature_point/orb/test/orb/orb-video.cpp -o CMakeFiles/orb-video.dir/orb/orb-video.cpp.s

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.requires:

.PHONY : test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.requires

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.provides: test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/orb-video.dir/build.make test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.provides.build
.PHONY : test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.provides

test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.provides.build: test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o


# Object files for target orb-video
orb__video_OBJECTS = \
"CMakeFiles/orb-video.dir/orb/orb-video.cpp.o"

# External object files for target orb-video
orb__video_EXTERNAL_OBJECTS =

../test/orb/orb-video: test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o
../test/orb/orb-video: test/CMakeFiles/orb-video.dir/build.make
../test/orb/orb-video: ../lib/libORBextractor.so
../test/orb/orb-video: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudastereo.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_stitching.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_superres.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_videostab.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudawarping.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_aruco.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_bgsegm.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_bioinspired.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_ccalib.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_dpm.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_face.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_photo.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudafilters.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_freetype.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_fuzzy.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_hdf.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_hfs.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_img_hash.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_optflow.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_reg.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_rgbd.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_saliency.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_stereo.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_structured_light.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_surface_matching.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_tracking.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_datasets.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_plot.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_text.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_dnn.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_ml.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_shape.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_video.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_ximgproc.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_calib3d.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_features2d.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_flann.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_highgui.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_videoio.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_objdetect.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_xphoto.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_imgproc.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_core.so.3.4.1
../test/orb/orb-video: /usr/local/lib/libopencv_cudev.so.3.4.1
../test/orb/orb-video: test/CMakeFiles/orb-video.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sky/workspace/myrepo/my_front/feature_point/orb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../test/orb/orb-video"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orb-video.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/orb-video.dir/build: ../test/orb/orb-video

.PHONY : test/CMakeFiles/orb-video.dir/build

test/CMakeFiles/orb-video.dir/requires: test/CMakeFiles/orb-video.dir/orb/orb-video.cpp.o.requires

.PHONY : test/CMakeFiles/orb-video.dir/requires

test/CMakeFiles/orb-video.dir/clean:
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test && $(CMAKE_COMMAND) -P CMakeFiles/orb-video.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/orb-video.dir/clean

test/CMakeFiles/orb-video.dir/depend:
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sky/workspace/myrepo/my_front/feature_point/orb /home/sky/workspace/myrepo/my_front/feature_point/orb/test /home/sky/workspace/myrepo/my_front/feature_point/orb/build /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test /home/sky/workspace/myrepo/my_front/feature_point/orb/build/test/CMakeFiles/orb-video.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/orb-video.dir/depend

