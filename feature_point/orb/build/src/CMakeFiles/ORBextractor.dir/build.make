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
include src/CMakeFiles/ORBextractor.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/ORBextractor.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/ORBextractor.dir/flags.make

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o: src/CMakeFiles/ORBextractor.dir/flags.make
src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o: ../src/ORBextractor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sky/workspace/myrepo/my_front/feature_point/orb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o -c /home/sky/workspace/myrepo/my_front/feature_point/orb/src/ORBextractor.cpp

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ORBextractor.dir/ORBextractor.cpp.i"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sky/workspace/myrepo/my_front/feature_point/orb/src/ORBextractor.cpp > CMakeFiles/ORBextractor.dir/ORBextractor.cpp.i

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ORBextractor.dir/ORBextractor.cpp.s"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sky/workspace/myrepo/my_front/feature_point/orb/src/ORBextractor.cpp -o CMakeFiles/ORBextractor.dir/ORBextractor.cpp.s

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.requires:

.PHONY : src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.requires

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.provides: src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/ORBextractor.dir/build.make src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.provides.build
.PHONY : src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.provides

src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.provides.build: src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o


# Object files for target ORBextractor
ORBextractor_OBJECTS = \
"CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o"

# External object files for target ORBextractor
ORBextractor_EXTERNAL_OBJECTS =

../lib/libORBextractor.so: src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o
../lib/libORBextractor.so: src/CMakeFiles/ORBextractor.dir/build.make
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudastereo.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_stitching.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_superres.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_videostab.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_aruco.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_bgsegm.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_bioinspired.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_ccalib.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_dpm.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_face.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_freetype.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_fuzzy.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_hdf.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_hfs.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_img_hash.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_optflow.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_reg.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_rgbd.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_saliency.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_stereo.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_structured_light.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_surface_matching.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_tracking.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_ximgproc.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_xphoto.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_shape.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudawarping.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_photo.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudafilters.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_datasets.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_plot.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_text.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_dnn.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_ml.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_video.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_calib3d.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_features2d.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_highgui.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_videoio.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_flann.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_objdetect.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_imgproc.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_core.so.3.4.1
../lib/libORBextractor.so: /usr/local/lib/libopencv_cudev.so.3.4.1
../lib/libORBextractor.so: src/CMakeFiles/ORBextractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sky/workspace/myrepo/my_front/feature_point/orb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../lib/libORBextractor.so"
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ORBextractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/ORBextractor.dir/build: ../lib/libORBextractor.so

.PHONY : src/CMakeFiles/ORBextractor.dir/build

src/CMakeFiles/ORBextractor.dir/requires: src/CMakeFiles/ORBextractor.dir/ORBextractor.cpp.o.requires

.PHONY : src/CMakeFiles/ORBextractor.dir/requires

src/CMakeFiles/ORBextractor.dir/clean:
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src && $(CMAKE_COMMAND) -P CMakeFiles/ORBextractor.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/ORBextractor.dir/clean

src/CMakeFiles/ORBextractor.dir/depend:
	cd /home/sky/workspace/myrepo/my_front/feature_point/orb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sky/workspace/myrepo/my_front/feature_point/orb /home/sky/workspace/myrepo/my_front/feature_point/orb/src /home/sky/workspace/myrepo/my_front/feature_point/orb/build /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src /home/sky/workspace/myrepo/my_front/feature_point/orb/build/src/CMakeFiles/ORBextractor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/ORBextractor.dir/depend

