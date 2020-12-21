#pragma once

#include <string>



struct VioConfig {
  VioConfig();
  void load(const std::string& filename);
  void save(const std::string& filename);

  std::string optical_flow_type;
  int optical_flow_detection_grid_size;
  float optical_flow_max_recovered_dist2;
  int optical_flow_pattern;
  int optical_flow_max_iterations;
  int optical_flow_levels;
  float optical_flow_epipolar_error;
  int optical_flow_skip_frames;

  int tracker_width;
  int tracker_height;
  int tracker_max_cnt;
  int tracker_flow_back;
  int tracker_equalize;
  int tracker_min_dist;
  double tracker_f_threshold;

  int input_device;
  int cam_id;
  int cam_fps;
  std::string video_name;
};
 // namespace basalt
