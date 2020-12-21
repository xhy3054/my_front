#include "vio_config.h"

#include <fstream>

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>



VioConfig::VioConfig() {
  // optical_flow_type = "patch";
  optical_flow_type = "frame_to_frame";
  optical_flow_detection_grid_size = 50;
  optical_flow_max_recovered_dist2 = 0.09f;
  optical_flow_pattern = 51;
  optical_flow_max_iterations = 5;
  optical_flow_levels = 3;
  optical_flow_epipolar_error = 0.005;
  optical_flow_skip_frames = 1;

  tracker_width = 640;
  tracker_height = 480;
  tracker_max_cnt = 1;
  tracker_flow_back = 0;
  tracker_equalize = 1;
  tracker_min_dist = 15;
  tracker_f_threshold = 1;

  input_device = 0;
  cam_id = 2;
  cam_fps = 120;
  video_name = "test.mp4";

}

void VioConfig::save(const std::string& filename) {
  std::ofstream os(filename);

  {
    cereal::JSONOutputArchive archive(os);
    archive(*this);
  }
  os.close();
}

void VioConfig::load(const std::string& filename) {
  std::ifstream is(filename);

  {
    cereal::JSONInputArchive archive(is);
    archive(*this);
  }
  is.close();
}
  // namespace basalt

namespace cereal {

template <class Archive>
void serialize(Archive& ar, VioConfig& config) {
  ar(CEREAL_NVP(config.optical_flow_type));
  ar(CEREAL_NVP(config.optical_flow_detection_grid_size));
  ar(CEREAL_NVP(config.optical_flow_max_recovered_dist2));
  ar(CEREAL_NVP(config.optical_flow_pattern));
  ar(CEREAL_NVP(config.optical_flow_max_iterations));
  ar(CEREAL_NVP(config.optical_flow_epipolar_error));
  ar(CEREAL_NVP(config.optical_flow_levels));
  ar(CEREAL_NVP(config.optical_flow_skip_frames));

  ar(CEREAL_NVP(config.tracker_width));
  ar(CEREAL_NVP(config.tracker_height));
  ar(CEREAL_NVP(config.tracker_max_cnt));
  ar(CEREAL_NVP(config.tracker_flow_back));
  ar(CEREAL_NVP(config.tracker_equalize));
  ar(CEREAL_NVP(config.tracker_min_dist));
  ar(CEREAL_NVP(config.tracker_f_threshold));

  ar(CEREAL_NVP(config.input_device));
  ar(CEREAL_NVP(config.cam_id));
  ar(CEREAL_NVP(config.cam_fps));
  ar(CEREAL_NVP(config.video_name));
}
}  // namespace cereal
