#include "video_io.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include "vio_config.h"


VioConfig vio_config;

int main(const int argc, char* argv[])
{
  std::string config_path;	
  if(argc == 2){
  	config_path = argv[1];
  }
  else 
  	std::cout<<"the config_path was not given!"<<std::endl;

  if (!config_path.empty()) {
    vio_config.load(config_path);
  }  

  VideoStreamer vs(vio_config);

  cv::namedWindow("basalt-tracker", cv::WINDOW_NORMAL);

  while(1){
    if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }

    auto out=vs.img.clone();
    cv::imshow( "basalt-tracker", out );
        // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(1);
    if(c==27){ break; }    
  }
  cv::destroyAllWindows();
  return 0;

}