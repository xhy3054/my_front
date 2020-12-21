#include "video_io.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include "vio_config.h"
#include <time.h>
#include "feature_tracker.h"

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
  vs.setImageSize(cv::Size(vio_config.tracker_width, vio_config.tracker_height));

  FeatureTracker trackerData(vio_config);

  cv::namedWindow("basalt-tracker", cv::WINDOW_NORMAL);

  while(1){
    if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }

    auto out=vs.img.clone();
    //cv::imshow("basalt-tracker",out);

    bool sign=trackerData.readImage(vs.input);
    //cv::imshow( "basalt-tracker", out );
    //if(!sign) continue;
    float x_scale(vs.W_scale), y_scale(vs.H_scale);
    for (int i=0; i<trackerData.forw_pts.size(); ++i)
    {
        if (trackerData.ids[i]!=-1)
        {
          /* code */
          cv::Point2f p_prev = cv::Point(int(trackerData.prev_pts[i].x*x_scale), int(trackerData.prev_pts[i].y*y_scale));
          cv::Point2f p_cur = cv::Point(int(trackerData.cur_pts[i].x*x_scale), int(trackerData.cur_pts[i].y*y_scale));
          //cv::circle(out, trackerData.prev_pts[i], 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
          //cv::circle(out, trackerData.cur_pts[i], 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
          //cv::line(out,  trackerData.prev_pts[i], trackerData.cur_pts[i], cv::Scalar( 0, 0, 255 ));          
          cv::circle(out, p_prev, 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
          cv::circle(out, p_cur, 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
          cv::line(out,  p_prev, p_cur, cv::Scalar( 0, 0, 255 ));
        }
        
    }
    trackerData.updateID();
    cv::imshow("basalt-tracker",out);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(1);
    if(c==27){ break; }    
  }


  cv::destroyAllWindows();
  return 0;

}