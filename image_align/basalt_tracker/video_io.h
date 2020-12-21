#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>
#include "vio_config.h"

class VideoStreamer{
private:
    // 枚举类型表示是相机还是视频
    enum class input_device{
        IS_CAMERA=0, 
        IS_VIDEO_FILE=1
    };
    input_device img_source;

    //--- INITIALIZE VIDEOCAPTURE  初始化opencv视频读取对象
    cv::VideoCapture cap;

    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera  设备相机id，默认设置为0
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API

    int MAX_FRAME_NUM = 1000000;    //最多读取的帧数
    int current_frame_num = 0;      //当前帧数，初始化为0
    cv::Size *pImgSize = NULL;      //opencv图像尺寸类型
    
public:
    // 默认打开相机0
    VideoStreamer(){
        cap.open(deviceID, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }
        img_source = input_device::IS_CAMERA;
    }

    ~VideoStreamer()
    {
        // When everything done, release the video capture object
        cap.release();
    }

    VideoStreamer(VioConfig& vio_config){
        if(!vio_config.input_device){
            img_source = input_device::IS_CAMERA;
        }
        else img_source = input_device::IS_VIDEO_FILE;

        if(img_source == input_device::IS_CAMERA){
            deviceID = vio_config.cam_id;
            cap.open(deviceID, apiID);
            cap.set(CV_CAP_PROP_FPS,vio_config.cam_fps);
            if(!cap.isOpened()){
                std::cerr << "ERROR! Unable to open camera\n";
                std::exit(1);
            }
        }
        else{
            cap.open(vio_config.video_name, apiID);
            if(!cap.isOpened()){
                std::cerr << "ERROR! Unable to open camera\n";
                std::exit(1);
            }            
        }
        cv::Mat test_grab;  //获取图像尺寸
        while(!cap.read(test_grab));
        pImgSize = new cv::Size(test_grab.size());
        std::cout<<"The initial size of cam is : "<<test_grab.size<<std::endl;        
    }


    float H_scale=1.0, W_scale=1.0; //scale为img相对与input的缩放倍数，因此在input上提取的角点坐标乘以此scale便为在image上的坐标
    cv::Mat img, input; //img为原图， input为缩放后的图
    cv::Mat read_image(const cv::String& path);
    // Read a image as grayscale and resize to img_size.

    bool next_frame();
    void setImageSize(const cv::Size &_size){ 

        pImgSize = new cv::Size(_size); 
        //std::cout<<"The final size of image is : "<<pImgSize<<std::endl;
    }
};

std::string cv_type2str(int type);

#endif