#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <time.h>

#include "tic_toc.h"
#include "vio_config.h"
#include "basaltOf/basaltOf.h"

using namespace std;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对每个相机进行角点LK光流跟踪
*/
class FeatureTracker
{
  public:
    FeatureTracker(const VioConfig& vio_config) :
            COL(vio_config.tracker_width), 
            ROW(vio_config.tracker_height), 
            MAX_CNT(vio_config.tracker_max_cnt), 
            FLOW_BACK(vio_config.tracker_flow_back), 
            EQUALIZE(vio_config.tracker_equalize), 
            MIN_DIST(vio_config.tracker_min_dist), 
            F_THRESHOLD(vio_config.tracker_f_threshold),
            config(vio_config) {}

    bool readImage(const cv::Mat &_img);

    void setMask();

    void addPoints();

    bool updateID();

    void rejectWithF();

    bool inBorder(const cv::Point2f &pt);

    double distance(cv::Point2f &pt1, cv::Point2f &pt2);

    int FLOW_BACK;
    int MAX_CNT;
    int MIN_DIST;
    int ROW, COL;
    double F_THRESHOLD;
    int EQUALIZE;
    cv::Mat mask;//图像掩码

    // prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据
    cv::Mat prev_img, cur_img, forw_img;

    vector<cv::Point2f> n_pts;//每一帧中新提取的特征点

    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    
    vector<int> ids;//能够被跟踪到的特征点的id

    vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数

    VioConfig config;




    static int n_id;//特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};