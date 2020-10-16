#include "ORBextractor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include<time.h>

int main(int argc, char const *argv[])
{
	if(argc != 6){
		std::cout<<"请按照如下格式输入参数："<<std::endl;
		std::cout<<"./exe cam_id img_width img_height fps feature_number"<<std::endl;
        return -1;
	}
	
	int cam_id = std::stoi(argv[1]);
	int width = std::stoi(argv[2]);
	int height = std::stoi(argv[3]);
	int fps = std::stoi(argv[4]);
	int feature_number = std::stoi(argv[5]);
	cv::VideoCapture cap(cam_id);

	cap.set(CV_CAP_PROP_FRAME_WIDTH,width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,height);
	cap.set(CV_CAP_PROP_FPS,fps);

	ORB_SLAM2::ORBextractor detect1(feature_number, 1.2, 8, 20,7);
	cv::BFMatcher bfmatcher(cv::NORM_HAMMING);

    std::vector<cv::KeyPoint> kp1,kp2;
    cv::Mat ds1,ds2;
    std::vector<std::vector<cv::DMatch>> matches_knn;

	if (!cap.isOpened())
	{
		std::cout<<"Error opening video stream or file"<<std::endl;
		return -1;
	}

	int num_frames = 0;
	time_t start,end;
	time(&start);
	std::cout<<"the fps of camera is "<<cap.get(CV_CAP_PROP_FPS)<<std::endl;

	cv::namedWindow("orb-slam", cv::WINDOW_NORMAL);

	while(1){
		num_frames++;
		cv::Mat frame,out;
		cap>>frame;
		out = frame.clone();
		if (frame.empty())
		{
			break;
		}
		//imshow("Frame",frame);

        // Step 1 ：将彩色图像转为灰度图像
        //若图片是3、4通道的，还需要转化成灰度图
        if(frame.channels()==3)
        {
            cv::cvtColor(frame,frame,CV_RGB2GRAY);
    	}
    	else if(frame.channels()==4)
    	{
            cv::cvtColor(frame,frame,CV_RGBA2GRAY);
    	}		

    	detect1(frame, cv::Mat(), kp1, ds1);
    	if (kp2.size()!=0)
    	{
    		/* code */
    		bfmatcher.knnMatch(ds2, ds1, matches_knn, 2);
    		for (int i = 0; i < matches_knn.size(); ++i)
    		{
    			/* code */
    			if (matches_knn[i][0].distance / matches_knn[i][1].distance <0.1)
    			{
    				cv::Point2f p2 = kp2[matches_knn[i][0].queryIdx].pt;
    				cv::Point2f p1 = kp1[matches_knn[i][0].trainIdx].pt;
    				cv::circle(out, p2, 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
    				cv::circle(out, p1, 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
    				cv::line(out,  p2, p1, cv::Scalar( 0, 0, 255 ));
    			}

    		}
    	}
    	imshow("orb-slam",out);

    	kp2=kp1;
    	ds2=ds1.clone();
		/*add code here*/
		char c=(char)cv::waitKey(1);
		if (c==27)
		{
			break;
		}
	}
	time(&end);
	double seconds = difftime(end,start);
	std::cout<<"Time taken : "<<seconds<<" seconds" <<std::endl;
	double fps1= num_frames/seconds;
	std::cout<<"the fps of the task is : "<<fps1<<std::endl;


	cap.release();
	cv::destroyAllWindows();
	return 0;
}