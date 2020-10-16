#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "feature_tracker.h"

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

	FeatureTracker trackerData(width,height,feature_number);

	if (!cap.isOpened())
	{
		std::cout<<"Error opening video stream or file"<<std::endl;
		return -1;
	}

	int num_frames = 0;
	time_t start,end;
	time(&start);
	std::cout<<"the fps of camera is "<<cap.get(CV_CAP_PROP_FPS)<<std::endl;
	//cv::namedWindow('vins_tracker', cv::WINDOW_NORMAL);
	cv::namedWindow("vins_tracker", cv::WINDOW_NORMAL);
    //cv::resizeWindow('vins_tracker', (640, 480));
    bool sign0=true;
    int lost_total=0;

	while(1){
		num_frames++;
		cv::Mat frame,out;
		cap>>frame;
		cv::resize(frame, frame, cv::Size(160,100), 0, 0, cv::INTER_LINEAR);
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

    	bool sign=trackerData.readImage(frame);
		if(!sign && !sign0)	{
			std::cout<<"the total lost is : "<<lost_total<<std::endl;
			time(&end);
			double seconds = difftime(end,start);
			std::cout<<"Time taken : "<<seconds<<" seconds" <<std::endl;
			double fps1= num_frames/seconds;
			std::cout<<"the fps of the task is : "<<fps1<<std::endl;			
			return -1;		

		}
		if (!sign)
		{
			lost_total++;
		}
		sign0=sign;

		
		//return -1;
    	//todo

    	for (int i=0; i<trackerData.forw_pts.size(); ++i)
    	{
    		if (trackerData.ids[i]!=-1)
    		{
    			/* code */
    			cv::circle(out, trackerData.prev_pts[i], 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
    			cv::circle(out, trackerData.cur_pts[i], 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
    			cv::line(out,  trackerData.prev_pts[i], trackerData.cur_pts[i], cv::Scalar( 0, 0, 255 ));
    		}
    		
    	}
    	trackerData.updateID();
    	imshow("vins_tracker",out);

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