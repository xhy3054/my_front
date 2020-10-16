#include "GCNextractor.h"
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


	int nFeatures = feature_number;
	float fScaleFactor = 1.2;
	int nLevels = 8;
	int fIniThFAST = 20;
	int fMinThFAST = 7;
	GCNextractor* mpGCNextractor = new GCNextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);



	if (!cap.isOpened())
	{
		std::cout<<"Error opening video stream or file"<<std::endl;
		return -1;
	}
	int num_frames = 0;
	time_t start,end;
	time(&start);
	std::cout<<"the fps of camera is "<<cap.get(CV_CAP_PROP_FPS)<<std::endl;

	cv::namedWindow("Frame", cv::WINDOW_NORMAL);

	while(1){
		
		std::vector<cv::KeyPoint> kp1;
		cv::Mat ds1;

		num_frames++;
		cv::Mat frame;
		cap>>frame;
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

    	
    	(*mpGCNextractor)(frame,cv::Mat(),kp1,ds1);
    	for (auto kp:kp1)
    	{
    		cv::circle(frame, kp.pt, 2, cv::Scalar( 0, 0, 0 ), -1, cv::LINE_AA);
    		cv::circle(frame, kp.pt, 1, cv::Scalar( 255, 255, 255 ), -1, cv::LINE_AA);
    	}
    	imshow("Frame",frame);

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

