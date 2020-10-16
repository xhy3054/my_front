#include <iostream>
#include <opencv2/opencv.hpp>
#include<time.h>

using namespace std;
using namespace cv;
int main(int argc, char** argv){
    CommandLineParser parser(argc, argv, "{@input | ../../data/1.png | input image}");
    Mat src = imread(parser.get<String>("@input"));
    if(src.empty()){
        std::cout<<"could not open the image!\n"<<std::endl;
        std::cout<<"Usage: "<<argv[0]<<" <Input image>"<<std::endl;
        return 1;
    }
    
    ORB_SLAM2::ORBextractor detect1(10, 1.2, 8, 30,100);
    std::vector<KeyPoint> kp1,kp2;
    Mat ds1,ds2;
    detect1(src, Mat(), kp1, ds1);
    cout<<"the size of keypoints is: "<<kp1.size()<<endl;
    for(int i=0;i<kp1.size();++i)
        cout<<"第"<< i+1 <<"个关键点的属性如下："<< endl << "坐标为：" << kp1[i].pt << endl << "size  = " << kp1[i].size << endl << "angle = " << kp1[i].angle << endl << "response =  "<< kp1[i].response << endl << "提取自哪一层金字塔　octave = "<< kp1[i].octave << endl << "class_id = " << kp1[i].class_id << endl ;
    cout<< "第一幅图像上的描述子Mat ds1的属性如下：" << endl << "rows = " << ds1.rows << endl << "cols =" << ds1.cols << endl << "dims = " << ds1.dims << endl << "size = " << ds1.size << endl << "type() = " << ds1.type() << endl ;
    
    Mat img_keypoints;
    drawKeypoints(src, kp1, img_keypoints);
    imshow("orbextractor", img_keypoints);

    Ptr<ORB> orb = ORB::create(10);
    orb->setFastThreshold(0);
    cout<<"此处提取orb特征点500个"<<endl;
    

    orb->detectAndCompute(src, Mat(), kp2, ds2);
    for(int i=0;i<kp2.size();++i)
        cout<<"第"<< i <<"个关键点的属性如下："<< endl << "坐标为：" << kp2[i].pt << endl << "size  = " << kp2[i].size << endl << "angle = " << kp2[i].angle << endl << "response =  "<< kp2[i].response << endl << "提取自哪一层金字塔　octave = "<< kp2[i].octave << endl << "class_id = " << kp2[i].class_id << endl ;
    cout<< "第一幅图像上的描述子Mat ds2的属性如下：" << endl << "rows = " << ds2.rows << endl << "cols =" << ds2.cols << endl << "dims = " << ds2.dims << endl << "size = " << ds2.size << endl << "type() = " << ds2.type() << endl ;
    
    waitKey();
    return 0;
}
