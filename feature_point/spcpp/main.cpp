#include "SPDetector.hpp"
#include "Tools.hpp"

using namespace SuperPointSLAM;

/**  You need to modify the path below that corresponds to your dataset and weight path. **/
const std::string weight_dir = "../Weights/superpoint.pt";
void test();


int main(const int argc, char* argv[])
{
    /** From the main argument, Retrieve waiting period to control Displaying.**/
    int ms;
    if(argc == 2)
    {   
        char* a = argv[1];
        ms = std::atoi(a);
    }
    else ms = 1;
    std::cout << "Frame rate is " << ms << "ms.\n";

    VideoStreamer vs(2);
    vs.setImageSize(cv::Size(320, 200));
    
    
    /** Superpoint Detector **/
    SPDetector SPF(weight_dir, torch::cuda::is_available());
    std::cout << "VC created, SPDetector Constructed.\n";

    cv::namedWindow("superpoint", cv::WINDOW_NORMAL);

    long long idx=0;
    while(++idx){
        // Capture frame-by-frame
        // Image's size is [640 x 480]
        if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }

        std::vector<cv::KeyPoint> Keypoints;
        cv::Mat Descriptors;

        auto start = std::chrono::system_clock::now();
        SPF.detect(vs.input, Keypoints, Descriptors);
        auto end = std::chrono::system_clock::now();
        std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        /* Logging */
        std::cout << idx << "th ProcessTime: " << mill.count() << "ms\n";
        std::cout << "Keypoint num: " << Keypoints.size() << std::endl;

        float x_scale(vs.W_scale), y_scale(vs.H_scale);

        auto kpt_iter = Keypoints.begin();
        for(; kpt_iter != Keypoints.end(); kpt_iter++)
        {
            float X(kpt_iter->pt.x), Y(kpt_iter->pt.y);
            double conf(kpt_iter->response);
            cv::circle(vs.img, cv::Point(int(X*x_scale), int(Y*y_scale)), 3, cv::Scalar(0, 0, (255 * conf * 10)), 2);
        }

        // Display the resulting frame
        cv::imshow( "superpoint", vs.img );

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(ms);
        if(c==27){ break; }
    }


    // Closes all the frames
    cv::destroyAllWindows();
}