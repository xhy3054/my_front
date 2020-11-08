#include "SPDetector.hpp"
#include "Tools.hpp"

using namespace SuperPointSLAM;

/**  You need to modify the path below that corresponds to your dataset and weight path. **/
const std::string weight_dir = "../superpoint.pt";



int main(const int argc, char* argv[])
{
    if(argc != 4){
        std::cout<<"请按照如下格式输入参数："<<std::endl;
        std::cout<<"./exe cam_id img_width img_height"<<std::endl;
        return -1;
    }

    int cam_id = std::stoi(argv[1]);
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);    

    VideoStreamer vs(cam_id);
    //vs.setImageSize(cv::Size(320, 240));
    //vs.setImageSize(cv::Size(160, 120));
    
    
    /** Superpoint Detector **/
    //SPDetector SPF(weight_dir, torch::cuda::is_available());
    SPDetector SPF("../superpoint.pt", torch::cuda::is_available(), width, height);
    std::cout << "VC created, SPDetector Constructed.\n";

    cv::Ptr<cv::DescriptorMatcher> bfmatcher = cv::DescriptorMatcher::create("BruteForce");

    std::vector<cv::KeyPoint> kp1,kp2;
    cv::Mat ds1,ds2,last_img;    
    std::vector<cv::DMatch> matches1, matches2;
    std::vector<cv::Point2f> p2v;
    std::vector<cv::Point2f> p1v;
    std::vector<size_t> indv;

    cv::namedWindow("sp-tracker", cv::WINDOW_NORMAL);

    long long idx=0;
    while(++idx){
        int mnum=0;
        // Capture frame-by-frame
        // Image's size is [640 x 480]
        if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }

        auto out=vs.img.clone();

        auto start = std::chrono::system_clock::now();
        SPF.detect(vs.input, kp1, ds1);
        auto end = std::chrono::system_clock::now();
        std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        /* Logging */
        std::cout << idx << "th DetectTime: " << mill.count() << "ms\n";
        std::cout << "Keypoint num: " << kp1.size() << std::endl;


        if (kp2.size()!=0)
        {
            
            // 双向匹配交叉验证
            auto start1 = std::chrono::system_clock::now();
            bfmatcher->match(ds2, ds1, matches1, cv::Mat());
            bfmatcher->match(ds1, ds2, matches2, cv::Mat());
            auto end1 = std::chrono::system_clock::now();
            std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << idx << "th MatchTime: " << mill.count() << "ms\n";


            //float x_scale(vs.W_scale), y_scale(vs.H_scale);
            for (size_t i = 0; i < matches1.size(); ++i)//matches_knn.size()
            {
                for(size_t j = 0; j< matches2.size(); ++j){
                    if (matches1[i].queryIdx==matches2[j].trainIdx && matches2[j].queryIdx==matches1[i].trainIdx)
                    {
                        if(matches1[i].distance < 0.7)
                        {
                            indv.push_back(matches1[i].trainIdx);
                            cv::Point2f p2 = kp2[matches1[i].queryIdx].pt;
                            cv::Point2f p1 = kp1[matches1[i].trainIdx].pt;
                            p2v.push_back(p2);
                            p1v.push_back(p1);
                        }                     
                    }
                }
            }
            std::vector<unsigned char> status;
            std::vector<float> error;

            cv::calcOpticalFlowPyrLK(last_img, out, p2v, p1v, status, error, cv::Size(21,21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

            for(size_t i=0; i<status.size(); i++){
                if(status[i]){
                    kp1[indv[i]].pt = p1v[i];
                    mnum++;
                    cv::circle(out, p2v[i], 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
                    cv::circle(out, p1v[i], 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
                    cv::line(out,  p2v[i], p1v[i], cv::Scalar( 0, 0, 255 ));  
                }
            }
            
            /*
            // 单纯匹配
            auto start1 = std::chrono::system_clock::now();
            bfmatcher->match(ds2, ds1, matches1, cv::Mat());
            auto end1 = std::chrono::system_clock::now();
            std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            std::cout << idx << "th MatchTime: " << mill.count() << "ms\n";


            float x_scale(vs.W_scale), y_scale(vs.H_scale);
            for (size_t i = 0; i < matches1.size(); ++i)//matches_knn.size()
            {   
                cv::Point2f p2 = kp2[matches1[i].queryIdx].pt;
                cv::Point2f p1 = kp1[matches1[i].trainIdx].pt;
                cv::Point2f p_2 = cv::Point(int(p2.x*x_scale), int(p2.y*y_scale));
                cv::Point2f p_1 = cv::Point(int(p1.x*x_scale), int(p1.y*y_scale));

                cv::circle(out, p_2, 2, cv::Scalar( 255, 255, 0 ), -1, cv::LINE_AA);
                cv::circle(out, p_1, 2, cv::Scalar( 0, 0, 255 ), -1, cv::LINE_AA);
                cv::line(out,  p_2, p_1, cv::Scalar( 0, 0, 255 ));                                    
            } 
            */    
            std::cout<<"当前帧匹配的数量为： "<<mnum<<std::endl;
                  
        }

/*

        float x_scale(vs.W_scale), y_scale(vs.H_scale);

        auto kpt_iter = Keypoints.begin();
        for(; kpt_iter != Keypoints.end(); kpt_iter++)
        {
            float X(kpt_iter->pt.x), Y(kpt_iter->pt.y);
            double conf(kpt_iter->response);
            cv::circle(vs.img, cv::Point(int(X*x_scale), int(Y*y_scale)), 3, cv::Scalar(0, 0, (255 * conf * 10)), 2);
        }
*/
        kp2=kp1;
        ds2=ds1.clone();
        last_img = vs.img.clone();
        p2v.clear();
        p1v.clear();
        indv.clear();

        if(mnum==0)  continue; 
        // Display the resulting frame
        cv::imshow( "sp-tracker", out );
        


        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(1);
        if(c==27){ break; }
    }


    // Closes all the frames
    cv::destroyAllWindows();
}