#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "SPextractor.h"


using namespace cv;
using namespace std;


const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind].x * ratio_width, pts_raw[select_ind].y * ratio_height, 1.0f));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
    
    descriptors.create(select_indice.size(), 32, CV_8U);

    for (int i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j<32; j++)
        {
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}



SPextractor::SPextractor()
{

    const char *net_fn = "../sp.pt";
    net_fn = (net_fn == nullptr) ? "sp.pt" : net_fn;
    module = torch::jit::load(net_fn);

}

void SPextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
{ 

    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    cv::Mat img;
    image.convertTo(img, CV_32FC1, 1.f / 255.f , 0);

    int img_width = 320;
    int img_height = 240;

    int border = 8;
    int dist_thresh = 4;
    
    float ratio_width = float(img.cols) / float(img_width);
    float ratio_height = float(img.rows) / float(img_height);
    
    cv::resize(img, img, cv::Size(img_width, img_height));

    #if defined(TORCH_NEW_API)
        std::vector<int64_t> dims = {1, img_height, img_width, 1};
        auto img_var = torch::from_blob(img.data, dims, torch::kFloat32).to(device);
        img_var = img_var.permute({0,3,1,2});
    #else 
        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img.data, {1, img_height, img_width, 1});
        img_tensor = img_tensor.permute({0,3,1,2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);
    #endif

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);
    auto output = module->forward(inputs).toTuple();

    auto pts  = output->elements()[0].toTensor().squeeze();
    auto desc = output->elements()[1].toTensor().squeeze();

    /**********************************************关键点生成************************************************************/
    auto kpts = (pts > 0.015);
    kpts = torch::nonzero(kpts);    

    std::vector<cv::KeyPoint> keypoints_no_nms;
    for(size_t i=0; i<kpts.size(); i++){
        float response = pts[kpts[i][0]][kpts[i][1]].item<float>();
        keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }

    /***********************************************描述子生成***********************************************************/
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);
    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
    }

    auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);    
    
    auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y    

    //nms(pts_mat, desc_mat, _keypoints, _descriptors, border, dist_thresh, img_width, img_height, ratio_width, ratio_height);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    nms(pts_mat, desc_mat, keypoints, descriptors, border, dist_thresh, img_width, img_height, ratio_width, ratio_height);
    //void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
      //  int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height)

    //_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    _keypoints=keypoints;
    
    int nkeypoints = keypoints.size();
    std::cout<<"the number of features is : "<<nkeypoints<<std::endl; 
    _descriptors.create(nkeypoints, 32, CV_8U);
    descriptors.copyTo(_descriptors.getMat());

    // std::cout << descriptors << std::endl;
    // std::cout << pts.size(0) << std::endl;
    // std::cout << keypoints.size() << std::endl;
    // cv::waitKey();

}

//namespace ORB_SLAM
