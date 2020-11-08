#ifndef SPDETECTOR_HPP
#define SPDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "SuperPoint.hpp"


namespace SuperPointSLAM
{


#define EPSILON 1e-19
//#define EPSILON 0

class SPDetector
{
public:

    // Default Constuctor. No Use.
    SPDetector();
    /**
     * @brief 构造函数，主要有三个工作
     * 1. 新建一个SuperPoint网络对象，并根据模型参数路径载入参数
     * 2. 将网络部署到gpu或者cpu设备上
     * 3. 将网络设置为验证模式
     * 
     * @param _weight_dir the PATH that contains pretrained weight.
     * @param _use_cuda whether the model operates in cpu or gpu.
     */
    SPDetector(std::string _weight_dir, bool _use_cuda, int width, int height);
    
    ~SPDetector(){}

    /**
     * @brief Detect input image's Keypoints and Compute Descriptor.
     * 
     * @param img Input image. We use img's deep copy object.
     * @return cv::Mat 
     */
    void detect(cv::InputArray _image, std::vector<cv::KeyPoint>& _keypoints,
                      cv::Mat &_descriptors);

    int n_keypoints;

private:
    std::shared_ptr<SuperPoint> model;  // Superpoint model object  sp网络模型对象              
    c10::TensorOptions tensor_opts;     // Contains necessary info for creating proper at::Tensor  
    c10::DeviceType mDeviceType;        // If our device can use the GPU, it has 'kCUDA', otherwise it has 'kCPU'.
    c10::Device mDevice;                // c10::Device corresponding to mDeviceType.

   
    torch::Tensor mProb; // Superpoint Output Probability Tensor  sp网络输出的概率张量            
    torch::Tensor mDesc; // Superpoint Output Descriptor Tensor   sp网络输出的描述子张量
    
    /**
     * kpts is [H, W] size Tensor. 
     * Its elements has 1 if there is a featrue, and 0 otherwise. 
     * After this function, kpts Tensor can guarantee that 
     * no overlapping feature points exist in each 3x3 patch.
     * 
     * This function is executed on the CPU because it requires direct access 
     * to the tensor elements. Therefore, in some cases, (Specifically, if 
     * detect() is done on the GPU,) performance can be greatly impaired  
     * because of the data travel time You can turn this option on and off 
     * using the 'nms' member variable.
     */
    void SemiNMS(at::Tensor& kpts);

    int width,height;

    // SemiNMS() on/off flag.
    bool nms = true; 

    // 1/64 = 0.015625 
    float mConfThres=0.015;
};

}


#endif