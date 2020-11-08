

#ifndef SP_H
#define SP_H

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// Compile will fail for opimizier since pytorch defined this
#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

#include <vector>
#include <list>
#include <opencv/cv.h>




class SPextractor
{
public:

    SPextractor();

    ~SPextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);


protected:


    std::shared_ptr<torch::jit::script::Module> module;
};

//namespace ORB_SLAM

#endif

