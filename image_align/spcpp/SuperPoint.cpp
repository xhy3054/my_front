#include "SuperPoint.hpp"

namespace SuperPointSLAM
{

SuperPoint::SuperPoint()
{
    /* 
        A Module is registered as a submodule to another Module 
        by calling register_module(), typically from within a parent 
        module’s constructor.
    */

    //SHARED ENCODER 
    conv1a = register_module("conv1a", Conv2d(Conv2dOptions(1, c1, 3).stride(1).padding(1)));   //input:1, output:c1, kernel size:3 
    conv1b = register_module("conv1b", Conv2d(Conv2dOptions(c1, c1, 3).stride(1).padding(1)));

    conv2a = register_module("conv2a", Conv2d(Conv2dOptions(c1, c2, 3).stride(1).padding(1)));
    conv2b = register_module("conv2b", Conv2d(Conv2dOptions(c2, c2, 3).stride(1).padding(1)));

    conv3a = register_module("conv3a", Conv2d(Conv2dOptions(c2, c3, 3).stride(1).padding(1)));
    conv3b = register_module("conv3b", Conv2d(Conv2dOptions(c3, c3, 3).stride(1).padding(1)));

    conv4a = register_module("conv4a", Conv2d(Conv2dOptions(c3, c4, 3).stride(1).padding(1)));
    conv4b = register_module("conv4b", Conv2d(Conv2dOptions(c4, c4, 3).stride(1).padding(1)));

    //DETECTOR
    convPa = register_module("convPa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convPb = register_module("convPb", Conv2d(Conv2dOptions(c5, 65, 1).stride(1).padding(0)));

    //DESCRIPTOR
    convDa = register_module("convDa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convDb = register_module("convDb", Conv2d(Conv2dOptions(c5, d1, 1).stride(1).padding(0)));
}

void SuperPoint::forward(torch::Tensor& x, torch::Tensor& Prob, torch::Tensor& Desc)
{
    //SHARED ENCODER 注：。。。此处可以尝试不同模块里的relu函数
    x = at::relu(conv1a->forward(x));
    x = at::relu(conv1b->forward(x));
    x = at::max_pool2d(x, 2, 2);

    x = at::relu(conv2a->forward(x));
    x = at::relu(conv2b->forward(x));
    x = at::max_pool2d(x, 2, 2);

    x = at::relu(conv3a->forward(x));
    x = at::relu(conv3b->forward(x));
    x = at::max_pool2d(x, 2, 2);

    x = at::relu(conv4a->forward(x));
    x = at::relu(conv4b->forward(x));

    //DETECTOR
    auto cPa = at::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa); // [B, 65, H/8, W/8]

    //DESCRIPTOR
    auto cDa = at::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa); // [B, 256, H/8, W/8]
    auto dn = at::norm(desc, 2, 1);
    // 此处写法需要调试
    desc = at::div((desc + EPSILON), unsqueeze(dn, 1));
    //desc = desc.div(unsqueeze(dn, 1));

    //DETECTOR - POST PROCESS 归一化，去除垃圾通道，恢复全分辨率响应图
    semi = softmax(semi, 1);            // 在tensor的第一维度（65维度）上进行softmax归一化操作
    semi = semi.slice(1, 0, 64);        // remove rest_bin 将第一维度上最后一维给删了
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);  //
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8}); //[B, H/8, W/8, 8, 8]
    semi = semi.permute({0, 1, 3, 2, 4});   // [B, H/8, 8, W/8, 8]
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]

    //Return Tensor 返回全分辨率响应图与低分辨率描述子
    Prob = semi;    // [B, H, W]
    Desc = desc;    // [B, 256, H/8, W/8]
}

} // Namespace NAMU_TEST END
