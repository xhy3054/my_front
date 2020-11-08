#include "optical_flow.h"

using namespace std;
using namespace cv;

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<cv::Point2f> &kp1,
        vector<cv::Point2f> &kp2,
        vector<uchar> &success,
        bool inverse
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 50;
    bool have_initial = !kp2.empty();
    // LK光流跟踪中每个特征点的跟踪之间是不相关的，因此此处逐个点进行光流跟踪
    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];

        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].x - kp.x;
            dy = kp2[i].y - kp.y;
        }

        double cost = 0, lastCost = 0;
        uchar succ = 1; // if this point succeeded

        // 使用高斯牛顿法迭代10次
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.x + dx <= half_patch_size || kp.x + dx >= img1.cols - half_patch_size ||
                kp.y + dy <= half_patch_size || kp.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = 0;
                break;
            }

            // compute cost and jacobian，LK窗口
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    //定义光度误差
                    double error = GetPixelValue(img1, kp.x + x, kp.y + y) -
                                   GetPixelValue(img2, kp.x + x + dx, kp.y + y + dy);
                    Eigen::Vector2d J;
                    if (inverse == false) {
                        //上述误差关于dx，dy的雅可比
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kp.x + dx + x + 1, kp.y + dy + y) -
                                       GetPixelValue(img2, kp.x + dx + x - 1, kp.y + dy + y)),
                                0.5 * (GetPixelValue(img2, kp.x + dx + x, kp.y + dy + y + 1) -
                                       GetPixelValue(img2, kp.x + dx + x, kp.y + dy + y - 1))
                        );
                    } else {
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img1, kp.x + x + 1, kp.y + y) -
                                       GetPixelValue(img1, kp.x + x - 1, kp.y + y)),
                                0.5 * (GetPixelValue(img1, kp.x + x, kp.y + y + 1) -
                                       GetPixelValue(img1, kp.x + x, kp.y + y - 1))
                        );
                    }

                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }

            Eigen::Vector2d update = H.ldlt().solve(b);
            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = 0;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = 1;
        }

        success.push_back(succ);
        if (have_initial) {
            kp2[i] = kp + Point2f(dx, dy);
        } else {
            cv::Point2f tracked = kp;
            tracked += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

//多层的没有不能设置初始值
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<cv::Point2f> &kp1,
        vector<cv::Point2f> &kp2,
        vector<uchar> &success,
        bool inverse) {

    std::cout<<"开始多层光流~"<<std::endl;
    // parameters
    int pyramids = 8;
    double pyramid_scale = 0.8;
    double scales[] = {1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // LK tracking in pyramids
    vector<cv::Point2f> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
    }

    if(!kp2.empty()){
        for (auto &kp:kp2) {
            auto kp_top = kp;
            kp_top *= scales[pyramids - 1];
            kp2_pyr.push_back(kp_top);
        }      
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse);

        if (level > 0) {
            // update kp1_pyr and kp2_pyr to next level
            for (auto &kp: kp1_pyr)
                kp /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp /= pyramid_scale;
        }
    }

    // set to kp2
    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);

    std::cout<<"跟踪到的点数为： "<<kp2.size()<<std::endl;
    std::cout<<"结束多层光流~"<<std::endl;

}
