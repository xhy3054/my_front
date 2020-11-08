#ifndef OV_CORE_TRACK_DESC_GPU_H
#define OV_CORE_TRACK_DESC_GPU_H


#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "TrackBase.h"


/**
 * @brief Descriptor-based visual tracking
 *
 * Here we use descriptor matching to track features from one frame to the next.
 * We track both temporally, and across stereo pairs to get stereo contraints.
 * Right now we use ORB descriptors as we have found it is the fastest when computing descriptors.
 */
class TrackDescriptor_GPU : public TrackBase
{

public:

    /**
     * @brief Public default constructor
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     */
    TrackDescriptor_GPU(std::map<size_t,Eigen::Matrix3d> camera_k,
                        std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
                        std::map<size_t,bool> camera_fisheye):
            TrackBase(camera_k,camera_d,camera_fisheye),threshold(10),grid_x(8),grid_y(5),knn_ratio(0.75) {
        // Extract our features and their descriptors
        // NOTE: seems that you need to set parameters using the contructor
        // NOTE: the set functions do not seem to work with this cuda version
        detector = cv::cuda::ORB::create(num_features);
        //detector->setMaxFeatures(num_features);
        //detector->setFastThreshold(threshold);
    }

    /**
     * @brief Public constructor with configuration variables
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     * @param fast_threshold FAST detection threshold
     * @param gridx size of grid in the x-direction / u-direction
     * @param gridy size of grid in the y-direction / v-direction
     * @param knnratio matching ratio needed (smaller value forces top two descriptors during match to be more different)
     */
    explicit TrackDescriptor_GPU(std::map<size_t,Eigen::Matrix3d> camera_k,
                                std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
                                std::map<size_t,bool> camera_fisheye,
                                int numfeats, int numaruco, int fast_threshold, int gridx, int gridy, double knnratio):
            TrackBase(camera_k,camera_d,camera_fisheye,numfeats,numaruco),threshold(fast_threshold),grid_x(gridx),grid_y(gridy),knn_ratio(knnratio) {
        // Extract our features and their descriptors
        // NOTE: seems that you need to set parameters using the contructor
        // NOTE: the set functions do not seem to work with this cuda version
        detector = cv::cuda::ORB::create(num_features);
        //detector->setMaxFeatures(num_features);
        //detector->setFastThreshold(threshold);
    }

    /**
     * @brief Process a new monocular image
     * @param timestamp timestamp the new image occurred at
     * @param img new cv:Mat grayscale image
     * @param cam_id the camera id that this new image corresponds too
     */
    void feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) override;

    /**
     * @brief Process new stereo pair of images
     * @param timestamp timestamp this pair occured at (stereo is synchronised)
     * @param img_left first grayscaled image
     * @param img_right second grayscaled image
     * @param cam_id_left first image camera id
     * @param cam_id_right second image camera id
     */
    void feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) override;


protected:

    /**
     * @brief Detects new features in the current image
     * @param img0 image we will detect features on
     * @param pts0 vector of extracted keypoints
     * @param desc0 vector of the extracted descriptors
     * @param ids0 vector of all new IDs
     *
     * Given a set of images, and their currently extracted features, this will try to add new features.
     * We return all extracted descriptors here since we DO NOT need to do stereo tracking left to right.
     * Our vector of IDs will be later overwritten when we match features temporally to the previous frame's features.
     * See robust_match() for the matching.
     */
    void perform_detection_monocular(const cv::cuda::GpuMat &img0, std::vector<cv::KeyPoint>& pts0, cv::cuda::GpuMat &d_desc0, std::vector<size_t>& ids0);

    /**
     * @brief Detects new features in the current stereo pair
     * @param img0 left image we will detect features on
     * @param img1 right image we will detect features on
     * @param pts0 left vector of new keypoints
     * @param pts1 right vector of new keypoints
     * @param desc0 left vector of extracted descriptors
     * @param desc1 left vector of extracted descriptors
     * @param ids0 left vector of all new IDs
     * @param ids1 right vector of all new IDs
     *
     * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
     * We also do STEREO matching from the left to right, and only return good matches that are found in both the left and right.
     * Our vector of IDs will be later overwritten when we match features temporally to the previous frame's features.
     * See robust_match() for the matching.
     */
    //void perform_detection_stereo(const cv::Mat& img0, const cv::Mat& img1, std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1,
    //                              cv::Mat& desc0, cv::Mat& desc1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

    /**
     * @brief Find matches between two keypoint+descriptor sets.
     * @param pts0 first vector of keypoints
     * @param pts1 second vector of keypoints
     * @param desc0 first vector of descriptors
     * @param desc1 second vector of decriptors
     * @param matches vector of matches that we have found
     *
     * This will perform a "robust match" between the two sets of points (slow but has great results).
     * First we do a simple KNN match from 1to2 and 2to1, which is followed by a ratio check and symmetry check.
     * Original code is from the "RobustMatcher" in the opencv examples, and seems to give very good results in the matches.
     * https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
     */
    void robust_match(std::vector<cv::KeyPoint>& pts0, std::vector<cv::KeyPoint> pts1,
                      cv::cuda::GpuMat& d_desc0, cv::cuda::GpuMat& d_desc1, std::vector<cv::DMatch>& matches);

    // Helper functions for the robust_match function
    // Original code is from the "RobustMatcher" in the opencv examples
    // https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
    void robust_ratio_test(std::vector<std::vector<cv::DMatch> >& matches);
    void robust_symmetry_test(std::vector<std::vector<cv::DMatch> >& matches1, std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& good_matches);

    // Timing variables
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;

    // ORB extractor used to get our features and descriptors
    cv::Ptr<cv::cuda::ORB> detector;

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // The ratio between two kNN matches, if that ratio is larger then this threshold
    // then the two features are too close, so should be considered ambiguous/bad match
    double knn_ratio;

    // Last set of cuda matrices that are on the current device
    std::map<size_t,cv::cuda::GpuMat> d_desc_last;


};



#endif /* OV_CORE_TRACK_DESC_GPU_H */