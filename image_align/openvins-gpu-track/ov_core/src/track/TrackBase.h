#ifndef OV_CORE_TRACK_BASE_H
#define OV_CORE_TRACK_BASE_H


#include <map>
#include <iostream>
#include <thread>

#include <ros/ros.h>
#include <boost/thread.hpp>
//#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Grider_FAST.h"
#include "Grider_DOG.h"
#include "FeatureDatabase.h"


/**
 * @brief Visual feature tracking base class
 *
 * This is the base class for all our visual trackers.
 * The goal here is to provide a common interface so all underlying trackers can simply hide away all the complexities.
 * We have something called the "feature database" which has all the tracking information inside of it.
 * The user can ask this database for features which can then be used in an MSCKF or batch-based setting.
 * The feature tracks store both the raw (distorted) and undistorted/normalized values.
 * Right now we just support two camera models, see: undistort_point_brown() and undistort_point_fisheye().
 *
 * This base class also handles most of the heavy lifting with the visualalization, but the sub-classes can override
 * this and do their own logic if they want (i.e. the auroctag tracker has its own logic for visualization).
 */
class TrackBase
{

public:

    /**
     * @brief Public default constructor
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     */
    TrackBase(std::map<size_t,Eigen::Matrix3d> camera_k,
              std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
              std::map<size_t,bool> camera_fisheye):
            database(new FeatureDatabase()), num_features(200) {
        // Save the camera parameters
        this->camera_k = camera_k;
        this->camera_d = camera_d;
        this->camera_fisheye = camera_fisheye;
        // Convert values to the OpenCV format
        for(auto const& camKp : camera_k) {
            cv::Matx33d tempK;
            tempK(0,0) = camKp.second(0,0);
            tempK(0,1) = camKp.second(0,1);
            tempK(0,2) = camKp.second(0,2);
            tempK(1,0) = camKp.second(1,0);
            tempK(1,1) = camKp.second(1,1);
            tempK(1,2) = camKp.second(1,2);
            tempK(2,0) = camKp.second(2,0);
            tempK(2,1) = camKp.second(2,1);
            tempK(2,2) = camKp.second(2,2);
            camera_k_OPENCV.insert({camKp.first,tempK});
        }
        for(auto const& camDp : camera_d) {
            cv::Vec4d tempD;
            tempD(0) = camDp.second(0,0);
            tempD(1) = camDp.second(1,0);
            tempD(2) = camDp.second(2,0);
            tempD(3) = camDp.second(3,0);
            camera_d_OPENCV.insert({camDp.first,tempD});
        }
    }

    /**
     * @brief Public constructor with configuration variables
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     */
    TrackBase(std::map<size_t,Eigen::Matrix3d> camera_k,
              std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
              std::map<size_t,bool> camera_fisheye,
              int numfeats, int numaruco):
            database(new FeatureDatabase()), num_features(numfeats) {
        // Our current feature ID should be larger then the number of aruco tags we have
        currid = (size_t)numaruco+1;
        // Save the camera parameters
        this->camera_k = camera_k;
        this->camera_d = camera_d;
        this->camera_fisheye = camera_fisheye;
        // Convert values to the OpenCV format
        for(auto const& camKp : camera_k) {
            cv::Matx33d tempK;
            tempK(0,0) = camKp.second(0,0);
            tempK(0,1) = camKp.second(0,1);
            tempK(0,2) = camKp.second(0,2);
            tempK(1,0) = camKp.second(1,0);
            tempK(1,1) = camKp.second(1,1);
            tempK(1,2) = camKp.second(1,2);
            tempK(2,0) = camKp.second(2,0);
            tempK(2,1) = camKp.second(2,1);
            tempK(2,2) = camKp.second(2,2);
            camera_k_OPENCV.insert({camKp.first,tempK});
        }
        for(auto const& camDp : camera_d) {
            cv::Vec4d tempD;
            tempD(0) = camDp.second(0,0);
            tempD(1) = camDp.second(1,0);
            tempD(2) = camDp.second(2,0);
            tempD(3) = camDp.second(3,0);
            camera_d_OPENCV.insert({camDp.first,tempD});
        }
    }


    /**
     * @brief Process a new monocular image
     * @param timestamp timestamp the new image occurred at
     * @param img new cv:Mat grayscale image
     * @param cam_id the camera id that this new image corresponds too
     */
    virtual void feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) = 0;

    /**
     * @brief Process new stereo pair of images
     * @param timestamp timestamp this pair occured at (stereo is synchronised)
     * @param img_left first grayscaled image
     * @param img_right second grayscaled image
     * @param cam_id_left first image camera id
     * @param cam_id_right second image camera id
     */
    virtual void feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) = 0;

    /**
     * @brief Shows features extracted in the last image
     * @param img_out image to which we will overlayed features on
     * @param r1,g1,b1 first color to draw in
     * @param r2,g2,b2 second color to draw in
     */
    virtual void display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2);

    /**
     * @brief Shows a "trail" for each feature (i.e. its history)
     * @param img_out image to which we will overlayed features on
     * @param r1,g1,b1 first color to draw in
     * @param r2,g2,b2 second color to draw in
     */
    virtual void display_history(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2);

    /**
     * @brief Get the feature database with all the track information
     * @return FeatureDatabase pointer that one can query for features
     */
    FeatureDatabase* get_feature_database() {
        return database;
    }

protected:

    /**
     * @brief Main function that will undistort/normalize a point.
     * @param pt_in uv 2x1 point that we will undistort
     * @param cam_id id of which camera this point is in
     * @return undistorted 2x1 point
     *
     * Given a uv point, this will undistort it based on the camera matrices.
     * This will call on the model needed, depending on what type of camera it is!
     * So if we have fisheye for camera_1 is true, we will undistort with the fisheye model.
     * In Kalibr's terms, the non-fisheye is `pinhole-radtan` while the fisheye is the `pinhole-equi` model.
     */
    cv::Point2f undistort_point(cv::Point2f pt_in, size_t cam_id) {
        // Determine what camera parameters we should use
        cv::Matx33d camK = this->camera_k_OPENCV[cam_id];
        cv::Vec4d camD = this->camera_d_OPENCV[cam_id];
        // Call on the fisheye if we should!
        if(this->camera_fisheye[cam_id]) {
            return undistort_point_fisheye(pt_in,camK,camD);
        }
        return undistort_point_brown(pt_in,camK,camD);
    }

    /**
     * @brief Undistort function RADTAN/BROWN.
     *
     * Given a uv point, this will undistort it based on the camera matrices.
     * To equate this to Kalibr's models, this is what you would use for `pinhole-radtan`.
     */
    cv::Point2f undistort_point_brown(cv::Point2f pt_in, cv::Matx33d& camK, cv::Vec4d& camD) {
        // Convert to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt_in.x;
        mat.at<float>(0, 1) = pt_in.y;
        mat = mat.reshape(2); // Nx1, 2-channel
        // Undistort it!
        cv::undistortPoints(mat, mat, camK, camD);
        // Construct our return vector
        cv::Point2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        return pt_out;
    }

    /**
     * @brief Undistort function FISHEYE/EQUIDISTANT.
     *
     * Given a uv point, this will undistort it based on the camera matrices.
     * To equate this to Kalibr's models, this is what you would use for `pinhole-equi`.
     */
    cv::Point2f undistort_point_fisheye(cv::Point2f pt_in, cv::Matx33d& camK, cv::Vec4d& camD) {
        // Convert point to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt_in.x;
        mat.at<float>(0, 1) = pt_in.y;
        mat = mat.reshape(2); // Nx1, 2-channel
        // Undistort it!
        cv::fisheye::undistortPoints(mat, mat, camK, camD);
        // Construct our return vector
        cv::Point2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        return pt_out;
    }

    // Database with all our current features
    FeatureDatabase* database;

    // Our camera information (used to undistort our added UV coordinates)
    // NOTE: we do NOT undistort and rectify the image as this takes a lot of time
    std::map<size_t,Eigen::Matrix3d> camera_k;
    std::map<size_t,Eigen::Matrix<double,4,1>> camera_d;

    // If we are a fisheye model or not
    std::map<size_t,bool> camera_fisheye;

    // Camera intrinsics in OpenCV format
    std::map<size_t,cv::Matx33d> camera_k_OPENCV;
    std::map<size_t,cv::Vec4d> camera_d_OPENCV;

    // number of features we should try to track frame to frame
    int num_features;

    // Last set of images
    std::map<size_t,cv::Mat> img_last;

    // Last set of tracked points
    std::map<size_t,std::vector<cv::KeyPoint>> pts_last;

    // Set of IDs of each current feature in the database
    size_t currid = 0;
    std::map<size_t,std::vector<size_t>> ids_last;


};



#endif /* OV_CORE_TRACK_BASE_H */