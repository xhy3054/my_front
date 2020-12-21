#include <Eigen/Geometry>
#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/parallel_for.h>

#include <opencv2/opencv.hpp>

#include "eigen_utils.hpp"
#include "sophus_utils.hpp"
#include "image.h"
#include "image_pyr.h"
#include "../vio_config.h"
#include "patch.h"
#include "patterns.h"
#include "../tic_toc.h"


namespace basalt
{
using KeypointId = uint32_t;  //特征点id的数据类型
//int optical_flow_pattern;
VioConfig config;

class track_base{
  public:
    using Ptr = std::shared_ptr<track_base>; 
    virtual void trackPoints(const basalt::ManagedImagePyr<u_int16_t> &pyr_1,
                     const basalt::ManagedImagePyr<u_int16_t> &pyr_2, 
                     const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> &
                     transform_map_1,
                     Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> &
                     transform_map_2,
                     std::vector<uchar> &status) const{
      std::cout<<"基类的跟踪，啥都不会做"<<std::endl;
    }

};

template<typename Scalar, template <typename> typename Pattern>
class track: public track_base{
  public:
    
    typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

    typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
    typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

    typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
    typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

    typedef Sophus::SE2<Scalar> SE2;

    /**
     * @function 单层图像上一个点到一个点的光流跟踪
     *
     * @img_2               当前图
     * @dp                  patch数据（已经根据前后两幅图在当前patch基础上计算出了光流跟踪需要的数据）
     * @transform           跟踪的初始位置，函数结束会更新到新的位置
     * 
     **/
    inline bool trackPointAtLevel(const Image<const u_int16_t> &img_2,
                                  const PatchT &dp,
                                  Eigen::AffineCompact2f &transform) const
    {
        //TicToc t_level;
        bool patch_valid = true;    

        //固定循环次数迭代优化
        for (int iteration = 0;
             patch_valid && iteration < config.optical_flow_max_iterations;
             iteration++) {
            typename PatchT::VectorP res;

            // 根据当前的transform对patch的坐标进行变换得到patch在当前帧上的位置
            typename PatchT::Matrix2P transformed_pat =
                transform.linear().matrix() * PatchT::pattern2; //对patch坐标使用初始角度进行旋转
            transformed_pat.colwise() += transform.translation();   //旋转后使用初始位置进行平移得到新的patch的坐标集合

            bool valid = dp.residual(img_2, transformed_pat, res);  //计算参考帧patch的像素值与当前帧patch的像素值间的残差（一个向量）

            if (valid) {    //如果计算残差有效（有效点数量大于1/2）
                // H*x = J^T * residual,所以更新量x = - H^-1 * J^T * res
                Vector3 inc = -dp.H_se2_inv_J_se2_T * res;  //更新值，se2的向量
                transform *= SE2::exp(inc).matrix();    //更新光流（se2）

                const int filter_margin = 2;

                if (!img_2.InBounds(transform.translation(), filter_margin))    //如果更新后不在图像范围内
                    patch_valid = false;
            }
            else {
                patch_valid = false;
            }
        }
        //std::cout<<"当前层跟踪花费"<<t_level.toc()<<"ms"<<std::endl;
        return patch_valid;
    }

        /**
     * @function 一个点到一个点的光流跟踪
     *
     * @pyr_1               图1的金字塔
     * @pyr_2               图2的金字塔
     * @transform_map_1     图1中的特征点（位置）
     * @transform_map_2     图2中的特征点（类型同上，此数据结构为输出结果）
     *
     **/
    inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t> &old_pyr,
                           const basalt::ManagedImagePyr<uint16_t> &pyr,
                           const Eigen::AffineCompact2f &old_transform,
                           Eigen::AffineCompact2f &transform) const
    {
        //TicToc t_track_one;
        bool patch_valid = true;    // 

        transform.linear().setIdentity();  //角度变换设为0 

        //多层金字塔逐层遍历，从塔尖到塔底
        for (int level = config.optical_flow_levels; level >= 0 && patch_valid; 
             level--) {
            const float scale = 1 << level;    //当前层位移尺度，比如第0层scale为1，第1层的scale为2

            transform.translation() /= scale;   //将第0层的位移变换到当前层上

            PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);  //根据该层图像与该点位置构造patch，并计算光流跟新需要用到的数据

            // Perform tracking on current level 在当前层执行光流跟踪，将结果保存在transform
            patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

            transform.translation() *= scale;   //将位移变换到第0层
        }

        transform.linear() = old_transform.linear() * transform.linear();
        //std::cout<<"当前点跟踪花费"<<t_track_one.toc()<<"ms"<<std::endl;

        return patch_valid;
    }


    /**
     * @function 帧到帧的光流跟踪
     *
     * @pyr_1               图1的金字塔
     * @pyr_2               图2的金字塔
     * @transform_map_1     图1中的特征点（类型为map，key为点id，value为点位置）
     * @transform_map_2     图2中的特征点（类型同上，此数据结构为输出结果）
     *
     **/
    void trackPoints(const basalt::ManagedImagePyr<u_int16_t> &pyr_1,
                     const basalt::ManagedImagePyr<u_int16_t> &pyr_2, 
                     const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> &
                     transform_map_1,
                     Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> &
                     transform_map_2,
                     std::vector<uchar> &status) const
    {
        size_t num_points = transform_map_1.size(); //图1中特征的数量

        //将图1中的特征点的id与位置分别存储（由于在多线程中只读，所以不用考虑线程安全）
        std::vector<KeypointId> ids;    
        Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec; //元素是eigen中类的对象的容器需要使用eigen自定义的容器来对齐内存

        ids.reserve(num_points);
        init_vec.reserve(num_points);

        for (const auto &kv : transform_map_1) {
            ids.push_back(kv.first);
            init_vec.push_back(kv.second);
        }

        TicToc t_track_all;
        //输出结果的容器，是tbb库提供的线程安全的容器
        tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f> result;
        tbb::concurrent_unordered_map<KeypointId, uchar> result1; //存储追踪结果
        for(size_t i=0; i<num_points; ++i){
          result1[i]=0;
        }

        //共tbb库并行计算时调用的函数体
        auto compute_func = [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t r = range.begin(); r != range.end(); ++r) { //遍历每一个特征点
                const KeypointId id = ids[r];   //取出特征id

                const Eigen::AffineCompact2f &transform_1 = init_vec[r];    //取出特征点位置
                Eigen::AffineCompact2f transform_2 = transform_1;   //初始化当前帧中特征点位置

                bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);    //对该点进行跟踪

                if (valid) {//如果跟踪成功，反向
                    Eigen::AffineCompact2f transform_1_recovered = transform_2; //初始化反向跟踪时图1中点的位置

                    valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);   //反向跟踪该点

                    if (valid) {    //如果反向跟踪成功，计算图1中特征点前后距离
                        float dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                            .squaredNorm();

                        if (dist2 < config.optical_flow_max_recovered_dist2) {  //如果距离小于某阈值，存储该结果
                            //result[id] = transform_2;
                            result1[id] = 1;
                        }
                    }
                }
                result[id] = transform_2;
            }
        };

        tbb::blocked_range<size_t> range(0, num_points);    //共tbb库并行计算时的范围参数

        tbb::parallel_for(range, compute_func); //tbb并行计算
        // compute_func(range);
        std::cout<<"所有点的跟踪花费"<<t_track_all.toc()<<"ms"<<std::endl;

        transform_map_2.clear();    //将结果放入
        transform_map_2.insert(result.begin(), result.end());

        for(size_t i=0; i<num_points; ++i){
          status.push_back(result1[i]);
        }
    }    
};

track_base::Ptr track1;


//typedef OpticalFlowPatch<float, Pattern<float>> PatchT;



std::vector<ManagedImage<uint16_t>::Ptr> get_image_data(const cv::Mat &img1, const cv::Mat &img2){
  //std::cout<<"开始读取图像"<<std::endl;
  std::vector<ManagedImage<uint16_t>::Ptr> res(2);
  for(size_t i=0; i<2; ++i){
    cv::Mat img;
    if(i==0) img=img1.clone();
    else  img=img2.clone();

        if (img.type() == CV_8UC1) {
          //std::cout<<"输入图像为1通道8位格式"<<std::endl;
          res[i].reset(new ManagedImage<uint16_t>(img.cols, img.rows)); //逐个像素复制

          const uint8_t *data_in = img.ptr();
          uint16_t *data_out = res[i]->ptr;

          size_t full_size = img.cols * img.rows;
          for (size_t i = 0; i < full_size; i++) {
            int val = data_in[i];
            val = val << 8;
            data_out[i] = val;
          }
        } else if (img.type() == CV_8UC3) {
          //std::cout<<"输入图像为3通道8位格式"<<std::endl;
          res[i].reset(new ManagedImage<uint16_t>(img.cols, img.rows));

          const uint8_t *data_in = img.ptr();
          uint16_t *data_out = res[i]->ptr;

          size_t full_size = img.cols * img.rows;
          for (size_t i = 0; i < full_size; i++) {
              // TODO: 这一块可能需要注意
            int val = data_in[i * 3];
            val = val << 8;
            data_out[i] = val;
          }
        } else if (img.type() == CV_16UC1) {
          //std::cout<<"输入图像为1通道16位格式"<<std::endl;
          res[i].reset(new ManagedImage<uint16_t>(img.cols, img.rows));
          std::memcpy(res[i]->ptr, img.ptr(),
                      img.cols * img.rows * sizeof(uint16_t));

        } else {
          std::cerr << "img.fmt.bpp " << img.type() << std::endl;
          std::abort();
        }   
  }
  //std::cout<<"读取图像完毕！"<<std::endl;
  return res;
}




bool Of(
	const cv::Mat &img1,
	const cv::Mat &img2,
	const std::vector<cv::Point2f> &kp1,
	std::vector<cv::Point2f> &kp2,
	std::vector<uchar> &status,
	const VioConfig& vio_config
)
{
	// 0. 设置config
	//optical_flow_pattern = config.optical_flow_pattern;
  config = vio_config;

	//1.先将mat转成basalt的image类型
  TicToc t_get_image;
	std::vector<ManagedImage<uint16_t>::Ptr> img_data = get_image_data(img1, img2);
  std::cout<<"图像类型转换花费 "<<t_get_image.toc()<<"ms"<<std::endl;

	//2.构造basalt的金字塔
  TicToc t_pyr;
	std::shared_ptr<std::vector<basalt::ManagedImagePyr<u_int16_t>>> pyramid;    //图像金字塔

  pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
  pyramid->resize(2);

            tbb::parallel_for(tbb::blocked_range<size_t>(0, 2),   //第一个参数是遍历范围，第二个是函数体
                              [&](const tbb::blocked_range<size_t> &r)
                              {
                                  for (size_t i = r.begin(); i != r.end(); ++i) {
                                      pyramid->at(i).setFromImage(      //图像金字塔构建，第一个参数图像，第二个金字塔层数
                                          *img_data[i],
                                          config.optical_flow_levels);
                                  }
                              });   
  std::cout<<"构建金字塔花费"<<t_pyr.toc()<<"ms"<<std::endl; 

	//3.挨个点光流逐层跟踪
  TicToc t_track;
  Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> old_pose,new_pose;
	for(size_t i=0; i<kp1.size(); ++i){
		Eigen::AffineCompact2f transform0, transform1;
		transform0.setIdentity();
		transform0.translation() << kp1[i].x, kp1[i].y;//transform.translation() = (kp1[i].x, kp1[i].y);
		old_pose[i] = transform0;
/*
		if(config.use_initial){
			if(kp1.size()!=kp2.size()){
				std::cout<<"前后两幅图特征数量不一致！"<<std::endl;
				return false;
			}
			transform1.setIdentity();
			transform1.translation() << kp2[i].x, kp2[i].y;			
		}
		else{
			transform1 = transform0;
		}

		new_pose[i] = transform1;
*/
	}    

            switch(config.optical_flow_pattern){
              case 24:
                track1.reset(new track<float, Pattern24>());
                break;
              case 52:
                track1.reset(new track<float, Pattern52>());             
                break;
              case 51:
                track1.reset(new track<float, Pattern51>()); 
                break;
              case 50:
                track1.reset(new track<float, Pattern50>()); 
                break;
              default:
                std::cerr << "config.optical_flow_pattern "
                      << config.optical_flow_pattern << "is not supported."
                      << std::endl;
                std::abort();
            }  

	track1->trackPoints(pyramid->at(0), pyramid->at(1), old_pose, new_pose, status);
  std::cout<<"跟踪花费"<<t_track.toc()<<"ms"<<std::endl;

	//4.返回数据
	kp2.resize(kp1.size());
	for(size_t i=0; i<kp1.size();++i){
		kp2[i].x = new_pose[i].translation()(0);
		kp2[i].y = new_pose[i].translation()(1);
	}

	return true;

	
}








}//end basalt namespace
