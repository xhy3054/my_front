#pragma once

/****************************basalt光流更新的核心代码**********************************/
#include <Eigen/Dense>

#include "image.h"
#include "patterns.h"

namespace basalt {


// 第一个模板参数表示patch坐标存储的数据类型；第二个模板参数表示patch的形式，定义在“patterns.h”中
template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;

  typedef Eigen::Matrix<int, 2, 1> Vector2i;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 1, 2> Vector2T;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 4> MatrixP4;
  typedef Eigen::Matrix<int, 2, PATTERN_SIZE> Matrix2Pi;

  static const Matrix2P pattern2; // Eigen的矩阵类型的patch对象，在此头文件最后会使用第二个模板参数设置

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() { mean = 0; }

  OpticalFlowPatch(const Image<const uint16_t> &img, const Vector2 &pos) {
    setFromImage(img, pos);
  }

  /**
   * @function     通过图像与点的位置设置点的灰度、patch的信息、光流更新的雅可比数据等（由于是逆向的光流，所以此处雅可比是在原图上求的，并且只求一次就好了）
   *
   * @img   图像
   * @pos   目标点在图像中的位置
   * 
   */
  void setFromImage(const Image<const uint16_t> &img, const Vector2 &pos) {
    this->pos = pos;

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);

    MatrixP2 grad;

    //遍历patch中的每个位置
    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p = pos + pattern2.col(i);  //patch中的第i个位置坐标
      if (img.InBounds(p, 2)) { //如果p在图像的有效范围内（没有超出边界）
        Vector3 valGrad = img.interpGrad<Scalar>(p);  //获得该点的信息（三维向量，第一个为灰度值，后两个是梯度）
        data[i] = valGrad[0];   //灰度
        sum += valGrad[0];
        grad.row(i) = valGrad.template tail<2>(); //梯度
        grad_sum += valGrad.template tail<2>();
        num_valid_points++; //有效点数量
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;  //该patch的平均灰度

    Scalar mean_inv = num_valid_points / sum; //

    Eigen::Matrix<Scalar, 2, 3> Jw_se2;
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    MatrixP3 J_se2; //雅可比矩阵，pattern_size * 3（灰度对se2向量（两个位移，一个角度？）的求导）

    // 遍历patch中每个位置
    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) {
        data[i] *= mean_inv;  //相比较平均灰度的比值（相对灰度值）
        Vector2 grad_i = grad.row(i); //梯度
        grad.row(i) = num_valid_points * (grad_i * sum - grad_sum * data[i]) /    //x,y处相对灰度值对x,y的求导
                      (sum * sum);
      } else {
        grad.row(i).setZero();  //如果该点无效，梯度设置为0
      }

      // Fill jacobians with respect to SE2 warp 
      // 此处Jw_se2是（x,y）对se2变换的求导，se2变换此处用（\theta, tx,ty）三个元素表示
      Jw_se2(0, 2) = -pattern2(1, i); //-y
      Jw_se2(1, 2) = pattern2(0, i);  //x

      // 最终雅可比（相对灰度值对se2的求导）
      J_se2.row(i) = grad.row(i) * Jw_se2; 
    }

    Matrix3 H_se2 = J_se2.transpose() * J_se2;  //H矩阵 = J^T * J
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv);

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose();  //H^-1 * J
  }
  /**
   * @function     计算参考帧patch的像素值与当前帧patch的像素值的残差
   *
   * @img   当前图像
   * @transformed_pattern   当前帧中patch的位置集合
   * @residual  残差
   * 
   */
  inline bool residual(const Image<const uint16_t> &img,
                       const Matrix2P &transformed_pattern,
                       VectorP &residual) const {
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);
    int num_valid_points = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {  //遍历patch逐个像素
      if (img.InBounds(transformed_pattern.col(i), 2)) {
        residual[i] = img.interp<Scalar>(transformed_pattern.col(i));
        sum += residual[i]; //当前帧的patch像素灰度的和
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }

    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {//逐个像素遍历patch
      if (residual[i] >= 0 && data[i] >= 0) { 
        Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i]; //当前帧当前点的相对灰度 - 前一帧当前点的相对灰度
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;  //有效点数量必须大于1/2
  }

  Vector2 pos;  //当前点的位置
  // 存储patch中每个位置上的相对灰度（当前灰度除以平均灰度），当为负数时表示该位置上的点无效
  VectorP data;  // negative if the point is not valid 

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3P H_se2_inv_J_se2_T;

  Scalar mean;  //该patch的平均灰度
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P
    OpticalFlowPatch<Scalar, Pattern>::pattern2 = Pattern::pattern2;

}  // namespace basalt
