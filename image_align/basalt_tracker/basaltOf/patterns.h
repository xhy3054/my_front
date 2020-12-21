
#pragma once

/*******************对光流跟踪时的窗口形式进行定义**********************/

#include <Eigen/Dense>

namespace basalt {

//包含24个元素的patch格式
template <class Scalar>
struct Pattern24 {
  //          00  01
  //
  //      02  03  04  05
  //
  //  06  07  08  09  10  11
  //
  //  12  13  14  15  16  17
  //
  //      18  19  20  21
  //
  //          22  23
  //
  // -----> x
  // |
  // |
  // y

  // 该patch的相对坐标集合
  static constexpr Scalar pattern_raw[][2] = {
      {-1, 5},  {1, 5},

      {-3, 3},  {-1, 3},  {1, 3},   {3, 3},

      {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},  {5, 1},

      {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1}, {5, -1},

      {-3, -3}, {-1, -3}, {1, -3},  {3, -3},

      {-1, -5}, {1, -5}

  };

  // 该patch的位置个数
  static constexpr int PATTERN_SIZE =
      sizeof(pattern_raw) / (2 * sizeof(Scalar));

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;  //一个刚好容纳该patch的eigen矩阵类型
  static const Matrix2P pattern2; //一个刚好容纳该patch的eigen矩阵对象
};

//初始化pattern2
template <class Scalar>
const typename Pattern24<Scalar>::Matrix2P Pattern24<Scalar>::pattern2 =
    Eigen::Map<Pattern24<Scalar>::Matrix2P>((Scalar *)
                                                Pattern24<Scalar>::pattern_raw);


//包含52个元素的patch格式
template <class Scalar>
struct Pattern52 {
  //          00  01  02  03
  //
  //      04  05  06  07  08  09
  //
  //  10  11  12  13  14  15  16  17
  //
  //  18  19  20  21  22  23  24  25
  //
  //  26  27  28  29  30  31  32  33
  //
  //  34  35  36  37  38  39  40  41
  //
  //      42  43  44  45  46  47
  //
  //          48  49  50  51
  //
  // -----> x
  // |
  // |
  // y

  static constexpr Scalar pattern_raw[][2] = {
      {-3, 7},  {-1, 7},  {1, 7},   {3, 7},

      {-5, 5},  {-3, 5},  {-1, 5},  {1, 5},   {3, 5},  {5, 5},

      {-7, 3},  {-5, 3},  {-3, 3},  {-1, 3},  {1, 3},  {3, 3},
      {5, 3},   {7, 3},

      {-7, 1},  {-5, 1},  {-3, 1},  {-1, 1},  {1, 1},  {3, 1},
      {5, 1},   {7, 1},

      {-7, -1}, {-5, -1}, {-3, -1}, {-1, -1}, {1, -1}, {3, -1},
      {5, -1},  {7, -1},

      {-7, -3}, {-5, -3}, {-3, -3}, {-1, -3}, {1, -3}, {3, -3},
      {5, -3},  {7, -3},

      {-5, -5}, {-3, -5}, {-1, -5}, {1, -5},  {3, -5}, {5, -5},

      {-3, -7}, {-1, -7}, {1, -7},  {3, -7}

  };

  static constexpr int PATTERN_SIZE =
      sizeof(pattern_raw) / (2 * sizeof(Scalar));

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern52<Scalar>::Matrix2P Pattern52<Scalar>::pattern2 =
    Eigen::Map<Pattern52<Scalar>::Matrix2P>((Scalar *)
                                                Pattern52<Scalar>::pattern_raw);

// Same as Pattern52 but twice smaller
// 与Pattern52一样的排列，不过更加紧凑，导致面积只有一半
template <class Scalar>
struct Pattern51 {
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern51<Scalar>::Matrix2P Pattern51<Scalar>::pattern2 =
    0.5 * Pattern52<Scalar>::pattern2;  //坐标全部除以2

// Same as Pattern52 but 0.75 smaller
// 与Pattern52一样，不过紧凑了0.75倍
template <class Scalar>
struct Pattern50 {
  static constexpr int PATTERN_SIZE = Pattern52<Scalar>::PATTERN_SIZE;

  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  static const Matrix2P pattern2;
};

template <class Scalar>
const typename Pattern50<Scalar>::Matrix2P Pattern50<Scalar>::pattern2 =
    0.75 * Pattern52<Scalar>::pattern2;

}  // namespace basalt
