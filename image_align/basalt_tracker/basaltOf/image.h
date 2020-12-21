#pragma once

#include <memory>

#include <Eigen/Dense>

#include "assert.h"

// Renamed Pangoling defines to avoid clash
#define BASALT_HOST_DEVICE
#define BASALT_EXTENSION_IMAGE
#ifdef BASALT_ENABLE_BOUNDS_CHECKS
#define BASALT_BOUNDS_ASSERT(...) BASALT_ASSERT(##__VA_ARGS__)
#else
#define BASALT_BOUNDS_ASSERT(...) ((void)0)
#endif

namespace basalt {

/// @brief Helper class for copying objects.
template <typename T>
struct CopyObject {
  CopyObject(const T& obj) : obj(obj) {}
  const T& obj;
};

inline void PitchedCopy(char* dst, unsigned int dst_pitch_bytes,
                        const char* src, unsigned int src_pitch_bytes,
                        unsigned int width_bytes, unsigned int height) {
  if (dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes) {
    std::memcpy(dst, src, height * width_bytes);
  } else {
    for (unsigned int row = 0; row < height; ++row) {
      std::memcpy(dst, src, width_bytes);
      dst += dst_pitch_bytes;
      src += src_pitch_bytes;
    }
  }
}

/// @brief Image class that supports sub-images, interpolation, element access.
template <typename T>
struct Image {
  using PixelType = T;

  inline Image() : pitch(0), ptr(0), w(0), h(0) {}

  inline Image(T* ptr, size_t w, size_t h, size_t pitch)
      : pitch(pitch), ptr(ptr), w(w), h(h) {}

  BASALT_HOST_DEVICE inline size_t SizeBytes() const { return pitch * h; }

  BASALT_HOST_DEVICE inline size_t Area() const { return w * h; }

  BASALT_HOST_DEVICE inline bool IsValid() const { return ptr != 0; }

  BASALT_HOST_DEVICE inline bool IsContiguous() const {
    return w * sizeof(T) == pitch;
  }

  //////////////////////////////////////////////////////
  // Iterators
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE inline T* begin() { return ptr; }

  BASALT_HOST_DEVICE inline T* end() { return RowPtr(h - 1) + w; }

  BASALT_HOST_DEVICE inline const T* begin() const { return ptr; }

  BASALT_HOST_DEVICE inline const T* end() const { return RowPtr(h - 1) + w; }

  BASALT_HOST_DEVICE inline size_t size() const { return w * h; }

  //////////////////////////////////////////////////////
  // Image transforms
  //////////////////////////////////////////////////////

  template <typename UnaryOperation>
  BASALT_HOST_DEVICE inline void Transform(UnaryOperation unary_op) {
    BASALT_ASSERT(IsValid());

    for (size_t y = 0; y < h; ++y) {
      T* el = RowPtr(y);
      const T* el_end = el + w;
      for (; el != el_end; ++el) {
        *el = unary_op(*el);
      }
    }
  }

  BASALT_HOST_DEVICE inline void Fill(const T& val) {
    Transform([&](const T&) { return val; });
  }

  BASALT_HOST_DEVICE inline void Replace(const T& oldval, const T& newval) {
    Transform([&](const T& val) { return (val == oldval) ? newval : val; });
  }

  inline void Memset(unsigned char v = 0) {
    BASALT_ASSERT(IsValid());
    if (IsContiguous()) {
      std::memset((char*)ptr, v, pitch * h);
    } else {
      for (size_t y = 0; y < h; ++y) {
        std::memset((char*)RowPtr(y), v, pitch);
      }
    }
  }

  inline void CopyFrom(const Image<T>& img) {
    if (IsValid() && img.IsValid()) {
      BASALT_ASSERT(w >= img.w && h >= img.h);
      PitchedCopy((char*)ptr, pitch, (char*)img.ptr, img.pitch,
                  std::min(img.w, w) * sizeof(T), std::min(img.h, h));
    } else if (img.IsValid() != IsValid()) {
      BASALT_ASSERT(false && "Cannot copy from / to an unasigned image.");
    }
  }

  //////////////////////////////////////////////////////
  // Reductions
  //////////////////////////////////////////////////////

  template <typename BinaryOperation>
  BASALT_HOST_DEVICE inline T Accumulate(const T init,
                                         BinaryOperation binary_op) {
    BASALT_ASSERT(IsValid());

    T val = init;
    for (size_t y = 0; y < h; ++y) {
      T* el = RowPtr(y);
      const T* el_end = el + w;
      for (; el != el_end; ++el) {
        val = binary_op(val, *el);
      }
    }
    return val;
  }

  std::pair<T, T> MinMax() const {
    BASALT_ASSERT(IsValid());

    std::pair<T, T> minmax(std::numeric_limits<T>::max(),
                           std::numeric_limits<T>::lowest());
    for (size_t r = 0; r < h; ++r) {
      const T* ptr = RowPtr(r);
      const T* end = ptr + w;
      while (ptr != end) {
        minmax.first = std::min(*ptr, minmax.first);
        minmax.second = std::max(*ptr, minmax.second);
        ++ptr;
      }
    }
    return minmax;
  }

  template <typename Tout = T>
  Tout Sum() const {
    return Accumulate((T)0,
                      [](const T& lhs, const T& rhs) { return lhs + rhs; });
  }

  template <typename Tout = T>
  Tout Mean() const {
    return Sum<Tout>() / Area();
  }

  //////////////////////////////////////////////////////
  // Direct Pixel Access
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE inline T* RowPtr(size_t y) {
    return (T*)((unsigned char*)(ptr) + y * pitch);
  }

  BASALT_HOST_DEVICE inline const T* RowPtr(size_t y) const {
    return (T*)((unsigned char*)(ptr) + y * pitch);
  }

  BASALT_HOST_DEVICE inline T& operator()(size_t x, size_t y) {
    BASALT_BOUNDS_ASSERT(InBounds(x, y));
    return RowPtr(y)[x];
  }

  BASALT_HOST_DEVICE inline const T& operator()(size_t x, size_t y) const {
    BASALT_BOUNDS_ASSERT(InBounds(x, y));
    return RowPtr(y)[x];
  }

  template <typename TVec>
  BASALT_HOST_DEVICE inline T& operator()(const TVec& p) {
    BASALT_BOUNDS_ASSERT(InBounds(p[0], p[1]));
    return RowPtr(p[1])[p[0]];
  }

  template <typename TVec>
  BASALT_HOST_DEVICE inline const T& operator()(const TVec& p) const {
    BASALT_BOUNDS_ASSERT(InBounds(p[0], p[1]));
    return RowPtr(p[1])[p[0]];
  }

  BASALT_HOST_DEVICE inline T& operator[](size_t ix) {
    BASALT_BOUNDS_ASSERT(InImage(ptr + ix));
    return ptr[ix];
  }

  BASALT_HOST_DEVICE inline const T& operator[](size_t ix) const {
    BASALT_BOUNDS_ASSERT(InImage(ptr + ix));
    return ptr[ix];
  }

  //////////////////////////////////////////////////////
  // Interpolated Pixel Access
  //////////////////////////////////////////////////////

  template <typename S>
  inline S interp(const Eigen::Matrix<S, 2, 1>& p) const {
    return interp<S>(p[0], p[1]);
  }

  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGrad(
      const Eigen::Matrix<S, 2, 1>& p) const {
    return interpGrad<S>(p[0], p[1]);
  }

  template <typename S>
  inline float interp(S x, S y) const {
    int ix = x; //将亚像素坐标转成像素坐标
    int iy = y;

    S dx = x - ix;//距离像素位置的距离
    S dy = y - iy;

    S ddx = 1.0f - dx;//距离另一边的距离
    S ddy = 1.0f - dy;

    //返回加权灰度值，越远位置的点权重越小
    return ddx * ddy * (*this)(ix, iy) + ddx * dy * (*this)(ix, iy + 1) +
           dx * ddy * (*this)(ix + 1, iy) + dx * dy * (*this)(ix + 1, iy + 1);
  }

  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGrad(S x, S y) const {
    int ix = x;//将亚像素坐标转成像素坐标
    int iy = y;

    S dx = x - ix;  //距离像素位置的距离
    S dy = y - iy;

    S ddx = 1.0f - dx;  //距离另一边的距离
    S ddy = 1.0f - dy;

    Eigen::Matrix<S, 3, 1> res; //三维向量

    //获得该亚像素位置周围四个像素（正方形）位置的灰度
    const T& px0y0 = (*this)(ix, iy); //第一个像素的灰度
    const T& px1y0 = (*this)(ix + 1, iy); //第二个像素的灰度
    const T& px0y1 = (*this)(ix, iy + 1); //第三个像素的灰度
    const T& px1y1 = (*this)(ix + 1, iy + 1); //第四个像素的灰度

    // 加权灰度值，越远位置的点的占比越小
    res[0] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 +
             dx * dy * px1y1;

    const T& pxm1y0 = (*this)(ix - 1, iy);  //正方形上面的两个点的像素灰度
    const T& pxm1y1 = (*this)(ix - 1, iy + 1);

    // 上面一个正方形的四个像素的加权灰度值
    S res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 +
               dx * dy * px0y1;

    const T& px2y0 = (*this)(ix + 2, iy); //正方形下面的两个点的像素灰度
    const T& px2y1 = (*this)(ix + 2, iy + 1);

    // 下面一个正方形的四个像素的加权灰度值
    S res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 +
               dx * dy * px2y1;

    res[1] = 0.5 * (res_px - res_mx); //上下两个差的1/2

    const T& px0ym1 = (*this)(ix, iy - 1);  //正方形左面的两个点的像素灰度
    const T& px1ym1 = (*this)(ix + 1, iy - 1);

    // 左面一个正方形的四个像素的加权灰度值
    S res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 +
               dx * dy * px1y0;

    const T& px0y2 = (*this)(ix, iy + 2); //正方形右面的两个点的像素灰度
    const T& px1y2 = (*this)(ix + 1, iy + 2);

    // 右面一个正方形的四个像素的加权灰度值
    S res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 +
               dx * dy * px1y2;

    res[2] = 0.5 * (res_py - res_my); //左右两个差的1/2

    return res;
  }

  //////////////////////////////////////////////////////
  // Bounds Checking
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE
  bool InImage(const T* ptest) const {
    return ptr <= ptest && ptest < RowPtr(h);
  }

  BASALT_HOST_DEVICE inline bool InBounds(int x, int y) const {
    return 0 <= x && x < (int)w && 0 <= y && y < (int)h;
  }

  BASALT_HOST_DEVICE inline bool InBounds(float x, float y,
                                          float border) const {
    return border <= x && x < (w - border - 1) && border <= y &&
           y < (h - border - 1);
  }

  template <typename Derived>
  BASALT_HOST_DEVICE inline bool InBounds(
      const Eigen::MatrixBase<Derived>& p,
      const typename Derived::Scalar border) const {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 2);

    using Scalar = typename Derived::Scalar;

    Scalar offset(0);
    if constexpr (std::is_floating_point_v<Scalar>) {
      offset = Scalar(1);
    }

    return border <= p[0] && p[0] < ((int)w - border - offset) &&
           border <= p[1] && p[1] < ((int)h - border - offset);
  }

  //////////////////////////////////////////////////////
  // Obtain slices / subimages
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE inline const Image<const T> SubImage(size_t x, size_t y,
                                                          size_t width,
                                                          size_t height) const {
    BASALT_ASSERT((x + width) <= w && (y + height) <= h);
    return Image<const T>(RowPtr(y) + x, width, height, pitch);
  }

  BASALT_HOST_DEVICE inline Image<T> SubImage(size_t x, size_t y, size_t width,
                                              size_t height) {
    BASALT_ASSERT((x + width) <= w && (y + height) <= h);
    return Image<T>(RowPtr(y) + x, width, height, pitch);
  }

  BASALT_HOST_DEVICE inline Image<T> Row(int y) const {
    return SubImage(0, y, w, 1);
  }

  BASALT_HOST_DEVICE inline Image<T> Col(int x) const {
    return SubImage(x, 0, 1, h);
  }

  //////////////////////////////////////////////////////
  // Data mangling
  //////////////////////////////////////////////////////

  template <typename TRecast>
  BASALT_HOST_DEVICE inline Image<TRecast> Reinterpret() const {
    BASALT_ASSERT_STREAM(sizeof(TRecast) == sizeof(T),
                         "sizeof(TRecast) must match sizeof(T): "
                             << sizeof(TRecast) << " != " << sizeof(T));
    return UnsafeReinterpret<TRecast>();
  }

  template <typename TRecast>
  BASALT_HOST_DEVICE inline Image<TRecast> UnsafeReinterpret() const {
    return Image<TRecast>((TRecast*)ptr, w, h, pitch);
  }

  //////////////////////////////////////////////////////
  // Deprecated methods
  //////////////////////////////////////////////////////

  //    PANGOLIN_DEPRECATED inline
  Image(size_t w, size_t h, size_t pitch, T* ptr)
      : pitch(pitch), ptr(ptr), w(w), h(h) {}

  // Use RAII/move aware pangolin::ManagedImage instead
  //    PANGOLIN_DEPRECATED inline
  void Dealloc() {
    if (ptr) {
      ::operator delete(ptr);
      ptr = nullptr;
    }
  }

  // Use RAII/move aware pangolin::ManagedImage instead
  //    PANGOLIN_DEPRECATED inline
  void Alloc(size_t w, size_t h, size_t pitch) {
    Dealloc();
    this->w = w;
    this->h = h;
    this->pitch = pitch;
    this->ptr = (T*)::operator new(h* pitch);
  }

  //////////////////////////////////////////////////////
  // Data members
  //////////////////////////////////////////////////////

  size_t pitch;
  T* ptr;
  size_t w;
  size_t h;

  BASALT_EXTENSION_IMAGE
};

template <class T>
using DefaultImageAllocator = std::allocator<T>;

/// @brief Image that manages it's own memory, storing a strong pointer to it's
/// memory
template <typename T, class Allocator = DefaultImageAllocator<T>>
class ManagedImage : public Image<T> {
 public:
  using PixelType = T;
  using Ptr = std::shared_ptr<ManagedImage<T, Allocator>>;

  // Destructor
  inline ~ManagedImage() { Deallocate(); }

  // Null image
  inline ManagedImage() {}

  // Row image
  inline ManagedImage(size_t w)
      : Image<T>(Allocator().allocate(w), w, 1, w * sizeof(T)) {}

  inline ManagedImage(size_t w, size_t h)
      : Image<T>(Allocator().allocate(w * h), w, h, w * sizeof(T)) {}

  inline ManagedImage(size_t w, size_t h, size_t pitch_bytes)
      : Image<T>(Allocator().allocate((h * pitch_bytes) / sizeof(T) + 1), w, h,
                 pitch_bytes) {}

  // Not copy constructable
  inline ManagedImage(const ManagedImage<T>& other) = delete;

  // Move constructor
  inline ManagedImage(ManagedImage<T, Allocator>&& img) {
    *this = std::move(img);
  }

  // Move asignment
  inline void operator=(ManagedImage<T, Allocator>&& img) {
    Deallocate();
    Image<T>::pitch = img.pitch;
    Image<T>::ptr = img.ptr;
    Image<T>::w = img.w;
    Image<T>::h = img.h;
    img.ptr = nullptr;
  }

  // Explicit copy constructor
  template <typename TOther>
  ManagedImage(const CopyObject<TOther>& other) {
    CopyFrom(other.obj);
  }

  // Explicit copy assignment
  template <typename TOther>
  void operator=(const CopyObject<TOther>& other) {
    CopyFrom(other.obj);
  }

  inline void Swap(ManagedImage<T>& img) {
    std::swap(img.pitch, Image<T>::pitch);
    std::swap(img.ptr, Image<T>::ptr);
    std::swap(img.w, Image<T>::w);
    std::swap(img.h, Image<T>::h);
  }

  inline void CopyFrom(const Image<T>& img) {
    if (!Image<T>::IsValid() || Image<T>::w != img.w || Image<T>::h != img.h) {
      Reinitialise(img.w, img.h);
    }
    Image<T>::CopyFrom(img);
  }

  inline void Reinitialise(size_t w, size_t h) {
    if (!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h) {
      *this = ManagedImage<T, Allocator>(w, h);
    }
  }

  inline void Reinitialise(size_t w, size_t h, size_t pitch) {
    if (!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h ||
        Image<T>::pitch != pitch) {
      *this = ManagedImage<T, Allocator>(w, h, pitch);
    }
  }

  inline void Deallocate() {
    if (Image<T>::ptr) {
      Allocator().deallocate(Image<T>::ptr,
                             (Image<T>::h * Image<T>::pitch) / sizeof(T));
      Image<T>::ptr = nullptr;
    }
  }

  // Move asignment
  template <typename TOther, typename AllocOther>
  inline void OwnAndReinterpret(ManagedImage<TOther, AllocOther>&& img) {
    Deallocate();
    Image<T>::pitch = img.pitch;
    Image<T>::ptr = (T*)img.ptr;
    Image<T>::w = img.w;
    Image<T>::h = img.h;
    img.ptr = nullptr;
  }

  template <typename T1>
  inline void ConvertFrom(const ManagedImage<T1>& img) {
    Reinitialise(img.w, img.h);

    for (size_t j = 0; j < img.h; j++) {
      T* this_row = this->RowPtr(j);
      const T1* other_row = img.RowPtr(j);
      for (size_t i = 0; i < img.w; i++) {
        this_row[i] = T(other_row[i]);
      }
    }
  }

  inline void operator-=(const ManagedImage<T>& img) {
    for (size_t j = 0; j < img.h; j++) {
      T* this_row = this->RowPtr(j);
      const T* other_row = img.RowPtr(j);
      for (size_t i = 0; i < img.w; i++) {
        this_row[i] -= other_row[i];
      }
    }
  }
};

}  // namespace basalt