#ifndef KOKKIDIO_TYPEALIASES_HPP
#define KOKKIDIO_TYPEALIASES_HPP

// #ifndef KOKKIDIO_PUBLIC_HEADER
// #error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
// #endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_PARALLELIZE
// #define NDEBUG
#define EIGEN_NO_DEBUG
#define EIGEN_FAST_MATH 0
#endif

/* defined in checkSycl.cmake */
// #ifdef KOKKIDIO_USE_SYCL
// #define EIGEN_USE_SYCL
// #endif

/* defined in checkEigen.cmake */
// #if defined(KOKKIDIO_USE_CUDA) || defined(KOKKIDIO_USE_SYCL)
// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
// #endif

/* must be included below the preprocessor definitions above */
#include "macros.hpp"
#include "syclify_macros.hpp"

#include <Eigen/Dense>

#include <type_traits>


namespace Kokkidio
{

#ifndef KOKKIDIO_REAL_SCALAR
#define KOKKIDIO_REAL_SCALAR float
// #define KOKKIDIO_REAL_SCALAR double
#endif
using scalar = KOKKIDIO_REAL_SCALAR;

// static constexpr scalar epsilon { std::is_same_v<scalar, float> ? 5e-4 : 1e-9 };
static constexpr scalar epsilon { std::is_same_v<scalar, float> ? 5e-3 : 1e-9 };

using Eigen::Index;
using Eigen::Dynamic;



using Eigen::Matrix;

template<Index rows, Index cols>
using MatrixNNs = Matrix<scalar, rows, cols>;

using MatrixXs = MatrixNNs<Dynamic, Dynamic>;

template<Index rows>
using VectorNs = Matrix<scalar, rows, 1>;

using VectorXs = VectorNs<Dynamic>;

using Vector1s = VectorNs<1>;
using Vector2s = VectorNs<2>;
using Vector3s = VectorNs<3>;



using Eigen::Array;

template<Index rows, Index cols>
using ArrayNNs = Array<scalar, rows, cols>;

template<Index rows>
using ArrayNXs = ArrayNNs<rows, Dynamic>;

using ArrayXXs = ArrayNNs<Dynamic, Dynamic>;

template<Index rows>
using ArrayNs = ArrayNNs<rows, 1>;

using ArrayXs = ArrayNs<Dynamic>;

using Array1s = ArrayNs<1>;
using Array2s = ArrayNs<2>;
using Array3s = ArrayNs<3>;


template<Index rows, Index cols>
using ArrayNNi = Array<Index, rows, cols>;

using ArrayXXi = ArrayNNi<Dynamic, Dynamic>;

using Eigen::Map;

template<Index rows>
using ArrayNXsMap = Map<ArrayNXs<rows>>;

using ArrayXsMap = Map<ArrayXs>;
using ArrayXXsMap = Map<ArrayXXs>;
using ArrayXsCMap  = Map<const ArrayXs >;
using ArrayXXsCMap = Map<const ArrayXXs>;

using OStride = Eigen::OuterStride<Dynamic>;

} // namespace Kokkidio

#endif
