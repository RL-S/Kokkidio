#ifndef KOKKIDIO_SYCLIFY_MACROS_HPP
#define KOKKIDIO_SYCLIFY_MACROS_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include <Kokkos_Core.hpp>
#include "macros.hpp"

#define KOKKIDIO_BACKENDS_COUNT KOKKIDIO_USE_CUDA + KOKKIDIO_USE_HIP + KOKKIDIO_USE_SYCL + KOKKIDIO_USE_OMPT

// #if KOKKIDIO_USE_CUDA + KOKKIDIO_USE_HIP + KOKKIDIO_USE_SYCL + KOKKIDIO_USE_OMPT > 1
#if KOKKIDIO_BACKENDS_COUNT > 1
#error "Only one backend can be used at a time!"
#elif KOKKIDIO_BACKENDS_COUNT == 0
#define KOKKIDIO_CPU_ONLY
#endif

#if defined(KOKKIDIO_USE_CUDA) || defined(KOKKIDIO_USE_HIP)
#define KOKKIDIO_USE_CUDAHIP
#endif

/* causes error: identifier NAME is undefined in device code */
#define KOKKIDIO_CONSTANT__old(NAME, VALUE) \
	template<typename T> \
	inline constexpr auto NAME##_v = \
		std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
	inline constexpr auto NAME = NAME##_v<scalar>

/* While Eigen already defines a macro for __host__ __device__ attributes
 * for multiple compilers, its definition depends on EIGEN_NO_CUDA,
 * which gets conditionally defined for host code compiled with nvcc,
 * and thus we have to define our own */
#if defined(KOKKIDIO_USE_CUDA)
	#define KOKKIDIO_HOST_DEVICE_VAR(...) __device__ __VA_ARGS__;
#elif defined(KOKKIDIO_USE_HIP)
	/* HIP doesn't need the __device__ specifier - 
	 * and in fact doesn't create the variable on host if it's used. */
	/* Duplication here to point out that this is NOT an oversight */
	#define KOKKIDIO_HOST_DEVICE_VAR(...) __VA_ARGS__;
#elif defined(KOKKIDIO_USE_OMPT)
	#define KOKKIDIO_HOST_DEVICE_VAR(...) \
	_Pragma("omp declare target") \
	__VA_ARGS__; \
	_Pragma("omp end declare target")
#else
	#define KOKKIDIO_HOST_DEVICE_VAR(...) __VA_ARGS__;
#endif

#define KOKKIDIO_CONSTANT KOKKIDIO_HOST_DEVICE_VAR

/* the list of SYCL-compatible CPUs is extremely short, 
 * and, by sheer coincidence I'm sure, doesn't include any AMD CPUs.
 * Defining this symbol means that, with SYCL as the backend,
 * parallel dispatch on host uses OpenMP instead of SYCL. */
#define KOKKIDIO_SYCL_DISABLE_ON_HOST


#endif