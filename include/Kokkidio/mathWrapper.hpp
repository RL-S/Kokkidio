#ifndef KOKKIDIO_MATHWRAPPER_HPP
#define KOKKIDIO_MATHWRAPPER_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include <cassert>
#include <cmath>

#include "Kokkidio/typeHelpers.hpp"
#include "Kokkidio/typeAliases.hpp"
#include "Kokkidio/syclify_macros.hpp"

namespace Kokkidio {

namespace detail {

#ifdef __CUDA_ARCH__
#define KOKKIDIO_MATH_NAMESPACE ::
#else
#define KOKKIDIO_MATH_NAMESPACE std::
#endif

#define KOKKIDIO_CALL_REAL(FUNC_FLOAT, FUNC_DOUBLE, ...) \
if constexpr ( std::is_same_v<scalar, float> ){ \
	return KOKKIDIO_MATH_NAMESPACE FUNC_FLOAT(__VA_ARGS__); \
} else { \
	assert( (std::is_same_v<scalar, double>) ); \
	return KOKKIDIO_MATH_NAMESPACE FUNC_DOUBLE(__VA_ARGS__); \
}

/* https://en.cppreference.com/w/Talk:Main_Page/suggestions:
 * libstdc++ is non-compliant with C++11 in regards to floating point function naming */
// #ifdef __GNUC__
// #define powf pow
// #define sqrtf sqrt
// #endif

KOKKOS_FUNCTION inline auto pow(scalar base, scalar exp){
	#ifdef __GNUC__
	KOKKIDIO_CALL_REAL(pow, pow, base, exp)
	#else
	KOKKIDIO_CALL_REAL(powf, pow, base, exp)
	#endif
}

KOKKOS_FUNCTION inline auto sqrt(scalar arg){
	#ifdef __GNUC__
	KOKKIDIO_CALL_REAL(sqrt, sqrt, arg)
	#else
	KOKKIDIO_CALL_REAL(sqrtf, sqrt, arg)
	#endif
}

} // namespace detail

template<typename T, typename Scalar>
KOKKOS_FUNCTION auto pow(const T& base, Scalar exp) -> decltype(auto) {
	using U = remove_qualifiers<T>;
	if constexpr ( std::is_base_of_v<Eigen::DenseBase<U>, U> ){
		#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
			if constexpr ( std::is_floating_point_v<Scalar> ){
				assert( base.size() == 1 );
				return detail::pow(base(0,0), exp);
			} else {
				return base.pow(exp);
			}
		#else
			return base.pow(exp);
		#endif
	} else {
		return detail::pow(base, exp);
	}
}

} // namespace Kokkidio

#endif