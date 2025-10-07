#ifndef KOKKIDIO_RANGEPOLICYHELPER_HPP
#define KOKKIDIO_RANGEPOLICYHELPER_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/TargetSpaces.hpp"
#include "Kokkidio/IndexRange_base.hpp"

namespace Kokkidio
{

template<typename T>
struct is_RangePolicy : std::false_type {};

template<typename ... Args>
struct is_RangePolicy<Kokkos::RangePolicy<Args ...>> : std::true_type {};

template<typename T>
inline constexpr bool is_RangePolicy_v = is_RangePolicy<T>::value;


template<Target target, typename Policy>
Kokkos::RangePolicy<ExecutionSpace<target>>
toRangePolicy( const Policy& pol ){
	if constexpr ( std::is_integral_v<Policy> ){
		return {0, pol};
	} else
	if constexpr ( is_IndexRange_v<Policy> ){
		return { pol.begin(), pol.end() };
	} else
	if constexpr ( is_RangePolicy_v<Policy> ){
		return pol;
	} else
	{
		static_assert( dependent_false<Policy>::type, "Unknown Policy type." );
	}
}


template<typename Policy>
struct PolicyHelper {
	static_assert( is_RangePolicy_v<Policy> );
	using index_type = typename Policy::index_type;
	/* no guarantees were made about index_type and member_type being the same
	 * in the docs, but it's currently implemented as such,
	 * and we rely on it. */
	static_assert( std::is_same_v<index_type, typename Policy::member_type> );
	static_assert( std::is_convertible_v<index_type, Index> );

	using execution_space = typename Policy::execution_space;
	static constexpr Target target { spaceToTarget<execution_space> };
	static constexpr bool
		isHost   { target == Target::host },
		isDevice { target == Target::device };
};

} // namespace Kokkidio

#endif