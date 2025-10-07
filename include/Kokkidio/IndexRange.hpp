#ifndef KOKKIDIO_INDEXRANGE_HPP
#define KOKKIDIO_INDEXRANGE_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/IndexRange_base.hpp"
#include "Kokkidio/RangePolicyHelper.hpp"

namespace Kokkidio
{

namespace detail
{

/* alternative to std::conditional,
 * to prevent the false clauses from being evaluated */
template<typename Policy>
auto typeFinder(){
	if constexpr (std::is_integral_v<Policy>){
		return Policy{};
	} else
	if constexpr (
		Kokkidio::is_RangePolicy_v<Policy> || 
		Kokkidio::is_IndexRange_v<Policy>
	){
		return typename Policy::index_type{};
	}
}

template<typename Policy>
using IndexType = decltype( typeFinder<Policy>() );

} // namespace detail

/**
 * @brief Takes an integer, IndexRange, or Kokkos::RangePolicy,
 * and returns an IndexRange which matches the argument 
 * both in value and index_type.
 * 
 * @tparam Policy 
 * @param pol 
 * @return IndexRange<detail::IndexType<Policy>> 
 */
template<typename Policy>
IndexRange<detail::IndexType<Policy>>
toIndexRange( const Policy& pol ){
	if constexpr ( Kokkidio::is_RangePolicy_v<Policy> ){
		return { pol.begin(), pol.end() };
	} else {
		static_assert(
			std::is_integral_v<Policy> ||
			is_IndexRange_v<Policy>
		);
		/* For scalar integers, there is a corresponding IndexRange constructor,
		 * which sets begin to zero and size to the integer value.
		 * For IndexRange types of the same integer type,
		 * no operation is required,
		 * while for IndexRange types with a different integer type,
		 * a casting constructor is available. */
		return pol;
	}
}

} // namespace Kokkidio

#endif