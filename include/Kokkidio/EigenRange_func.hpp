#ifndef KOKKIDIO_EIGENRANGE_FUNC_HPP
#define KOKKIDIO_EIGENRANGE_FUNC_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/ViewMap.hpp"
#include "Kokkidio/DualViewMap.hpp"
#include "Kokkidio/ompSegment.hpp"
#include "Kokkidio/typeHelpers.hpp"
#include "Kokkidio/typeAliases.hpp"
#include "Kokkidio/macros.hpp"
#include "Kokkidio/TargetEnum.hpp"

namespace Kokkidio
{



#define KOKKIDIO_INL_AUTO \
KOKKOS_FUNCTION inline decltype(auto)


namespace detail
{

template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO colRange( const Rng& rng, EigenObj&& obj ){
	using URng = remove_qualifiers<Rng>;
	using UObj = remove_qualifiers<EigenObj>;
	static_assert(std::is_base_of_v<Eigen::DenseBase<UObj>, UObj>);
	if constexpr ( std::is_integral_v<URng> ){
		assert(obj.cols() > rng);
		return obj.col(rng);
	} else {
		static_assert( is_IndexRange_v<Rng> );
		assert( obj.cols() >= rng.start() + rng.size() );
		return obj.middleCols( rng.start(), rng.size() );
	}
}

template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO rowRange( const Rng& rng, EigenObj&& obj ){
	using URng = remove_qualifiers<Rng>;
	using UObj = remove_qualifiers<EigenObj>;
	static_assert(std::is_base_of_v<Eigen::DenseBase<UObj>, UObj>);
	if constexpr ( std::is_same_v<URng, Index> ){
		assert(obj.rows() > rng);
		return obj.row(rng);
	} else {
		static_assert( is_IndexRange_v<Rng> );
		assert( obj.rows() >= rng.start() + rng.size() );
		return obj.middleRows( rng.start(), rng.size() );
	}
}

template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO autoRange( const Rng& rng, EigenObj&& obj ){
	using UObj = remove_qualifiers<EigenObj>;
	static_assert(std::is_base_of_v<Eigen::DenseBase<UObj>, UObj>);
	if constexpr ( UObj::ColsAtCompileTime == 1 ){
		return detail::rowRange(rng, std::forward<EigenObj>(obj) );
	} else {
		/* the whole idea here doesn't really make sense for RowMajor objects */
		static_assert( !UObj::IsRowMajor );
		return detail::colRange(rng, std::forward<EigenObj>(obj) );
	}
}

template<typename T>
KOKKIDIO_INL_AUTO eigenObj( T&& t ){
	using U = remove_qualifiers<T>;
	if constexpr (std::is_base_of_v<Eigen::DenseBase<U>, U>){
		return t;
	} else
	if constexpr ( is_ViewMap_v<U> ){
		return t.map();
	} else
	if constexpr ( is_DualViewMap_v<U> ){
		return t.map_target();
	}
}

} // namespace detail


template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO colRange( const Rng& rng, EigenObj&& obj ){
	return detail::colRange( rng, detail::eigenObj(obj) );
}

template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO rowRange( const Rng& rng, EigenObj&& obj ){
	return detail::rowRange( rng, detail::eigenObj(obj) );
}

template<typename EigenObj, typename Rng>
KOKKIDIO_INL_AUTO autoRange( const Rng& rng, EigenObj&& obj ){
	return detail::autoRange( rng, detail::eigenObj(obj) );
}


} // namespace Kokkidio

#endif
