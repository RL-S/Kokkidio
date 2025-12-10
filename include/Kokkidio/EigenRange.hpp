#ifndef KOKKIDIO_EIGENRANGE_HPP
#define KOKKIDIO_EIGENRANGE_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/EigenRange_func.hpp"
#include "Kokkidio/EigenTypeHelpers.hpp"

namespace Kokkidio
{

template<Target _target>
class EigenRange {
public:
	static constexpr Target target {_target};
	static constexpr bool isDevice {target == Target::device};
	static constexpr bool isHost   {target == Target::host};
	using MemberType = std::conditional_t<isHost, IndexRange<Index>, int>;

protected:
	MemberType m_rng;

public:
	KOKKOS_FUNCTION
	EigenRange() = default;

	KOKKOS_FUNCTION
	EigenRange( MemberType arg ) :
		m_rng { std::move(arg) }
	{}

	KOKKOS_FUNCTION
	auto get() const -> const MemberType& {
		return m_rng;
	}

	KOKKOS_FUNCTION
	auto get() -> MemberType& {
		return m_rng;
	}

	KOKKOS_FUNCTION
	constexpr auto size() const -> std::conditional_t<isHost, Index, int> {
		if constexpr (isHost){
			return m_rng.size();
		} else {
			return 1;
		}
	}

	KOKKOS_FUNCTION
	auto asIndexRange() const
		-> std::conditional_t<isHost, const IndexRange<Index>&, IndexRange<Index>>
	{
		if constexpr (isHost){
			return m_rng;
		} else {
			return {m_rng, 1};
		}
	}

	template<typename EigenObj>
	KOKKIDIO_INL_AUTO colRange( EigenObj&& obj ) const {
		return Kokkidio::colRange( this->get(), std::forward<EigenObj>(obj) );
	}

	template<typename EigenObj>
	KOKKIDIO_INL_AUTO rowRange( EigenObj&& obj ) const {
		return Kokkidio::rowRange( this->get(), std::forward<EigenObj>(obj) );
	}

	template<typename EigenObj>
	KOKKIDIO_INL_AUTO
	range( EigenObj&& obj ) const {
		return Kokkidio::autoRange( this->get(), std::forward<EigenObj>(obj) );
	}

	template<typename EigenObj>
	KOKKIDIO_INL_AUTO
	operator() ( EigenObj&& obj ) const {
		return Kokkidio::autoRange( this->get(), std::forward<EigenObj>(obj) );
	}
};

} // namespace Kokkidio

#endif