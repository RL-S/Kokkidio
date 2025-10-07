#ifndef KOKKIDIO_INDEXRANGE_BASE_HPP
#define KOKKIDIO_INDEXRANGE_BASE_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include <utility>
#include <type_traits>

#include "Kokkidio/syclify_macros.hpp"

namespace Kokkidio
{



/**
 * @brief Use this as the template parameter to IndexRange,
 * if you wish to provide the end index (exclusive) 
 * as the limit (second) argument to its constructor, e.g.
 * 
 * auto range = IndexRange(beg, end, LimitIsEnd{});
 * 
 */
struct LimitIsEnd {};

/**
 * @brief Use this (or nothing) as the template parameter to IndexRange,
 * if you wish to provide the size/count 
 * as the limit (second) argument, e.g.
 * 
 * auto range = IndexRange(start, size, LimitIsSize{});
 * 
 * Because this is the default, the above is equivalent to
 * 
 * auto range = IndexRange(start, size);
 */
struct LimitIsSize {};


template<typename _Integer>
class IndexRange {
public:
	using Integer = _Integer;
	using index_type = Integer;
	using value_type = Integer;

	static_assert( std::is_integral_v<Integer> );

	struct Aggregate {
		Integer start, size;
	} values;

public:
	KOKKOS_FUNCTION Integer start() const { return values.start; }
	KOKKOS_FUNCTION Integer size () const { return values.size; }
	KOKKOS_FUNCTION Integer begin() const { return this->start(); }
	KOKKOS_FUNCTION Integer count() const { return this->size(); }
	KOKKOS_FUNCTION Integer end  () const { return this->start() + this->size(); }

	KOKKOS_FUNCTION void start(Integer arg){ values.start = arg; }
	KOKKOS_FUNCTION void size (Integer arg){ values.size  = arg; }
	KOKKOS_FUNCTION void begin(Integer arg){ this->start(arg); }
	KOKKOS_FUNCTION void count(Integer arg){ this->size(arg); }
	KOKKOS_FUNCTION void end  (Integer arg){ this->size( arg - this->start() ); }

	template<typename LimitType = LimitIsSize>
	KOKKOS_FUNCTION void set(Integer start, Integer limit){
		this->start(start);
		if constexpr ( std::is_same_v<LimitType, LimitIsEnd> ){
			this->end(limit);
		} else {
			this->size(limit);
		}
	}

	KOKKOS_FUNCTION IndexRange() = default;

	KOKKOS_FUNCTION IndexRange(Integer size){
		this->set(0, size);
	}

	template<typename LimitType = LimitIsSize>
	KOKKOS_FUNCTION IndexRange(Integer start, Integer limit, LimitType = {}){
		this->template set<LimitType>(start, limit);
	}

	template<typename OtherInt, typename = typename std::enable_if_t<
		std::is_integral_v<OtherInt> && !std::is_same_v<Integer, OtherInt>
	>>
	KOKKOS_FUNCTION IndexRange( const IndexRange<OtherInt>& other ) :
		values {
			static_cast<Integer>(other.values.start),
			static_cast<Integer>(other.values.size )
		}
	{}
};

static_assert( std::is_trivially_copyable_v<IndexRange<int>> );
static_assert( std::is_trivially_copyable_v<IndexRange<Index>> );

template<typename T>
struct is_IndexRange : std::false_type {};

template<typename Integer>
struct is_IndexRange<IndexRange<Integer>> : std::true_type {};

template<typename T>
inline constexpr bool is_IndexRange_v = is_IndexRange<T>::value;


} // namespace Kokkidio


#endif