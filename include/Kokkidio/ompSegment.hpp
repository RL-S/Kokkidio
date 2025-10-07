#ifndef KOKKIDIO_OMPSEGMENT_HPP
#define KOKKIDIO_OMPSEGMENT_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/IndexRange.hpp"
#include "Kokkidio/macros.hpp"
#include <utility>
#include <cassert>

#ifdef KOKKIDIO_OPENMP
#include <omp.h>
#include <algorithm> // for std::max, only needed when OpenMP is enabled.
#endif

namespace Kokkidio
{



template<typename Integer>
Integer ompSegmentMaxSize(Integer n){
	#ifdef KOKKIDIO_OPENMP
	return n / static_cast<Integer>( omp_get_max_threads() ) + 1;
	#else
	return n;
	#endif
}


/* Template for defining omp segment */
template<typename Integer>
IndexRange<Integer> ompSegment( IndexRange<Integer> range ){
	assert(range.size() >= 0);
	#ifdef KOKKIDIO_OPENMP

	/* Simplest approach: Remainder is added to last thread */
	// Integer
	// nLen {n / omp_get_num_threads()},
	// nRem {n % omp_get_num_threads()};

	// Integer nBeg {nLen * omp_get_thread_num()};
	// nLen += (omp_get_thread_num() == omp_get_num_threads() - 1) ? nRem : 0;
	// return {nBeg, nLen};

	/* More even approach:
	 * The remainder nRem is distributed evenly among the first nRem threads */
	Integer
		n  {range.size()},
		np {static_cast<Integer>( omp_get_num_threads() )},
		p  {static_cast<Integer>( omp_get_thread_num () )},
		count { n / np },
		rem   { n % np },
		start;

	start = range.start() + count * p + std::min(rem, p);
	count += p < rem ? 1 : 0;

	return {start, count};

	#else
	return range;
	#endif
}

template<typename Policy>
IndexRange<detail::IndexType<Policy>>
ompSegment( const Policy& pol ){
	return ompSegment( toIndexRange(pol) );
}


} // namespace Kokkidio

#endif
