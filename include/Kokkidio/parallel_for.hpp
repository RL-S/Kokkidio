#ifndef KOKKIDIO_PARALLEL_FOR_HPP
#define KOKKIDIO_PARALLEL_FOR_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/ParallelRange.hpp"

namespace Kokkidio
{

namespace detail
{

template<typename Func, Target t>
inline constexpr bool is_range_target_invocable {
	std::is_invocable_v<Func, ParallelRange<t>>
};

template<typename Func>
inline constexpr bool is_range_invocable {
	std::is_invocable_v<Func, ParallelRange<Target::host>> ||
	std::is_invocable_v<Func, ParallelRange<Target::device>>
};

template<typename Policy, typename Func>
void parallel_for_host(const Policy& pol, Func&& func){
	printd("Redirected Kokkidio::parallel_for to parallel_for_host.\n");
	auto range { toIndexRange(pol) };
	KOKKIDIO_OMP_PRAGMA(parallel for)
	for (int i=range.start(); i<range.end(); ++i){
		func(i);
	}
}

template<Target target, typename Policy, typename Func>
void parallel_for( const Policy& pol, Func&& func ){
	using T = Target;

	if constexpr ( target == T::device ){
		Kokkos::parallel_for( pol, std::forward<Func>(func) );
	} else
	if constexpr ( target == T::host ){
		#if defined(KOKKIDIO_USE_SYCL) && !defined(KOKKIDIO_SYCL_DISABLE_ON_HOST)
			Kokkos::parallel_for( pol, std::forward<Func>(func) );
		#else
			parallel_for_host( pol, std::forward<Func>(func) );
		#endif
	}
}

} // namespace detail


template<Target target = DefaultTarget, typename Policy, typename Func>
void parallel_for_range(
	const Policy& pol,
	Func&& func
){
	using T = Target;

	static_assert( detail::is_range_target_invocable<Func, target> );

	/* If a Kokkos::RangePolicy is not set to the correct ExecutionSpace,
	 * that shouldn't be an error */
	// if constexpr ( is_RangePolicy_v<Policy> ){
	// 	using P = PolicyHelper<Policy>;
	// 	static_assert( P::target == target );
	// }

	if constexpr (target == T::host){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			/* we could place a condition here to only call the function,
			 * if the ParallelRange has a non-zero size.
			 * While this would make writing most code easier,
			 * it would also prevent any synchronisations or reductions 
			 * from working, which seems to be a much bigger downside. */
			func( ParallelRange<T::host>{pol} );
		}
	} else
	if constexpr (target == T::device){
		/* Additional indirection via lambda is needed,
		 * instead of implicit conversion from Kokkidio::id<dim> to ParallelRange.
		 * Otherwise, when using SYCL, the compilation fails with: 
		 * "error: no matching function for call to 'getElement'" */
		Kokkos::parallel_for(pol, KOKKOS_LAMBDA(int i){
			func( ParallelRange<T::device>(i) );
		});
	} else {
		static_assert( dependent_false<Func>::value && false );
	}
}


template<Target target = DefaultTarget, typename Policy, typename Func>
void parallel_for( const Policy& pol, Func&& func ){
	if constexpr ( detail::is_range_invocable<Func> ){
		parallel_for_range<target>( pol, std::forward<Func>(func) );
	} else {
		detail::parallel_for<target>( pol, std::forward<Func>(func) );
	}
}

template<Target target = DefaultTarget, typename Policy, typename Func>
// KOKKIDIO_INLINE 
void parallel_for_chunks(const Policy& pol, Func&& func){

	if constexpr ( target == Target::host ){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			ParallelRange<target> rng {pol};
			rng.for_each_chunk( std::forward<Func>(func) );
		}
	} else {
		static_assert( target == Target::device );
		static_assert( std::is_invocable_v<Func, EigenRange<target>> );
		Kokkos::parallel_for( pol, KOKKOS_LAMBDA(int i){
			func( EigenRange<target>(i) );
		});
	}
}

} // namespace Kokkidio

#endif
