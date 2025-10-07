#ifndef KOKKIDIO_PARALLEL_REDUCE_HPP
#define KOKKIDIO_PARALLEL_REDUCE_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/ParallelRange.hpp"
#include "Kokkidio/RangePolicyHelper.hpp"


namespace Kokkidio
{

/**
 * @brief Contains factory functions for Kokkos::ReducerConcepts.
 * Currently defined are:
 * sum,
 * prod,
 * min,
 * max
 * 
 * Use this in calls to Kokkidio::parallel_reduce, e.g.
 * parallel_reduce<target>( nItems, myFunc, sum(result) );
 */
namespace redux
{

#define KOKKIDIO_REDUX_FACTORY(KOKKOS_NAME, OUR_NAME) \
template<typename Scalar, Target target = Target::host> \
Kokkos::KOKKOS_NAME<Scalar, ExecutionSpace<target>> OUR_NAME(Scalar& result){ \
	return {result}; \
}

KOKKIDIO_REDUX_FACTORY(Sum, sum)
KOKKIDIO_REDUX_FACTORY(Prod, prod)
KOKKIDIO_REDUX_FACTORY(Min, min)
KOKKIDIO_REDUX_FACTORY(Max, max)

#undef KOKKIDIO_REDUX_FACTORY

} // namespace redux

namespace detail
{

// template<typename Func, Target target, typename ReduxArg>
// inline constexpr bool is_range_target_invocable_redux {
// 	std::is_invocable_v<Func, ParallelRange<target>, ReduxArg>
// };

using T = Target;
template<typename Func, typename ReduxArg>
inline constexpr bool is_range_invocable_redux {
	std::is_invocable_v<Func, ParallelRange<T::host  >, ReduxArg> ||
	std::is_invocable_v<Func, ParallelRange<T::device>, ReduxArg>
};

template<typename Policy, typename Func, typename Reducer>
// KOKKIDIO_INLINE 
void reduce_host( const Policy& pol, Func&& func, const Reducer& reducer ){

	using Scalar = typename Reducer::value_type;
	/* Somehow, nvcc doesn't resolve ExecutionSpace<Target::host> 
	 * to the underlying type */
	using Space = typename detail::ExecutionSpace<Target::host>::Type;
	// using Space = ExecutionSpace<Target::host>;
	Scalar var;
	reducer.init(var);

	#define KOKKIDIO_REDUCE_IF(NAME) \
		if constexpr ( std::is_same_v<Reducer, Kokkos::NAME<Scalar, Space>> )
	#define KOKKIDIO_REDUCE_BODY \
		{ func( ParallelRange<Target::host>(pol), var ); }

	KOKKIDIO_REDUCE_IF(Sum){
		KOKKIDIO_OMP_PRAGMA( parallel reduction (+:var) )
		KOKKIDIO_REDUCE_BODY
	}
	KOKKIDIO_REDUCE_IF(Prod){
		KOKKIDIO_OMP_PRAGMA( parallel reduction (*:var) )
		KOKKIDIO_REDUCE_BODY
	}
	KOKKIDIO_REDUCE_IF(Min){
		KOKKIDIO_OMP_PRAGMA( parallel reduction (min:var) )
		KOKKIDIO_REDUCE_BODY
	}
	KOKKIDIO_REDUCE_IF(Max){
		KOKKIDIO_OMP_PRAGMA( parallel reduction (max:var) )
		KOKKIDIO_REDUCE_BODY
	}

	#undef KOKKIDIO_REDUCE_BODY
	#undef KOKKIDIO_REDUCE_IF

	reducer.reference() = var;
}

} // namespace detail

/**
 * @brief Parallel dispatch with reduction, similar to Kokkos::parallel_reduce.
 * Optimises reductions on host (@a target == Target::host),
 * when @a func is invocable with the following two arguments:
 * 1. a Kokkidio::ParallelRange, and
 * 2, a reference to @a Reducer::value_type.
 * If @a target == Target::device 
 * or if @a func is not invocable with the above arguments, 
 * then this function redirects to Kokkos::parallel_reduce.
 * 
 * @tparam target: Must be specified.
 * @tparam Func: deduced, do not specify.
 * @tparam Reducer: deduced, do not specify.
 * @param maxIdx: Maximum index of the parallel dispatch (exclusive).
 * @param func: Functor called in parallel dispatch, e.g. a KOKKOS_LAMBDA
 * @param reducer Use the factory functions in Kokkidio::redux
 * with the result variable as its argument, e.g. redux::sum(yourResultVar).
 */
template<Target target = DefaultTarget, typename Policy, typename Func, typename Reducer>
// KOKKIDIO_INLINE 
void parallel_reduce( const Policy& pol, Func&& func, const Reducer& reducer ){

	using Scalar = typename Reducer::value_type;
	/* passing an integer as RangePolicy to Kokkos::parallel_reduce 
	 * resulted in a runtime error.
	 * 
	 * OpenMPTarget: 
	 * "PluginInterface" error: Faliure to copy data from device to host. 
	 * Pointers: host = 0x00007ffd85d442fc, device = 0x00007b0415a00a00, size = 4: 
	 * Error in cuMemcpyDtoHAsync: an illegal memory access was encountered
	 * 
	 * CUDA:
	 * ViewMap.hpp:255: auto Kokkidio::ViewMap<_PlainObjectType, targetArg>::map() const
	 * ->Eigen::Map<_PlainObjectType, 0, Eigen::Stride<0, 0>> 
	 * [with _PlainObjectType = const Eigen::Matrix<float, -1, -1, 0, -1, -1>; 
	 *  Kokkidio::Target targetArg = Kokkidio::Target::host]: 
	 * block: [0,0,0], thread: [0,31,0] Assertion `isAlloc()` failed.
	 * 
	 * So, the error could still well be on this side.
	 * However, it does not occur when simply constructing a RangePolicy
	 * from [0, pol). 
	 * RangePolicy does not have a constructor from an integral scalar,
	 * so it isn't unreasonable to assume that this construction 
	 * is indeed necessary.
	 * */
	auto make_kpol = [&](){ return toRangePolicy<target>(pol); };

	if constexpr ( detail::is_range_invocable_redux<Func, Scalar&> ){
		if constexpr ( target == Target::host ){
			detail::reduce_host( pol, std::forward<Func>(func), reducer );
		} else {
			Kokkos::parallel_reduce( make_kpol(), KOKKOS_LAMBDA(int i, Scalar& result){
				func( ParallelRange<target>(i), result );
			}, reducer );
		}
	} else {
		Kokkos::parallel_reduce( make_kpol(), std::forward<Func>(func), reducer );
	}
}

template<Target target = DefaultTarget, typename Policy, typename Func, typename Reducer>
// KOKKIDIO_INLINE 
void parallel_reduce_chunks(const Policy& pol, Func&& func, const Reducer& reducer){

	using Scalar = typename Reducer::value_type;
	auto make_kpol = [&](){ return toRangePolicy<target>(pol); };

	static_assert( std::is_invocable_v<Func, Chunk<target>, Scalar&> );

	if constexpr ( target == Target::host ){
		detail::reduce_host(
			pol,
			[&](ParallelRange<Target::host> rng, Scalar& var){
				// for (Index i=0; i<rng.nChunks(); ++i){
				// 	func( rng.make_chunk(i), var );
				// }
				Index chunkStart {rng.get().start()}, chunkSize;
				while ( chunkStart < rng.get().end() ){
					chunkSize = rng.chunkSize(chunkStart);
					func( rng.make_chunk_s(chunkStart, chunkSize), var );
					chunkStart += chunkSize;
				}
			},
			reducer
		);
	} else {
		static_assert( target == Target::device );
		Kokkos::parallel_reduce( make_kpol(), KOKKOS_LAMBDA(int i, Scalar& result){
			func( Chunk<target>(i), result );
		}, reducer );
	}
}
	
} // namespace Kokkidio

#endif
