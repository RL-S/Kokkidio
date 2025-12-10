#ifndef KOKKIDIO_PARALLELRANGE_HPP
#define KOKKIDIO_PARALLELRANGE_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/EigenRange.hpp"
#include "Kokkidio/ParallelRange_buffer.hpp"

#include <tuple>

namespace Kokkidio
{


template<Target _target = DefaultTarget>
class ParallelRange : public EigenRange<_target> {
public:
	static constexpr Target target {_target};
	using Base = EigenRange<target>;
	static constexpr bool
		isDevice {Base::isDevice},
		isHost   {Base::isHost};
	using MemberType = typename Base::MemberType;
	using ChunkType = EigenRange<target>;
	using ChunkInfoType = ChunkInfo<target>;

private:
	using Base::m_rng;
	ChunkInfoType m_chunks;
public:

	KOKKOS_FUNCTION ParallelRange() = default;

	template<typename Policy>
	KOKKOS_FUNCTION ParallelRange( const Policy& pol ){
		set(pol);
	}

protected:
	template<typename Policy>
	KOKKOS_FUNCTION void set( const Policy& pol ){
		/* If a Kokkos::RangePolicy is not set to the matching ExecutionSpace,
		 * that shouldn't be an error, because it must be explicitly defined
		 * when creating a ParallelRange. */
		// if constexpr ( is_RangePolicy_v<Policy> ){
		// 	static_assert( target == PolicyHelper<Policy>::target );
		// }
		if constexpr (isHost){
			m_rng = ompSegment(pol);
			if constexpr ( is_RangePolicy_v<Policy> ){
				this->setChunks( pol.chunk_size() );
			} else {
				this->setChunks(chunk::defaultSize);
			}
			#ifdef KOKKIDIO_DEBUG_OUTPUT
			auto irng { toIndexRange(pol) };
			printdl(
				"Created ParallelRange from [%i, %i) (n=%i)\n"
				"\t-> rng: [%i, %i) (n=%i)\n"
				"\t-> chk: size=%i, n=%i\n"
				,  irng.start(),  irng.end(),  irng.size()
				, m_rng.start(), m_rng.end(), m_rng.size()
				, m_chunks.size, m_chunks.num
			);
			#endif
		} else {
			static_assert( std::is_integral_v<Policy> );
			m_rng = static_cast<int>(pol);
		}
	}

public:
	KOKKOS_FUNCTION auto chunkInfo() const -> const ChunkInfo<target>& {
		return m_chunks;
	}

	KOKKOS_FUNCTION inline constexpr Index
	chunkSize() const {
		if constexpr (isHost){
			return m_chunks.size;
		} else {
			return 1;
		}
	}

	KOKKOS_FUNCTION inline constexpr Index
	nChunks() const {
		if constexpr (isHost){
			return m_chunks.num;
		} else {
			return 1;
		}
	}


	KOKKOS_FUNCTION
	void setChunks(Index chunkSizeMax = chunk::defaultSize){
		this->setChunkSize(chunkSizeMax);
		// this->setNChunks();
	}

protected:
	KOKKOS_FUNCTION
	void setChunkSize( [[maybe_unused]] Index chunkSizeMax ){
		if constexpr (isHost){
			m_chunks.size = std::min( chunkSizeMax, this->get().size() );
		}
	}

	// KOKKOS_FUNCTION
	// void setNChunks(){
	// 	if constexpr (isHost){
	// 		const auto& r { this->get().values };
	// 		const Index& chunkSize { this->chunkSize() };
	// 		Index& nChunks {m_chunks.num};
	// 		nChunks = 0;
	// 		if ( chunkSize > 0){
	// 			nChunks  =  r.size / chunkSize;
	// 			nChunks += (r.size % chunkSize) > 0 ? 1 : 0;
	// 		}
	// 	}
	// }

public:
	// KOKKOS_FUNCTION
	// ChunkType make_chunk(Index i) const {
	// 	using T = Target;
	// 	if constexpr (isHost){
	// 		const MemberType& cols { this->get() };
	// 		const ChunkInfo<T::host>& chunks { this->chunkInfo() };
	// 		Index
	// 			chunkStart   { cols.start() + i * chunks.size },
	// 			colsUntilEnd { cols.end() - chunkStart },
	// 			chunkSizeCur { std::min(chunks.size, colsUntilEnd) };
	// 		return { IndexRange<Index>{chunkStart, chunkSizeCur} };
	// 	} else {
	// 		return {};
	// 	}
	// }

	KOKKOS_FUNCTION
	constexpr Index chunkSize([[maybe_unused]] Index chunkStart) const {
		if constexpr (isHost){
			return std::min(
				this->get().end() - chunkStart,
				this->chunkSize()
			);
		} else {
			return 1;
		}
	}

	KOKKOS_FUNCTION
	ChunkType make_chunk_s(Index chunkStart, Index chunkSizeCur) const {
		using T = Target;
		if constexpr (isHost){
			return { IndexRange<Index>{chunkStart, chunkSizeCur} };
		} else {
			return {};
		}
	}

	template<typename Func>
	KOKKOS_FUNCTION 
	KOKKIDIO_INLINE 
	void for_each( Func&& func ) const {
		if constexpr (isDevice){
			func(m_rng);
		} else {
			for ( int i=m_rng.start(); i<m_rng.end(); ++i ){
				func(i);
			}
		}
	}

	template<typename Func>
	KOKKOS_FUNCTION 
	KOKKIDIO_INLINE 
	void for_each_chunk(Func&& func) const {
		static_assert( std::is_invocable_v<Func, ChunkType> );

		if constexpr (isHost){
			#ifdef KOKKIDIO_OPENMP
			assert( omp_get_max_threads() == 1 || omp_in_parallel() );
			#endif

			// for (Index i=0; i<this->nChunks(); ++i){
			// 	func( this->make_chunk(i) );
			// }

			Index chunkStart {this->get().start()}, chunkSize;
			while ( chunkStart < this->get().end() ){
				chunkSize = this->chunkSize(chunkStart);
				func( this->make_chunk_s(chunkStart, chunkSize) );
				chunkStart += chunkSize;
			}
		} else {
			func( ChunkType{m_rng} );
		}
	}
};



/* error: __host__ __device__ extended lambdas cannot be generic lambdas */
#ifdef THIS_DOES_NOT_WORK

template<Target target = Target::host, int dim, typename Func>
void rangedLambda(
	int maxIdx,
	Func&& func
){
	Kokkos::parallel_for(maxIdx, KOKKOS_LAMBDA(int idx){
		auto rng = ParallelRange<target>( maxIdx, idx );
		func(rng);
	});
}

#endif


template<Target target, typename... Args>
auto range_tuple(const EigenRange<target>& rng, Args&&... args) {
	return std::make_tuple( rng( std::forward<Args>(args) ) ... );
}

#ifdef KOKKIDIO_USE_CUDAHIP
#include "Kokkidio/ParallelRange_cudahip.hpp"
#define apply_range(FUNC_NAME, RANGE, ...) \
FUNC_NAME( KOKKIDIO_APPLY_FUNC(RANGE, __VA_ARGS__) )

#else
#define apply_range(FUNC_NAME, RANGE, ...) \
std::apply( \
	[]KOKKOS_FUNCTION(auto&& ... args){ \
		FUNC_NAME( std::forward<decltype(args)>(args) ... ); \
	}, Kokkidio::range_tuple(RANGE, __VA_ARGS__) \
)
#endif

} // namespace Kokkidio

#endif
