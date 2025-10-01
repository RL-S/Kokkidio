#ifndef KOKKIDIO_PARALLELRANGE_BUFFER_HPP
#define KOKKIDIO_PARALLELRANGE_BUFFER_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Core.hpp instead."
#endif

#include "Kokkidio/ViewMap.hpp"
#include "Kokkidio/IndexRange.hpp"

#include <memory>
#include <vector>

namespace Kokkidio
{


template<Target target>
using Chunk = EigenRange<target>;

template<Target target>
struct ChunkInfo {};

template<>
struct ChunkInfo<Target::host> {
	Index size, num;
};
static_assert( std::is_trivially_copyable_v<ChunkInfo<Target::device>> );


namespace chunk
{

KOKKIDIO_CONSTANT(constexpr int defaultSize {
	std::is_same_v<scalar, float> ? 200 : 100
};)

template<Target _target, typename _ColType>
struct BufferTypeHelper {
	using T = Target;
	static constexpr T target {_target};
	using ColType = _ColType;

	static_assert( std::is_base_of_v<Eigen::DenseBase<ColType>, ColType> );
	// static_assert( ColType::ColsAtCompileTime == 1 );

	static constexpr int RowsAtCompileTime { ColType::RowsAtCompileTime };
	static_assert( RowsAtCompileTime != Eigen::Dynamic );
	static constexpr int ColsAtCompileTime {
		target == T::host ? Eigen::Dynamic : 1
	};

	using Scalar = typename ColType::Scalar;

	static_assert( 
		is_eigen_matrix_v<ColType> ||
		is_eigen_array_v <ColType>
	);
	using PlainObjectType = std::conditional_t<is_eigen_matrix_v<ColType>,
		Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>,
		Eigen::Array <Scalar, RowsAtCompileTime, ColsAtCompileTime>
	>;

	using Type = std::conditional_t<
		target == T::host,
		Eigen::Map<PlainObjectType>,
		ColType
	>;
};

template<Target target, typename ColType>
using LoopType = typename BufferTypeHelper<target, ColType>::Type;

/* if you need to go backwards from the object type to the ColType,
 * i.e. the input type to the BufferType. */
template<typename _DenseType>
struct ColTypeHelper {
	using DenseType = _DenseType;

	static_assert( std::is_base_of_v<Eigen::DenseBase<DenseType>, DenseType> );

	static constexpr int RowsAtCompileTime { DenseType::RowsAtCompileTime };
	static_assert( RowsAtCompileTime != Eigen::Dynamic );

	using Scalar = typename DenseType::Scalar;
	static_assert(
		is_eigen_matrix_v<DenseType> ||
		is_eigen_array_v <DenseType>
	);
	using Type = std::conditional_t<is_eigen_matrix_v<DenseType>,
		Eigen::Matrix<Scalar, RowsAtCompileTime, 1>,
		Eigen::Array <Scalar, RowsAtCompileTime, 1>
	>;
};

template<typename DenseType>
using ColType = typename ColTypeHelper<DenseType>::Type;


template<typename _ColType>
struct DeviceBuffer {
	using ColType  = _ColType;
	using LoopType = chunk::LoopType<Target::device, ColType>;
};


template<typename _ColType>
class HostBuffer {
public:
	using ColType = _ColType;
	using Scalar = typename ColType::Scalar;
	static constexpr auto target {Target::host};
	using MemorySpace = Kokkidio::MemorySpace<target>;
	using LoopType    = chunk::LoopType<target, ColType>;

	using DataType = Scalar***;
	using ViewType = Kokkos::View<DataType, Kokkos::LayoutLeft, MemorySpace>;

	static constexpr Index RowsAtCompileTime {ColType::RowsAtCompileTime};
	static_assert(RowsAtCompileTime != Eigen::Dynamic);

protected:
	ViewType m_view;

	template<typename Policy>
	void set(const Policy& pol, Index chunkSizeMax){
		auto rng { toIndexRange(pol) };
		if constexpr (is_RangePolicy_v<Policy>){
			chunkSizeMax = pol.chunk_size();
		}
		std::size_t
			rows {RowsAtCompileTime},
			cols {static_cast<std::size_t>( std::min(rng.size(), chunkSizeMax) )};
		this->m_view = ViewType{
			Kokkos::view_alloc(MemorySpace{}, Kokkos::WithoutInitializing, ""),
			rows,
			cols,
			this->maxThreads()
		};
		printd(
			"HostBuffer::set: range [%i, %i), chunkSizeMax = %i\n"
			"\t(%p) View extents: (%lu, %lu, %lu)\n"
			, rng.begin(), rng.end(), chunkSizeMax
			, (void*) ( this->m_view.data() )
			, this->m_view.extent(0)
			, this->m_view.extent(1)
			, this->m_view.extent(2)
		);
	}


public:
	static std::size_t maxThreads(){
		#ifdef _OPENMP
		return omp_get_max_threads();
		#else
		return 1;
		#endif
	}

	HostBuffer() = default;

	/**
	 * @brief Creates a chunk buffers for each thread.
	 * Uses omp_get_max_threads as the number of threads,
	 * the number of rows of @a ColType, i.e. @a RowsAtCompileTime,
	 * and @a chunkSizeMax as the number of columns.
	 * If a Kokkos::RangePolicy is used as the parameter,
	 * its @a chunk_size is ignored.
	 * 
	 * @tparam Policy 
	 * @param pol can be an integral type, a Kokkos::RangePolicy, or an IndexRange.
	 * @param chunkSizeMax 
	 */
	template<typename Policy>
	HostBuffer(const Policy& pol, Index chunkSizeMax){
		set( pol, chunkSizeMax );
	}

	/**
	 * @brief Creates a chunk buffers for each thread.
	 * Uses omp_get_max_threads as the number of threads,
	 * and the number of rows of @a ColType, i.e. @a RowsAtCompileTime.
	 * The number of columns is set to the chunk size, 
	 * which is set to Kokkos::RangePolicy::chunk_size 
	 * if such an argument is used, and to chunk::defaultSize otherwise.
	 * 
	 * @tparam Policy 
	 * @param pol 
	 */
	template<typename Policy>
	HostBuffer(const Policy& pol){
		if constexpr (is_RangePolicy_v<Policy>){
			set( pol, pol.chunk_size() );
		} else
		{
			set( pol, chunk::defaultSize );
		}
	}

public:
	LoopType get( const Chunk<Target::host>& chunk ) const {
		#ifdef _OPENMP
		assert( ( omp_get_max_threads() == 1 || omp_in_parallel() ) );
		int threadNo { omp_get_thread_num() };
		#else
		int threadNo {0};
		#endif

		// #ifndef __CUDACC__
		printdl(
			"(%p) HostBuffer::get: Thread #%i, mapping to view element...\n"
			"\tView address range: %p - %p\n"
			"\tView extents: (%lu, %lu, %lu)\n"
			"\tMap size: (%i, %i)\n"
			, (void*) &( this->m_view(0, 0, threadNo) )
			, threadNo
			, (void*) ( this->m_view.data() )
			, (void*) ( this->m_view.data() + this->m_view.size() )
			, this->m_view.extent(0)
			, this->m_view.extent(1)
			, this->m_view.extent(2)
			, RowsAtCompileTime, chunk.get().size()
		);
		// #endif

		return {
			&( this->m_view(0, 0, threadNo) ),
			RowsAtCompileTime,
			chunk.get().size()
		};
	}
};


template<typename ColType, Target target>
struct Buffer {
	using Type = std::conditional_t<target == Target::host,
		HostBuffer<ColType>,
		DeviceBuffer<ColType>
	>;
};

} // namespace chunk


template<typename ColType, Target target>
using ChunkBuffer = typename chunk::Buffer<ColType, target>::Type;




/**
 * @brief Creates a target-specific buffer, whose columns behave like @a ColType.
 * If @a target is Target::host, then a chunk buffer is created for each thread.
 * In that case, it uses
 * omp_get_max_threads as the number of threads,
 * @a ColType::RowsAtCompileTime as the number of rows,
 * and @a chunkSizeMax as the number of columns.
 * If a Kokkos::RangePolicy is used as the parameter,
 * its @a chunk_size is ignored.
 * 
 * If @a target is Target::device, then the return type only contains the
 * type of @a ColType, and no action is performed at runtime.
 * 
 * This function must be called outside of a call to parallel_for.
 * To access the buffer, use getBuffer inside a parallel_for lambda.
 * 
 * @tparam ColType 
 * @tparam target 
 * @tparam Policy 
 * @param pol 
 * @param chunkSizeMax 
 * @return ChunkBuffer<ColType, target> 
 */
template<typename ColType, Target target, typename Policy>
ChunkBuffer<ColType, target>
makeBuffer( const Policy& pol, Index chunkSizeMax ){
	if constexpr (target == Target::host){
		return chunk::HostBuffer<ColType>(pol, chunkSizeMax);
	} else {
		return {};
	}
}



/**
 * @brief Creates a target-specific buffer, whose columns behave like @a ColType.
 * If @a target is Target::host, then a chunk buffer is created for each thread.
 * In that case, it uses
 * omp_get_max_threads as the number of threads,
 * @a ColType::RowsAtCompileTime as the number of rows,
 * and @a chunkSizeMax as the number of columns.
 * If a Kokkos::RangePolicy is used as the parameter,
 * its @a chunk_size is used, and chunk::defaultSize otherwise.
 * 
 * If @a target is Target::device, then the return type only contains the
 * type of @a ColType, and no action is performed at runtime.
 * 
 * This function must be called outside of a call to parallel_for.
 * To access the buffer, use getBuffer inside a parallel_for lambda.
 * 
 * @tparam ColType 
 * @tparam target 
 * @tparam Policy 
 * @param pol 
 * @return ChunkBuffer<ColType, target> 
 */
template<typename ColType, Target target, typename Policy>
ChunkBuffer<ColType, target>
makeBuffer( const Policy& pol ){
	return makeBuffer<ColType, target>( pol, [&](){
		if constexpr (is_RangePolicy_v<Policy>){
			return pol.chunk_size();
		} else {
			return chunk::defaultSize;
		}
	}() );
}


/* on host */
template<typename ColType>
typename ChunkBuffer<ColType, Target::host>::LoopType
getBuffer(
	const chunk::HostBuffer<ColType>& chunkBuf,
	const Chunk<Target::host>& chunk
){
	return chunkBuf.get(chunk);
}

/* on device */
template<typename ColType>
KOKKOS_FUNCTION
typename ChunkBuffer<ColType, Target::device>::LoopType
getBuffer(
	const chunk::DeviceBuffer<ColType>&,
	const Chunk<Target::device>&
){
	return {};
}



} // namespace Kokkidio


#endif