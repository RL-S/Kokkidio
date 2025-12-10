#include "raxpy.hpp"

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

template<>
void raxpy<host, K::cstyle_seq>(KOKKIDIO_RAXPY_ARGS){

	scalar* zptr { z.data() };
	const scalar
		* xptr { x.data() },
		* yptr { y.data() };

	for (int i = 0; i<z.rows(); ++i){
		scalar z_local;
		raxpy_sum(
			z_local, a,
			xptr[i],
			yptr[i],
			nRuns
		);
		zptr[i] = z_local;
	}
}

template<>
void raxpy<host, K::cstyle_par>(KOKKIDIO_RAXPY_ARGS){

	scalar* zptr { z.data() };
	const scalar
		* xptr { x.data() },
		* yptr { y.data() };

	KOKKIDIO_OMP_PRAGMA(parallel for)
	for (int i = 0; i<z.rows(); ++i){
		scalar z_local;
		raxpy_sum(
			z_local, a,
			xptr[i], 
			yptr[i],
			nRuns
		);
		zptr[i] = z_local;
	}
}

template<>
void raxpy<host, K::eigen_seq_nochunks>(KOKKIDIO_RAXPY_ARGS){

	raxpy_sum(z, a, x, y, nRuns);
}

template<>
void raxpy<host, K::eigen_seq>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min(z.rows(), chunksize) };
	// ArrayXs bufm { chunksizeMax };
	Index start {0}, chunksizeCur;
	while (start < z.rows() ){
		chunksizeCur = std::min( z.rows() - start, chunksizeMax);
		auto chunk = [&](auto& obj){
			return obj.segment(start, chunksizeCur);
		};
		// auto buf { bufm.head(chunksize) };
		raxpy_sum(
			chunk(z), a,
			chunk(x),
			chunk(y),
			nRuns
		);
		// chunk(z) = buf;
		start += chunksizeCur;
	}
}

template<>
void raxpy<host, K::eigen_par_buf>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min( ompSegmentMaxSize( z.rows() ), chunksize) };
	ArrayXXs bufs ( chunksizeMax, omp_get_max_threads() );
	// printf("bufs size: %ix%i\n", bufs.rows(), bufs.cols() );

	KOKKIDIO_OMP_PRAGMA(parallel)
	{
		auto threadseg = ompSegment( z.rows() );
		Index start { threadseg.start() }, chunksizeCur;
		while (start < threadseg.end() ){
			chunksizeCur = std::min( threadseg.end() - start, chunksizeMax);
			// printf(
			// 	"Thread #%i"
			// 	", segment [%i,%i) (n=%i)"
			// 	", chunk start %i, chunk size %i"
			// 	"\n"
			// 	, omp_get_thread_num()
			// 	, threadseg.start(), threadseg.end(), threadseg.size()
			// 	, start, chunksizeCur
			// );
			auto chunk = [&](auto& obj){
				return obj.segment(start, chunksizeCur);
			};
			auto buf { bufs.col( omp_get_thread_num() ).head(chunksizeCur) };
			raxpy_sum(
				buf, a,
				chunk(x),
				chunk(y),
				nRuns
			);
			chunk(z) = buf;
			start += chunksizeCur;
		}
	}
}

template<>
void raxpy<host, K::eigen_par>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min( ompSegmentMaxSize( z.rows() ), chunksize) };

	KOKKIDIO_OMP_PRAGMA(parallel)
	{
		auto threadseg = ompSegment( z.rows() );
		Index start { threadseg.start() }, chunksizeCur;
		while (start < threadseg.end() ){
			chunksizeCur = std::min( threadseg.end() - start, chunksizeMax);
			// printf(
			// 	"Thread #%i"
			// 	", segment [%i,%i) (n=%i)"
			// 	", chunk start %i, chunk size %i"
			// 	"\n"
			// 	, omp_get_thread_num()
			// 	, threadseg.start(), threadseg.end(), threadseg.size()
			// 	, start, chunksizeCur
			// );
			auto chunk = [&](auto& obj){
				return obj.segment(start, chunksizeCur);
			};
			raxpy_sum(
				chunk(z), a,
				chunk(x),
				chunk(y),
				nRuns
			);
			start += chunksizeCur;
		}
	}
}

} // namespace Kokkidio::cpu
