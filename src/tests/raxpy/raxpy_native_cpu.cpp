#include "raxpy.hpp"

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

template<>
void raxpy<host, K::cstyle_seq>(KOKKIDIO_RAXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		scalar* zptr { z.data() };
		const scalar
			* xptr { x.data() },
			* yptr { y.data() };

		for (int i = 0; i<z.rows(); ++i){
			raxpy_sum( zptr[i], a, xptr[i], yptr[i] );
		}
	}
}

template<>
void raxpy<host, K::cstyle_par>(KOKKIDIO_RAXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		scalar* zptr { z.data() };
		const scalar
			* xptr { x.data() },
			* yptr { y.data() };

		KOKKIDIO_OMP_PRAGMA(parallel for)
		for (int i = 0; i<z.rows(); ++i){
			raxpy_sum( zptr[i], a, xptr[i], yptr[i] );
		}
	}
}

template<>
void raxpy<host, K::eigen_seq_nochunks>(KOKKIDIO_RAXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		raxpy_sum(z, a, x, y);
	}
}

template<>
void raxpy<host, K::eigen_seq>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min(z.rows(), Kokkidio::chunk::defaultSize) };
	// ArrayXs bufm { chunksizeMax };
	for (volatile int run = 0; run < nRuns; ++run){
		Index start {0}, chunksize;
		while (start < z.rows() ){
			chunksize = std::min( z.rows() - start, chunksizeMax);
			auto chunk = [&](auto& obj){
				return obj.segment(start, chunksize);
			};
			// auto buf { bufm.head(chunksize) };
			raxpy_sum(
				chunk(z), a,
				chunk(x),
				chunk(y)
			);
			// chunk(z) = buf;
			start += chunksize;
		}
	}
}

template<>
void raxpy<host, K::eigen_par_buf>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min( ompSegmentMaxSize( z.rows() ), 200) };
	ArrayXXs bufs ( chunksizeMax, omp_get_max_threads() );
	// printf("bufs size: %ix%i\n", bufs.rows(), bufs.cols() );

	for (volatile int run = 0; run < nRuns; ++run){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto threadseg = ompSegment( z.rows() );
			Index start { threadseg.start() }, chunksize;
			while (start < threadseg.end() ){
				chunksize = std::min( threadseg.end() - start, chunksizeMax);
				// printf(
				// 	"Thread #%i"
				// 	", segment [%i,%i) (n=%i)"
				// 	", chunk start %i, chunk size %i"
				// 	"\n"
				// 	, omp_get_thread_num()
				// 	, threadseg.start(), threadseg.end(), threadseg.size()
				// 	, start, chunksize
				// );
				auto chunk = [&](auto& obj){
					return obj.segment(start, chunksize);
				};
				auto buf { bufs.col( omp_get_thread_num() ).head(chunksize) };
				raxpy_sum( buf, a, chunk(x), chunk(y) );
				chunk(z) = buf;
				start += chunksize;
			}
		}
	}
}

template<>
void raxpy<host, K::eigen_par>(KOKKIDIO_RAXPY_ARGS){

	Index chunksizeMax { std::min( ompSegmentMaxSize( z.rows() ), 200) };

	for (volatile int run = 0; run < nRuns; ++run){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto threadseg = ompSegment( z.rows() );
			Index start { threadseg.start() }, chunksize;
			while (start < threadseg.end() ){
				chunksize = std::min( threadseg.end() - start, chunksizeMax);
				// printf(
				// 	"Thread #%i"
				// 	", segment [%i,%i) (n=%i)"
				// 	", chunk start %i, chunk size %i"
				// 	"\n"
				// 	, omp_get_thread_num()
				// 	, threadseg.start(), threadseg.end(), threadseg.size()
				// 	, start, chunksize
				// );
				auto chunk = [&](auto& obj){
					return obj.segment(start, chunksize);
				};
				raxpy_sum(
					chunk(z), a,
					chunk(x),
					chunk(y)
				);
				start += chunksize;
			}
		}
	}
}

} // namespace Kokkidio::cpu
