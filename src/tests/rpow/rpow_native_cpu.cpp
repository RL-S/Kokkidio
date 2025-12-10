#include "rpow.hpp"

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

template<>
void rpow<host, K::cstyle_seq>(KOKKIDIO_RPOW_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		      scalar* optr { out.data() };
		const scalar* iptr { in.data() };

		for (int i = 0; i<out.rows(); ++i){
			optr[i] = rpow_sum( iptr[i] );
		}
	}
}

template<>
void rpow<host, K::cstyle_par>(KOKKIDIO_RPOW_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		      scalar* optr { out.data() };
		const scalar* iptr { in.data() };

		KOKKIDIO_OMP_PRAGMA(parallel for)
		for (int i = 0; i<out.rows(); ++i){
			optr[i] = rpow_sum( iptr[i] );
		}
	}
}

template<>
void rpow<host, K::eigen_seq>(KOKKIDIO_RPOW_ARGS){

	Index chunksizeMax { std::min(out.size(), Kokkidio::chunk::defaultSize) };
	ArrayXs bufm { chunksizeMax };
	for (volatile int run = 0; run < nRuns; ++run){
		Index start {0}, chunksize;
		while (start < out.size() ){
			chunksize = std::min( out.size() - start, chunksizeMax);
			auto chunk = [&](auto& obj){
				return obj.segment(start, chunksize);
			};
			auto buf { bufm.tail(chunksize) };
			buf = 0;
			for (Index i{0}; i<rpow_ctrl::nIter; ++i){
				auto sign = static_cast<scalar>( -2 * (i % 2) + 1 );
				buf += sign * chunk(in).pow( static_cast<scalar>(i) );
			}
			chunk(out) = buf;
			start += chunksize;
		}
	}
}

template<>
void rpow<host, K::eigen_par>(KOKKIDIO_RPOW_ARGS){

	Index chunksizeMax { std::min( ompSegmentMaxSize( out.size() ), 200) };
	ArrayXXs bufs ( chunksizeMax, omp_get_max_threads() );
	// printf("bufs size: %ix%i\n", bufs.rows(), bufs.cols() );

	for (volatile int run = 0; run < nRuns; ++run){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto threadseg = ompSegment( out.size() );
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
				auto buf { bufs.col( omp_get_thread_num() ).tail(chunksize) };
				buf = 0;
				for (Index i{0}; i<rpow_ctrl::nIter; ++i){
					auto sign = static_cast<scalar>( -2 * (i % 2) + 1 );
					buf += sign * chunk(in).pow( static_cast<scalar>(i) );
				}
				chunk(out) = buf;
				start += chunksize;
			}
		}
	}
}

} // namespace Kokkidio::cpu
