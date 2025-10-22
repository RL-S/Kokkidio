#include <Kokkidio.hpp>

namespace Kokkidio
{

#define KOKKIDIO_RAXPY_ARGS \
	ArrayXs& z, scalar a, const ArrayXs& x, const ArrayXs& y, \
	Index nRuns, [[maybe_unused]] Index chunksize


template<typename Z, typename XY>
KOKKOS_FUNCTION
KOKKIDIO_INLINE
void raxpy_sum(Z&& z, scalar a, const XY& x, const XY& y, int nIter){
	z = 0;
	for (Index i{0}; i<nIter; ++i){
		auto sign = static_cast<scalar>( -2 * (i % 2) + 1 );
		z += sign * a * x + sign * y;
		// z += a * x + y;
	}
}


namespace unif
{

enum class Kernel {
	cstyle,
	kokkos,
	kokkos_writeonce,
	kokkidio_index,
	kokkidio_range,
	kokkidio_range_writebuf,
	kokkidio_range_nobuf,
	kokkidio_range_accbuf,
};

template<Target, Kernel>
void raxpy(KOKKIDIO_RAXPY_ARGS);

} // namespace unif



namespace cpu
{

enum class Kernel {
	cstyle_seq,
	cstyle_par,
	eigen_seq,
	eigen_seq_nochunks,
	eigen_par,
	eigen_par_buf,
};

template<Target, Kernel>
void raxpy(KOKKIDIO_RAXPY_ARGS);

} // namespace cpu



namespace gpu
{

enum class Kernel {
	cstyle,
};

template<Target, Kernel>
void raxpy(KOKKIDIO_RAXPY_ARGS);

} // namespace gpu



} // namespace Kokkidio
