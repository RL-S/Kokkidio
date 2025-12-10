#include <Kokkidio.hpp>

namespace Kokkidio
{

#define KOKKIDIO_RPOW_ARGS \
	ArrayXs& out, const ArrayXs& in, Index nRuns

namespace rpow_ctrl
{

KOKKIDIO_HOST_DEVICE_VAR(static constexpr Index nIter {100});

} // namespace rpow_ctrl


KOKKOS_FUNCTION
inline scalar rpow_sum(scalar base){
	scalar sum {0};
	for (Index i{0}; i<rpow_ctrl::nIter; ++i){
		auto sign = static_cast<scalar>( -2 * (i % 2) + 1 );
		sum += sign * pow( base, static_cast<scalar>(i) );
	}
	return sum;
}


namespace unif
{

enum class Kernel {
	cstyle,
	kokkos,
	kokkidio_index,
	kokkidio_range,
};

template<Target, Kernel>
void rpow(KOKKIDIO_RPOW_ARGS);

} // namespace unif



namespace cpu
{

enum class Kernel {
	cstyle_seq,
	cstyle_par,
	eigen_seq,
	eigen_par,
};

template<Target, Kernel>
void rpow(KOKKIDIO_RPOW_ARGS);

} // namespace cpu



namespace gpu
{

enum class Kernel {
	cstyle,
};

template<Target, Kernel>
void rpow(KOKKIDIO_RPOW_ARGS);

} // namespace gpu



} // namespace Kokkidio
