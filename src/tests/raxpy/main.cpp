#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "raxpy.hpp"

#include "testMacros.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(raxpy_unif, unif::raxpy)
KOKKIDIO_FUNC_WRAPPER(raxpy_cpu ,  cpu::raxpy)
KOKKIDIO_FUNC_WRAPPER(raxpy_gpu ,  gpu::raxpy)

void run_raxpy(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running repeat saxpy/daxpy benchmark...\n";
	}

	scalar a, z_correct;

	ArrayXs x ( std::max(b.nRows, b.nCols) ), y, z;
	y.resizeLike(x);
	z.resizeLike(x);

	Array3s randVals;
	randVals.setRandom();
	a = randVals[0];
	x = randVals[1];
	y = randVals[2];

	z_correct = a * x[0] + y[0];

	auto pass = [&](){
		/* mapping to the results so that we can check that they're all equal */
		bool same { z.isApproxToConstant(z_correct, epsilon) };
		if ( !same ){
			std::cerr.precision(16);
			std::cerr << "z:\n";
			if (z.size() < 30){
				std::cerr << z;
			} else {
				std::cerr
					<< z.head(3)
					<< "\n...\n"
					<< z.tail(3)
				;
			}
		#ifndef NDEBUG
		} else {
			z = 0; // reset z, so that leaving results untouched is not a pass
		#endif
		}
		return same;
	};

	RunOpts opts;
	opts.useGnuplot = b.gnuplot;
	auto setNat = [&](){
		opts.groupComment = "native";
		opts.skipWarmup = false;
	};
	auto setUni = [&](){
		opts.groupComment = "unified";
		opts.skipWarmup = true;
	};

	using T = Target;
	using uK = unif::Kernel;
	/* Run on GPU */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		setNat();
		using gK = gpu::Kernel;
		runAndTime<raxpy_gpu, T::device, gK
			, gK::cstyle // first one is for warmup
			, gK::cstyle
		>( opts, pass, z, a, x, y, b.nRuns );

		setUni();
		runAndTime<raxpy_unif, T::device, uK
			// , uK::cstyle // warmup is skipped
			, uK::cstyle
			// KRUN_IF_ALL(
			, uK::kokkos
			// )
			, uK::kokkos_writeonce
			, uK::kokkidio_index
			, uK::kokkidio_range
			, uK::kokkidio_range_writebuf
			, uK::kokkidio_range_nobuf
			, uK::kokkidio_range_accbuf
		>( opts, pass, z, a, x, y, b.nRuns );
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && (z.size() <= 1000 * 1000 * 1000 || b.nRuns <= 500) ){
		setNat();
		using cK = cpu::Kernel;
		runAndTime<raxpy_cpu, T::host, cK
			, cK::cstyle_par // first one is for warmup
			// , cK::cstyle_seq
			, cK::cstyle_par
			// , cK::eigen_seq_nochunks
			// , cK::eigen_seq
			, cK::eigen_par_buf
			, cK::eigen_par
		>( opts, pass, z, a, x, y, b.nRuns );

		setUni();
		runAndTime<raxpy_unif, T::host, uK
			// , uK::cstyle // warmup is skipped
			, uK::cstyle
			// KRUN_IF_ALL(
			, uK::kokkos
			// )
			, uK::kokkidio_index
			, uK::kokkidio_range
			, uK::kokkidio_range_writebuf
			, uK::kokkidio_range_nobuf
			, uK::kokkidio_range_accbuf
		>( opts, pass, z, a, x, y, b.nRuns );
	}

	if (!b.gnuplot){
		std::cout
			<< "raxpy result:\n" << z_correct << '\n'
			<< "raxpy: Finished runs.\n\n";
	}
}

} // namespace Kokkidio

int main(int argc, char** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	Kokkidio::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	Kokkidio::run_raxpy(b);

	return 0;
}
