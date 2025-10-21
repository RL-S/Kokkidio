#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "axpy.hpp"

#include "testMacros.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(axpy_unif, unif::axpy)
KOKKIDIO_FUNC_WRAPPER(axpy_cpu ,  cpu::axpy)
KOKKIDIO_FUNC_WRAPPER(axpy_gpu ,  gpu::axpy)

constexpr auto axpyStr { std::is_same_v<scalar, float> ? "saxpy" : "daxpy" };

void run_axpy(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running " << axpyStr << " benchmark...\n";
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
		}
		return same;
	};

	RunOpts opts;
	opts.useGnuplot = b.gnuplot;
	opts.impl = b.impl;
	auto setNat = [&](){
		opts.groupComment = "native";
		opts.skipWarmup = b.skipWarmup;
	};
	auto setUni = [&](){
		opts.groupComment = "unified";
		opts.skipWarmup = true;
		if (b.group != "all" || b.impl != "all"){
			opts.skipWarmup = b.skipWarmup;
		}
	};

	using T = Target;
	using uK = unif::Kernel;
	/* Run on GPU */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		if (b.group != "unified"){
			setNat();
			using gK = gpu::Kernel;
			runAndTime<axpy_gpu, T::device, gK
				, gK::cstyle // first one is for warmup
				, gK::cstyle
			>( opts, pass, z, a, x, y, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<axpy_unif, T::device, uK
				, uK::cstyle // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::kokkos
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, z, a, x, y, b.nRuns );
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && (z.size() <= 1000 * 1000 * 1000 || b.nRuns <= 500) ){
		if (b.group != "unified"){
			setNat();
			using cK = cpu::Kernel;
			runAndTime<axpy_cpu, T::host, cK
				, cK::eigen_par // first one is for warmup
				, cK::cstyle_seq
				, cK::cstyle_par
				, cK::eigen_seq
				, cK::eigen_par
			>( opts, pass, z, a, x, y, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<axpy_unif, T::host, uK
				, uK::cstyle // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::kokkos
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, z, a, x, y, b.nRuns );
		}
	}

	if (!b.gnuplot){
		std::cout
			<< axpyStr << " result:\n" << z_correct << '\n'
			<< axpyStr << ": Finished runs.\n\n";
	}
}

} // namespace Kokkidio

int main(int argc, char** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	namespace K = Kokkidio;
	K::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	if ( !K::checkImpl<
		K::unif::Kernel, 
		K::gpu::Kernel, 
		K::cpu::Kernel>(b) 
	){
		return 1;
	}
	K::run_axpy(b);

	return 0;
}
