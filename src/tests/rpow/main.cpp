#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "rpow.hpp"

#include "testMacros.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(rpow_unif, unif::rpow)
KOKKIDIO_FUNC_WRAPPER(rpow_cpu ,  cpu::rpow)
KOKKIDIO_FUNC_WRAPPER(rpow_gpu ,  gpu::rpow)

void run_rpow(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running rational power (high OI) benchmark...\n";
	}

	scalar out_correct, in_all;

	ArrayXs in ( std::max(b.nRows, b.nCols) ), out;
	out.resizeLike(in);

	Array1s randVals;
	randVals.setRandom();
	in_all = randVals[0];
	in = in_all;

	out_correct = [&](){
		scalar sum {0};
		for (Index i{0}; i<rpow_ctrl::nIter; ++i){
			auto sign = static_cast<scalar>(-2 * (i % 2) + 1);
			sum += sign * pow( in_all, static_cast<scalar>(i) );
		}
		return sum;
	}();

	auto pass = [&](){
		/* mapping to the results so that we can check that they're all equal */
		bool same { out.isApproxToConstant(out_correct, epsilon) };
		if ( !same ){
			std::cerr.precision(16);
			std::cerr << "out:\n";
			if (out.size() < 30){
				std::cerr << out;
			} else {
				std::cerr
					<< out.head(3)
					<< "\n...\n"
					<< out.tail(3)
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
			runAndTime<rpow_gpu, T::device, gK
				, gK::cstyle // first one is for warmup
				, gK::cstyle
			>( opts, pass, out, in, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<rpow_unif, T::device, uK
				, uK::cstyle // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::kokkos
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, out, in, b.nRuns );
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && (out.size() <= 1000 * 1000 * 1000 || b.nRuns <= 500) ){
		if (b.group != "unified"){
			setNat();
			using cK = cpu::Kernel;
			runAndTime<rpow_cpu, T::host, cK
				, cK::cstyle_par // first one is for warmup
				, cK::cstyle_seq
				, cK::cstyle_par
				, cK::eigen_seq
				, cK::eigen_par
			>( opts, pass, out, in, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<rpow_unif, T::host, uK
				, uK::cstyle // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::kokkos
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, out, in, b.nRuns );
		}
	}

	if (!b.gnuplot){
		std::cout
			<< "rpow result:\n" << out_correct << '\n'
			<< "rpow: Finished runs.\n\n";
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
	K::run_rpow(b);

	return 0;
}
