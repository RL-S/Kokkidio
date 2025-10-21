#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "friction.hpp"

#include "testMacros.hpp"

namespace Kokkidio
{


KOKKIDIO_FUNC_WRAPPER(fric_unif, unif::friction)
KOKKIDIO_FUNC_WRAPPER(fric_cpu , cpu::friction)
KOKKIDIO_FUNC_WRAPPER(fric_gpu , gpu::friction)


// void runFric(int b.nCols, int b.nRuns, const std::string& b.target, bool b.gnuplot){
void runFric(const BenchOpts& b){
	if ( !b.gnuplot ){
		std::cout << "Running friction benchmark...\n";
	}
	scalar
		dVal {2},
		vx   {3},
		vy   {4},
		nVal {0.025};
	Array2s vCol {vx, vy};
	Array3s flux { dVal, 1, 2 };

	ArrayXXs
		flux_out { 3, b.nCols },
		flux_in  { 3, b.nCols },
		d { 1, b.nCols },
		v { 2, b.nCols },
		n { 1, b.nCols };
	n = nVal;
	d = dVal;
	v.colwise() = vCol;
	flux_in.colwise() = flux;
	flux_out = 0;

	auto pass = [&](){
		// return true;
		Array2s correctFlux {
			0.9119160834218206,
			1.865452694025437
		};
		bool isCorrect { flux_out.bottomRows(2).isApprox(
			correctFlux.replicate(1, b.nCols)
		) };
		if (!isCorrect){
			std::cerr.precision(16);
			std::cerr << "flux_out:\n";
			if (flux_out.size() < 30){
				std::cerr << flux_out;
			} else {
				std::cerr
					<< flux_out.leftCols(3)
					<< "\n...\n"
					<< flux_out.rightCols(3)
				;
			}
		}
		return isCorrect;
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

	// using T = Target;
	using K = fric::Kernel;
	using uK = unif::Kernel;
	/* Run on GPU */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		if (b.group != "unified"){
			setNat();
			runAndTime<fric_gpu, Target::device, K
				, K::cstyle // first one is for warmup
				, K::cstyle
				, K::eigen_colwise_fullbuf
			>(
				opts, pass,
				flux_out, flux_in, d, v, n, b.nRuns
			);
		}

		if (b.group != "native"){
			setUni();
			runAndTime<fric_unif, Target::device, uK
				, uK::kokkidio_range_chunkbuf // warmup is skipped
				// #ifndef KOKKIDIO_USE_SYCL
				, uK::cstyle
				// #endif
				, uK::kokkidio_index_fullbuf
				, uK::kokkidio_index_stackbuf
				, uK::kokkidio_range_fullbuf
				, uK::kokkidio_range_chunkbuf
				KRUN_IF_ALL(
				, uK::context_ranged
				)
			>(
				opts, pass,
				flux_out, flux_in, d, v, n, b.nRuns
			);
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		if (b.group != "unified"){
			setNat();
			runAndTime<fric_cpu, Target::host, K
				, K::cstyle // first one is for warmup
				, K::cstyle
				, K::eigen_ranged_fullbuf
			>(
				opts, pass,
				flux_out, flux_in, d, v, n, b.nRuns
			);
		}

		if (b.group != "native"){
			setUni();
			runAndTime<fric_unif, Target::host, uK
				, uK::kokkidio_range_chunkbuf // warmup is skipped
				, uK::cstyle
				, uK::kokkidio_index_fullbuf  // painfully slow
				, uK::kokkidio_index_stackbuf // painfully slow
				, uK::kokkidio_range_fullbuf
				, uK::kokkidio_range_chunkbuf
				KRUN_IF_ALL(
				, uK::context_ranged
				)
			>(
				opts, pass,
				flux_out, flux_in, d, v, n, b.nRuns
			);
		}
	}


	if (!b.gnuplot){
		std::cout << "Friction: Finished runs.\n\n";
	}
}


} // namespace Kokkidio

int main(int argc, char ** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	namespace K = Kokkidio;
	K::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	if ( !K::checkImpl<
		K::unif::Kernel, 
		K::fric::Kernel>(b) 
	){
		return 1;
	}
	K::runFric(b);
	
	return 0;
}
