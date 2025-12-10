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

	long size { b.nRows*b.nCols };
	
	long flopCpu, flopGpu;
	flopCpu = 4 * size * b.nRuns;
	flopGpu = 2 * (b.nRuns + 10) * size;

	// double tflopCpu, tflopGpu;
	// tflopCpu = static_cast<double>(flopCpu) / 1e12;
	// tflopGpu = static_cast<double>(flopGpu) / 1e12;

	scalar a, z_correct;

	ArrayXs x (size), y, z;
	y.resizeLike(x);
	z.resizeLike(x);

	// Kokkos::ChunkSize cs { static_cast<int>(b.nRows) };
	// auto pol = Kokkos::RangePolicy<ExecutionSpace<Target::host>>(0, 10, cs);
	// std::cout << "pol.chunk_size(): " << pol.chunk_size() << '\n';

	Array3s randVals;
	randVals.setRandom();
	a = randVals[0];
	x = randVals[1];
	y = randVals[2];

	// z_correct = a * x[0] + y[0];
	raxpy_sum(z_correct, a, x[0], y[0], b.nRuns % 2 ? 1 : 2);

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
			runAndTime<raxpy_gpu, T::device, gK
				, gK::cstyle // first one is for warmup
				, gK::cstyle
			>( opts, pass, z, a, x, y, b.nRuns, b.nRows );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<raxpy_unif, T::device, uK
				, uK::kokkos_writeonce // first one is for warmup
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
			>( opts, pass, z, a, x, y, b.nRuns, b.nRows );
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && (z.size() <= 1'000'000'000 || b.nRuns <= 500) ){
		if (b.group != "unified"){
			setNat();
			using cK = cpu::Kernel;
			runAndTime<raxpy_cpu, T::host, cK
				, cK::eigen_par // first one is for warmup
				// , cK::cstyle_seq
				, cK::cstyle_par
				// , cK::eigen_seq_nochunks
				// , cK::eigen_seq
				// , cK::eigen_par_buf
				, cK::eigen_par
			>( opts, pass, z, a, x, y, b.nRuns, b.nRows );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<raxpy_unif, T::host, uK
				, uK::kokkidio_range_accbuf // first one is for warmup
				, uK::cstyle
				// // KRUN_IF_ALL(
				// , uK::kokkos
				// // )
				, uK::kokkos_writeonce
				, uK::kokkidio_index
				// , uK::kokkidio_range
				// , uK::kokkidio_range_writebuf
				// , uK::kokkidio_range_nobuf
				, uK::kokkidio_range_accbuf
			>( opts, pass, z, a, x, y, b.nRuns, b.nRows );
		}
	}

	if (!b.gnuplot){
		std::cout
			<<   "raxpy result:\n" << z_correct
			<< "\nraxpy: Expecting n FLOP on CPU: " << flopCpu
			<< "\nraxpy: Expecting n FLOP on GPU: " << flopGpu
			<< "\nraxpy: Finished runs."
			<< "\n\n";
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
	K::run_raxpy(b);

	return 0;
}
