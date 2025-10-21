#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "norm.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(norm_gpu , gpu ::norm)
KOKKIDIO_FUNC_WRAPPER(norm_cpu , cpu ::norm)
KOKKIDIO_FUNC_WRAPPER(norm_unif, unif::norm)

/* computes the magnitude/norm of vectors and finds the largest one */
void runNorm(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running norm benchmark...\n";
	}
	/* Initialize a matrix with random values */
	MatrixXs mat = MatrixXs::Random(b.nRows, b.nCols);

	/* let's store the results of each run,
	 * so that we can compare them and report when they're not equal,
	 * rather than having to check that manually. */
	std::vector<scalar> results;
	results.reserve(16);

	/* mapping to the results so that we can check that they're all equal */
	auto resMap = [&]() -> ArrayXsMap {
		return { results.data(), static_cast<Index>( results.size() ) };
	};

	auto pass = [&](scalar result){
		results.push_back( result );
		/* mapping to the results so that we can check that they're all equal */
		bool same { resMap().isApproxToConstant(results[0], epsilon) };
		if ( !same ){
			std::cerr.precision(16);
			std::cerr << "Diverging results!\n" << resMap() << '\n';
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
			// #ifdef KOKKIDIO_USE_CUDAHIP
			setNat();
			using gK = gpu::Kernel;
			runAndTime<norm_gpu, T::device, gK
				, gK::cstyle_blockbuf // first one is for warmup
				, gK::cstyle_blockbuf
			>( opts, pass, mat, b.nRuns );
			// #endif
		}

		if (b.group != "native"){
			setUni();
			runAndTime<norm_unif, T::device, uK
				, uK::kokkidio_range // warmup is skipped
				, uK::cstyle
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, mat, b.nRuns );
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		if (b.group != "unified"){
			setNat();
			using cK = cpu::Kernel;
			runAndTime<norm_cpu, T::host, cK
				, cK::eigen_par // first one is for warmup
				, cK::cstyle_seq
				, cK::cstyle_par
				, cK::eigen_seq
				, cK::eigen_par
			>( opts, pass, mat, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<norm_unif, T::host, uK
				, uK::kokkidio_range // warmup is skipped
				, uK::cstyle
				, uK::kokkidio_index
				, uK::kokkidio_range
			>( opts, pass, mat, b.nRuns );
		}
	}

	if (!b.gnuplot){
		std::cout
			<< "Norm maxima:\n" << resMap() << '\n'
			<< "Norm: Finished runs.\n\n";
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
	K::runNorm(b);

	return 0;
}
