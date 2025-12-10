#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "dotProduct.hpp"

#include "testMacros.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(dot_gpu , gpu ::dotProduct)
KOKKIDIO_FUNC_WRAPPER(dot_cpu , cpu ::dotProduct)
KOKKIDIO_FUNC_WRAPPER(dot_unif, unif::dotProduct)


void runDot(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running dot product benchmark...\n";
	}
	#ifdef TEST_INITIALIZATION
	/* Test whether everything is okay */
	MatrixXs m1 (b.nRows, b.nCols), m2 (b.nRows, b.nCols);
	m1.array() = 1;
	m2 = m1;
	m2.array().row(0) = -1;
	#else
	/* Initialize the matrices with random values */
	MatrixXs m1 = MatrixXs::Random(b.nRows, b.nCols);
	MatrixXs m2 = MatrixXs::Random(b.nRows, b.nCols);
	#endif

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
			#ifndef KOKKIDIO_USE_SYCL
			setNat();
			using gK = gpu::Kernel;
			runAndTime<dot_gpu, T::device, gK
				, gK::cstyle_blockbuf // first one is for warmup
				, gK::cstyle_blockbuf
			>( opts, pass, m1, m2, b.nRuns );
			#else
			/* native cstyle dot product is implemented, but incredibly slow:
			* each iteration with just 4x10000 takes about a hundredth of a second,
			* so the test will never finish */
			runAndTime<dot_unif, T::device, uK
				, uK::cstyle // warmup only
			>( opts, pass, m1, m2, b.nRuns );
			#endif
		}

		if (b.group != "native"){
			setUni();
			runAndTime<dot_unif, T::device, uK
				, uK::kokkidio_range // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::cstyle_nobuf
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
				KRUN_IF_ALL(
				, uK::kokkidio_range_chunks
				, uK::kokkidio_range_trace
				, uK::kokkidio_range_for_each
				, uK::kokkidio_index_merged
				, uK::kokkidio_range_for_each_merged
				)
			>( opts, pass, m1, m2, b.nRuns );
		}
	}
	#endif

	/* Run on CPU */
	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		if (b.group != "unified"){
			setNat();
			using cK = cpu::Kernel;
			runAndTime<dot_cpu, T::host, cK
				, cK::eigen_par_arrProd // first one is for warmup
				, cK::cstyle_seq
				, cK::cstyle_par
				, cK::eigen_seq_colwise
				, cK::eigen_seq_arrProd
				, cK::eigen_par_colwise
				, cK::eigen_par_arrProd
			>( opts, pass, m1, m2, b.nRuns );
		}

		if (b.group != "native"){
			setUni();
			runAndTime<dot_unif, T::host, uK
				, uK::kokkidio_range // warmup is skipped
				, uK::cstyle
				KRUN_IF_ALL(
				, uK::cstyle_nobuf
				)
				, uK::kokkidio_index
				, uK::kokkidio_range
				KRUN_IF_ALL(
				, uK::kokkidio_range_chunks
				, uK::kokkidio_range_trace
				, uK::kokkidio_range_for_each
				, uK::kokkidio_index_merged
				, uK::kokkidio_range_for_each_merged
				)
			>( opts, pass, m1, m2, b.nRuns );
		}
	}

	if (!b.gnuplot){
		std::cout
			<< "DotProduct sums:\n" << resMap() << '\n'
			<< "DotProduct: Finished runs.\n\n";
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
	K::runDot(b);

	return 0;
}
