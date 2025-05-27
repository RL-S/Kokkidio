
#include "unifyBackends.hpp"
#include "redux.hpp"
#include "redux_cudahip_generic.hpp"

#include <magic_enum.hpp>


namespace Kokkidio::gpu
{


template<Target target, Reduction reduction>
scalar reduce(const ArrayXXs& values, int nRuns){

	static constexpr Index blockSize {1024};

	auto elementsToBlocks = [&](Index nElem, Index blocksize){
		/* in the generic reduction kernel, each block does twice the work */
		return (nElem + (blocksize*2) - 1) / (blocksize*2);
	};

	/* Every run must start with this value */
	Index
		nElementsInit {values.size()},
		nBlocksInit { elementsToBlocks(nElementsInit, blockSize) };

	/* We don't need very many device arrays this time.
	 * Just one input and one output array. */
	scalar
		*values_d,
		*red_d;

	/* the first reduction step is performed on a per-block-level,
	 * so we need as many elements as blocks for the output data */
	gpuAlloc(red_d, nBlocksInit);
	gpuAllocAndCopy(values_d, values);

	for (int run{0}; run<nRuns; ++run){
		/* Reset values at the beginning of each run */
		scalar* reduxVals {values_d};
		int
			nBlocks   {nBlocksInit},
			nElements {nElementsInit};

		[[maybe_unused]] int i{0};

		/* we may need several iterations to arrive at the reduction result */
		do {
			printd(
				"Iteration %i, %s: Reducing %i elements "
				"using %i block(s) of size %i.\n"
				, i++
				, std::string{magic_enum::enum_name(reduction)}.c_str()
				, nElements
				, nBlocks
				, blockSize
			);

			dim3 gridDim (nBlocks, 1, 1);
			dim3 blockDim(blockSize, 1, 1);
			using R = Reduction;
			if constexpr (reduction == R::sum){
				chLaunchKernel( redux::sum<blockSize>,
					gridDim, blockDim, 0, 0, // no streams specified
					reduxVals, red_d, nElements
				);
			} else
			if constexpr (reduction == R::prod){
				chLaunchKernel( redux::prod<blockSize>,
					gridDim, blockDim, 0, 0, // no streams specified
					reduxVals, red_d, nElements
				);
			} else
			if constexpr (reduction == R::min){
				chLaunchKernel( redux::min<blockSize>,
					gridDim, blockDim, 0, 0, // no streams specified
					reduxVals, red_d, nElements
				);
			} else
			if constexpr (reduction == R::max){
				chLaunchKernel( redux::max<blockSize>,
					gridDim, blockDim, 0, 0, // no streams specified
					reduxVals, red_d, nElements
				);
			}
			/* after the first reduction, we switch what to reduce 
			 * from the original data to the redux results */
			reduxVals = red_d;
			/* The second level reduction reduces the results from the first,
			 * so each thread takes reduction results from the previous 
			 * iteration as input, until in the final iteration, 
			 * only a single in-block-reduction is performed. */
			nElements = nBlocks;
			nBlocks = elementsToBlocks(nBlocks, blockSize);
		/* A single element is fully reduced, so that's our cutoff criterion */
		} while ( nElements > 1 );
	}

	Array1s red_h;
	gpuMemcpyDeviceToHost(red_d, red_h);

	/* Cleanup */
	gpuFree(values_d);
	gpuFree(red_d);

	return red_h[0];
}


constexpr Target dev { Target::device };

using R = Reduction;
template scalar reduce<dev, R::sum >(const ArrayXXs& values, int nRuns);
template scalar reduce<dev, R::prod>(const ArrayXXs& values, int nRuns);
template scalar reduce<dev, R::min >(const ArrayXXs& values, int nRuns);
template scalar reduce<dev, R::max >(const ArrayXXs& values, int nRuns);


} // namespace Kokkidio::gpu