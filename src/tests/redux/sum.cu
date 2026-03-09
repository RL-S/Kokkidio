#include "redux.hpp"
#include "redux_cudahip_generic.hpp"
#include "unifyBackends.hpp"

#include <magic_enum.hpp>

namespace Kokkidio::gpu
{

namespace kernel
{

template <Index blockSize, typename Scalar>
__device__ void warpSum(volatile Scalar *sdata, unsigned int tid){
	#define KOKKIDIO_WR_UNROLL(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		sdata[tid] += sdata[tid + BLOCKSIZE_MAX / 2]; \
	}
	KOKKIDIO_WR_UNROLL(64)
	KOKKIDIO_WR_UNROLL(32)
	KOKKIDIO_WR_UNROLL(16)
	KOKKIDIO_WR_UNROLL( 8)
	KOKKIDIO_WR_UNROLL( 4)
	KOKKIDIO_WR_UNROLL( 2)
	#undef KOKKIDIO_WR_UNROLL
}


#define KOKKIDIO_REDUX_SUM_NAME sum6
#include "redux_cu_sum6_impl.hpp"

#define KOKKIDIO_REDUX_SUM6_CONSTEXPR_BRANCH
#define KOKKIDIO_REDUX_SUM_NAME sum6_cbranch
#include "redux_cu_sum6_impl.hpp"


#define KOKKIDIO_REDUX_SUM_NAME sum0_dyn
#include "redux_cu_sum0_impl.hpp"

#define KOKKIDIO_REDUX_SUM0_CONSTEXPR_BLOCKSIZE
#define KOKKIDIO_REDUX_SUM_NAME sum0_fixed
#include "redux_cu_sum0_impl.hpp"


} // namespace kernel


int getNextPowerOf2(int n){
	int k = 1;
	while (k < n)
		k *= 2;
	return k;
}


template<Target target, Kernel k>
scalar sum(const ArrayXXs& values, int nRuns){

	static constexpr Index blockSize {1024};

	using K = Kernel;
	auto elementsToBlocks = [&](Index nElem, Index blocksize){
		/* in sum6, each block does twice the work */
		if constexpr (
			k == K::sum6 ||
			k == K::sum6_cbranch ||
			k == K::sum_generic
		){
			blocksize *= 2;
		}
		return (nElem + blocksize - 1) / blocksize;
	};
	/* in the case of dynamically-sized shared memory,
	 * there can only be advantages, if we reduce the blocksize accordingly */
	auto p2blockSize = [&](Index nElem){
		return std::min( blockSize, getNextPowerOf2(nElem) );
	};

	/* Every run must start with this value */
	Index
		nElementsInit {values.size()},
		nBlocksInit { elementsToBlocks(nElementsInit, blockSize) };

	/* We don't need very many device arrays this time.
	 * Just one input and one output array. */
	scalar
		*values_d,
		*sum_d;

	/* the first reduction step is performed on a per-block-level,
	 * so we need as many elements as blocks for the output data */
	gpuAlloc(sum_d, nBlocksInit);
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
			#define PRINT_ITER(SIZEVAR) printd( \
				"Iteration %i: Reducing %i elements " \
				"using %i block(s) of size %i.\n" \
				, i++ \
				, nElements \
				, nBlocks \
				, SIZEVAR )

			dim3 gridDim (nBlocks, 1, 1);
			if constexpr (k == K::sum0_dyn){
				/* this one always is a power of 2. */
				int blockSize_p2 { p2blockSize(nElements) };
				dim3 blockDim(blockSize_p2, 1, 1);
				PRINT_ITER(blockSize_p2);
				chLaunchKernel( kernel::sum0_dyn,
					gridDim, blockDim,
					/* shared memory, dynamically-sized */
					blockSize_p2*sizeof(scalar),
					0, // no streams specified
					reduxVals, sum_d, nElements
				);
			} else
			if constexpr (k == K::sum0_fixed){
				dim3 blockDim(blockSize, 1, 1);
				PRINT_ITER(blockSize);
				chLaunchKernel( kernel::sum0_fixed<blockSize>,
					gridDim, blockDim,
					0, // shared memory, static size is defined in kernel
					0, // no streams specified
					reduxVals, sum_d, nElements
				);
			} else
			if constexpr (k == K::sum6){
				dim3 blockDim(blockSize, 1, 1);
				PRINT_ITER(blockSize);
				chLaunchKernel( kernel::sum6<blockSize>,
					gridDim, blockDim,
					0, // shared memory, static size is defined in kernel
					0, // no streams specified
					reduxVals, sum_d, nElements
				);
			} else
			if constexpr (k == K::sum6_cbranch){
				dim3 blockDim(blockSize, 1, 1);
				PRINT_ITER(blockSize);
				#define KOKKIDIO_LAUNCH_SUM6(CONSTBRANCH_BOOL) \
				chLaunchKernel( \
					(kernel::sum6_cbranch<blockSize, CONSTBRANCH_BOOL>), \
					gridDim, blockDim, 0, 0, reduxVals, sum_d, nElements )
				if ( nElements > 2 * blockSize ){
					KOKKIDIO_LAUNCH_SUM6(false);
				} else {
					KOKKIDIO_LAUNCH_SUM6(true);
				}
			} else
			if constexpr (k == K::sum_generic){
				dim3 blockDim(blockSize, 1, 1);
				PRINT_ITER(blockSize);
				chLaunchKernel( redux::sum<blockSize>,
					gridDim, blockDim,
					0, // shared memory, static size is defined in kernel
					0, // no streams specified
					reduxVals, sum_d, nElements
				);
			}
			/* after the first reduction, we switch what to reduce 
			 * from the original data to the redux results */
			reduxVals = sum_d;
			/* The second level reduction reduces the results from the first,
			 * so each thread takes reduction results from the previous 
			 * iteration as input, until in the final iteration, 
			 * only a single in-block-reduction is performed. */
			nElements = nBlocks;
			nBlocks = elementsToBlocks(nBlocks, blockSize);
		/* A single element is fully reduced, so that's our cutoff criterion */
		} while ( nElements > 1 );
	}

	Array1s sum_h;
	gpuMemcpyDeviceToHost(sum_d, sum_h);

	/* Cleanup */
	gpuFree(values_d);
	gpuFree(sum_d);

	return sum_h[0];
}

constexpr Target dev { Target::device };
using K = Kernel;

template scalar sum<dev, K::sum0_fixed  >(const ArrayXXs& values, int nRuns);
template scalar sum<dev, K::sum0_dyn    >(const ArrayXXs& values, int nRuns);
template scalar sum<dev, K::sum6        >(const ArrayXXs& values, int nRuns);
template scalar sum<dev, K::sum6_cbranch>(const ArrayXXs& values, int nRuns);
template scalar sum<dev, K::sum_generic >(const ArrayXXs& values, int nRuns);



} // namespace Kokkidio::gpu

