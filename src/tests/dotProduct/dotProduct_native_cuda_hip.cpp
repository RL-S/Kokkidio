#include "dotProduct.hpp"

#include "unifyBackends.hpp"
#include <Kokkidio.hpp>

#include <magic_enum.hpp>


namespace Kokkidio::gpu
{

namespace dot
{

template<Index blockSize>
__device__ void warpSum(volatile scalar *sdata, unsigned int tid){
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

template<Index blockSize>
__global__ void colwise_manual(const scalar* m1, const scalar* m2, scalar* blocksums, int nRows, int nCols){
	static __shared__ scalar sdata[blockSize];
	/* Every thread processes two values now,
	 * so i and gridSize get doubled.
	 * gridSize is the number of elements that are skipped in the while-loop */
	unsigned int
		tid = threadIdx.x,
		i = 2 * blockSize * blockIdx.x + tid,
		gridSize = 2 * blockSize * gridDim.x;
	sdata[tid] = 0;

	unsigned int uCols { static_cast<unsigned int>(nCols) };
	while (i < uCols){
		for (int j = 0; j<nRows; ++j){
			int idx = i * nRows + j;
			sdata[tid] += m1[idx] * m2[idx];
			if (i+blockDim.x < uCols){
				idx = (i+blockDim.x) * nRows + j;
				sdata[tid] += m1[idx] * m2[idx];
			}
		}
		i += gridSize;
	}
	__syncthreads();

	#define R6_SUM(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		if (tid < BLOCKSIZE_MAX / 2){ \
			sdata[tid] += sdata[tid + BLOCKSIZE_MAX / 2]; \
		} \
		__syncthreads(); \
	}
	R6_SUM(1024)
	R6_SUM( 512)
	R6_SUM( 256)
	R6_SUM( 128)
	#undef R6_SUM

	if (tid < 32) warpSum<blockSize>(sdata, tid);
	if (tid == 0) blocksums[blockIdx.x] = sdata[0];

	// if (blockIdx.x == 0 && tid == 0){
	// // if (tid == 0){
	// 	printf( "Current reduction result: %f\n", sdata[0]);
	// }
	// __syncthreads();
	// output[0] = 10;
}

} // namespace dot


int getNextPowerOf2(int n){
	int k = 1;
	while (k < n)
		k *= 2;
	return k;
}

using K = Kernel;
constexpr Target dev { Target::device };

template<>
scalar dotProduct<dev, K::cstyle_blockbuf>(
	const MatrixXs& m1, const MatrixXs& m2, int nRuns
){
	/* Dimensions check */
	assert(m1.rows() == m2.rows());
	assert(m1.cols() == m2.cols());

	int nRows = m1.rows();
	int nCols = m1.cols();

	static constexpr Index blockSize {1024};

	auto elementsToBlocks = [&](Index nElem, Index blocksize){
		/* in this implementation, each block does twice the work */
		blocksize *= 2;
		return (nElem + blocksize - 1) / blocksize;
	};

	/* Every run must start with this value */
	Index nBlocks { elementsToBlocks(nCols, blockSize) };

	ArrayXs sum_h(nBlocks);

	/* Allocate memory on device and copy data */
	scalar *d_m1, *d_m2, *sum_d;
	gpuAllocAndCopy(d_m1, m1);
	gpuAllocAndCopy(d_m2, m2);
	/* the first reduction step is performed on a per-block-level,
	 * so we need as many elements as blocks for the output data */
	gpuAlloc(sum_d, sum_h);

	scalar result {0};
	for (int run{0}; run<nRuns; ++run){
		result = 0;

		dim3 gridDim (nBlocks, 1, 1);
		dim3 blockDim(blockSize, 1, 1);
		chLaunchKernel( dot::colwise_manual<blockSize>,
			gridDim, blockDim,
			0, // shared memory, static size is defined in kernel
			0, // no streams specified
			d_m1, d_m2, sum_d, nRows, nCols
		);

		/* Copy vector of partial products to host and reduce */
		gpuMemcpyDeviceToHost(sum_d, sum_h);
		result = sum_h.sum();
	}

	gpuFree(d_m1);
	gpuFree(d_m2);
	gpuFree(sum_d);

	return result;
}

} // namespace Kokkidio::gpu