#include "norm.hpp"

#include "unifyBackends.hpp"
#include <Kokkidio.hpp>

#include <magic_enum.hpp>


namespace Kokkidio::gpu
{

namespace kernel
{

__device__ constexpr scalar smin {std::numeric_limits<scalar>::min()};

template<Index blockSize>
__device__ void warpMax(volatile scalar *sdata, unsigned int tid){
	#define KOKKIDIO_WR_UNROLL(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		sdata[tid] = ::max(sdata[tid], sdata[tid + BLOCKSIZE_MAX / 2]); \
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
__global__ void norm_cstyle(const scalar* mat, scalar* blocknorms, int nRows, int nCols){
	static __shared__ scalar sdata[blockSize];
	/* Every thread processes two values now,
	 * so j and gridSize get doubled.
	 * gridSize is the number of elements that are skipped in the while-loop */
	unsigned int
		tid = threadIdx.x,
		j = 2 * blockSize * blockIdx.x + tid,
		gridSize = 2 * blockSize * gridDim.x;
	sdata[tid] = smin;

	unsigned int uCols { static_cast<unsigned int>(nCols) };
	while (j < uCols){
		scalar norm1 {0}, norm2 {0};
		for (int i = 0; i<nRows; ++i){
			int idx = j * nRows + i;
			norm1 += mat[idx] * mat[idx];
			// sdata[tid] = mat[idx] * m2[idx];
			if (j+blockDim.x < uCols){
				idx = (j+blockDim.x) * nRows + i;
				norm2 += mat[idx] * mat[idx];
				// sdata[tid] += m1[idx] * m2[idx];
			}
		}
		sdata[tid] = ::max( sqrt(norm1), sqrt(norm2) );
		j += gridSize;
	}
	__syncthreads();

	#define R6_MAX(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		if (tid < BLOCKSIZE_MAX / 2){ \
			sdata[tid] = ::max(sdata[tid], sdata[tid + BLOCKSIZE_MAX / 2]); \
		} \
		__syncthreads(); \
	}
	R6_MAX(1024)
	R6_MAX( 512)
	R6_MAX( 256)
	R6_MAX( 128)
	#undef R6_MAX

	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0) blocknorms[blockIdx.x] = sdata[0];
}

} // namespace kernel


int getNextPowerOf2(int n){
	int k = 1;
	while (k < n)
		k *= 2;
	return k;
}

using K = Kernel;
constexpr Target dev { Target::device };

template<>
scalar norm<dev, K::cstyle_blockbuf>(const MatrixXs& mat, int nRuns){

	int nRows = mat.rows();
	int nCols = mat.cols();

	static constexpr Index blockSize {1024};

	auto elementsToBlocks = [&](Index nElem, Index blocksize){
		/* in this implementation, each block does twice the work */
		blocksize *= 2;
		return (nElem + blocksize - 1) / blocksize;
	};

	/* Every run must start with this value */
	Index nBlocks { elementsToBlocks(nCols, blockSize) };

	ArrayXs max_h {nBlocks};

	/* Allocate memory on device and copy data */
	scalar *mat_d, *max_d;
	gpuAllocAndCopy(mat_d, mat);
	/* the first reduction step is performed on a per-block-level,
	 * so we need as many elements as blocks for the output data */
	gpuAlloc(max_d, max_h);

	scalar result {0};
	for (int run{0}; run<nRuns; ++run){
		result = 0;

		dim3 gridDim (nBlocks, 1, 1);
		dim3 blockDim(blockSize, 1, 1);
		chLaunchKernel( kernel::norm_cstyle<blockSize>,
			gridDim, blockDim,
			0, // shared memory, static size is defined in kernel
			0, // no streams specified
			mat_d, max_d, nRows, nCols
		);

		/* Copy vector of partial products to host and reduce */
		gpuMemcpyDeviceToHost(max_d, max_h);
		result = max_h.maxCoeff();
	}

	gpuFree(mat_d);
	gpuFree(max_d);

	return result;
}

} // namespace Kokkidio::gpu