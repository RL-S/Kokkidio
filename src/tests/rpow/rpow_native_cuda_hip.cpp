#include "rpow.hpp"

#include "unifyBackends.hpp"

namespace Kokkidio::gpu
{

namespace kernel
{

/* CUDA kernel with rpow logic */
__global__ void cstyle(scalar* optr, const scalar* iptr, int nRows){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nRows){
		optr[idx] = rpow_sum( iptr[idx] );
	}
}

} // namespace kernel

using K = Kernel;
constexpr Target dev { Target::device };

template<>
void rpow<dev, K::cstyle>( KOKKIDIO_RPOW_ARGS ){

	const int nRows = out.rows();

	/* Allocate memory on device and copy data */
	scalar *out_d, *in_d;
	gpuAlloc(out_d, out);
	gpuAllocAndCopy(in_d, in);

	/* Run calculation multiple times */
	for (volatile int run = 0; run < nRuns; ++run){
		/* Define block and grid dimensions */
		dim3 dimGrid((nRows + 1023) / 1024, 1, 1);
		dim3 dimBlock(1024, 1, 1);

		/* Call the kernel function */
		chLaunchKernel(
			kernel::cstyle,
			dimGrid, dimBlock, 0, 0,
			out_d, in_d, nRows
		);
	}

	/* Copy vector of dot products to host */
	gpuMemcpyDeviceToHost(out_d, out);


	/* Deallocate device memory */
	gpuFree(out_d);
	gpuFree(in_d);
}

} // namespace Kokkidio::gpu
