#include "raxpy.hpp"

#include "unifyBackends.hpp"

namespace Kokkidio::gpu
{

namespace kernel
{

/* CUDA kernel with raxpy logic */
__global__ void cstyle(scalar* z, scalar a, const scalar* x, const scalar* y, int nRows){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nRows){
		raxpy_sum( z[idx], a, x[idx], y[idx] );
	}
}

} // namespace kernel

using K = Kernel;
constexpr Target dev { Target::device };

template<>
void raxpy<dev, K::cstyle>( KOKKIDIO_RAXPY_ARGS ){

	const int nRows = z.rows();

	/* Allocate memory on device and copy data */
	scalar *x_d, *y_d, *z_d;
	gpuAllocAndCopy(x_d, x);
	gpuAllocAndCopy(y_d, y);
	gpuAlloc(z_d, z);

	/* Run calculation multiple times */
	for (volatile int run = 0; run < nRuns; ++run){
		/* Define block and grid dimensions */
		dim3 dimGrid((nRows + 1023) / 1024, 1, 1);
		dim3 dimBlock(1024, 1, 1);

		/* Call the kernel function */
		chLaunchKernel(
			kernel::cstyle,
			dimGrid, dimBlock, 0, 0,
			z_d, a, x_d, y_d, nRows
		);
	}

	/* Copy vector of dot products to host */
	gpuMemcpyDeviceToHost(z_d, z);


	/* Deallocate device memory */
	gpuFree(x_d);
	gpuFree(y_d);
	gpuFree(z_d);
}

} // namespace Kokkidio::gpu
