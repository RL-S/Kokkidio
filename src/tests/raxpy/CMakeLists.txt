add_executable( raxpy "" )

target_sources( raxpy PRIVATE
	main.cpp
	raxpy_unif_cpu.cpp
	raxpy_unif_gpu.cpp
	raxpy_native_cpu.cpp
)

set_is_cpu( 
	main.cpp
	raxpy_unif_cpu.cpp
	raxpy_native_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( raxpy PRIVATE
		raxpy_native_cuda_hip.cpp
	)
endif()

# if (KOKKIDIO_USE_SYCL)
# 	target_sources( raxpy PRIVATE
# 		raxpy_native_sycl.cpp
# 	)
# endif()

# # if (KOKKIDIO_USE_OMPT)
# # 	target_sources( dotProduct PRIVATE 
# # 		dotProduct_gpu_ompt.cpp
# # 	)
# # endif()

conf(raxpy)

