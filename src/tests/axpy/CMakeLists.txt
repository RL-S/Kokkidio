add_executable( test_axpy "" )
target_link_libraries( test_axpy PRIVATE kokkidio )
target_sources( test_axpy PRIVATE
	main.cpp
	axpy_unif_cpu.cpp
	axpy_unif_gpu.cpp
	axpy_native_cpu.cpp
)

set_is_cpu( 
	main.cpp
	axpy_unif_cpu.cpp
	axpy_native_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( test_axpy PRIVATE
		axpy_native_cuda_hip.cpp
	)
endif()

if (KOKKIDIO_USE_SYCL)
	target_sources( test_axpy PRIVATE
		axpy_native_sycl.cpp
	)
endif()

# if (KOKKIDIO_USE_OMPT)
# 	target_sources( dotProduct PRIVATE 
# 		dotProduct_gpu_ompt.cpp
# 	)
# endif()

conf(test_axpy)
add_test(NAME test_axpy COMMAND test_axpy)

