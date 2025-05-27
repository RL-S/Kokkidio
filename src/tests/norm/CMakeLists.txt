add_executable( norm "" )
target_link_libraries( norm PRIVATE kokkidio )
target_sources( norm PRIVATE
	main.cpp
	norm_cpu.cpp
	norm_unif_cpu.cpp
)

set_is_cpu( 
	main.cpp
	norm_cpu.cpp
	norm_unif_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( norm PRIVATE
		norm_native_cuda_hip.cpp
	)
endif()

if (KOKKIDIO_USE_SYCL)
	target_sources( norm PRIVATE
		norm_native_sycl.cpp
	)
endif()

if(NOT KOKKIDIO_CPU_ONLY)
	target_sources( norm PRIVATE
		norm_unif_gpu.cpp
	)
endif()

conf(norm)
add_test(NAME norm COMMAND norm)
