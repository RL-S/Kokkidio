add_executable( rpow "" )

target_sources( rpow PRIVATE
	main.cpp
	rpow_unif_cpu.cpp
	rpow_unif_gpu.cpp
	rpow_native_cpu.cpp
)

set_is_cpu( 
	main.cpp
	rpow_unif_cpu.cpp
	rpow_native_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( rpow PRIVATE
		rpow_native_cuda_hip.cpp
	)
endif()

# if (KOKKIDIO_USE_SYCL)
# 	target_sources( rpow PRIVATE
# 		rpow_native_sycl.cpp
# 	)
# endif()

# # if (KOKKIDIO_USE_OMPT)
# # 	target_sources( dotProduct PRIVATE 
# # 		dotProduct_gpu_ompt.cpp
# # 	)
# # endif()

conf(rpow)

