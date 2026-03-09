add_executable( redux "" )
target_link_libraries( redux PRIVATE kokkidio )
target_sources( redux PRIVATE
	main.cpp
	sum_run.cpp
	sum_cpu.cpp
	reduce_run.cpp
	reduce_cpu.cpp
)

set_is_cpu(
	main.cpp
	sum_run.cpp
	sum_cpu.cpp
	reduce_run.cpp
	reduce_cpu.cpp
)

if (KOKKIDIO_USE_CUDA)
	target_sources( redux PRIVATE
		sum.cu
		reduce_generic.cu
	)
endif()

# if(KOKKIDIO_USE_SYCL)
# 	target_sources( friction_context PRIVATE
# 		friction_context_gpu.cpp
# 	)
# endif()

# if (KOKKIDIO_USE_OMPT)
# 	target_sources( friction_context PRIVATE 
# 		friction_context_gpu.cpp
# 	)
# endif()

conf(redux)
add_test(NAME redux COMMAND redux)
