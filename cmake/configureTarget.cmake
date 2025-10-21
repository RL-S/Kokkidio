
# this must be called on every target
macro(kokkidio_configure_target TARGET_NAME)
	# default visibility is PRIVATE, but it can be changed via optional argument
	set(TARGET_VISIBILITY PRIVATE)
	# Cannot use ARGN directly with list() command,
	# so copy it to a variable first.
	set (extra_args ${ARGN})
	list(LENGTH extra_args extra_count)
	if (${extra_count} GREATER 0)
		list(GET extra_args 0 TARGET_VISIBILITY)
	endif ()

	target_link_libraries( ${TARGET_NAME} ${TARGET_VISIBILITY}
		Kokkidio::kokkidio 
	)

	if (OpenMP_CXX_FOUND)
		message(STATUS "Found OpenMP, v. ${OpenMP_CXX_VERSION}. Linking target ${TARGET_NAME}...")
		target_link_libraries( ${TARGET_NAME} ${TARGET_VISIBILITY}
			OpenMP::OpenMP_CXX
		)
		# IntelLLVM (icpx) doesn't seem to define _OPENMP 
		# when passing -fiopenmp/-qopenmp, 
		# and we can't just add it as a compile definition here,
		# because something breaks inside icpx' "omp.h" 
		# when _OPENMP is defined beforehand (i.e. the normal way...).
		# So instead only use KOKKIDIO_OPENMP as a replacement,
		# and do the rest in macros.hpp.
		# The error is a conflicting declaration of omp_is_initial_device:
		# /opt/intel/oneapi/compiler/2024.1/bin/compiler/../../opt/compiler/include/omp.h:533:23: 
		# error: static declaration of 'omp_is_initial_device' follows non-static declaration
		# 533 |     static inline int omp_is_initial_device(void) { return 1; }
		#     |                       ^
		# /opt/intel/oneapi/compiler/2024.1/bin/compiler/../../opt/compiler/include/omp.h:135:40: 
		# note: previous declaration is here
		# 135 |     extern int  __KAI_KMPC_CONVENTION  omp_is_initial_device (void);
		if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
			target_compile_definitions( ${TARGET_NAME} ${TARGET_VISIBILITY}
				KOKKIDIO_OPENMP
			)
		endif()
	else()
		message(WARNING "Could not find OpenMP.")
	endif()

	target_compile_definitions( ${TARGET_NAME} ${TARGET_VISIBILITY}
		${KOKKIDIO_BACKEND}
		${KOKKIDIO_KOKKOS_BACKEND}
		KOKKIDIO_REAL_SCALAR=${KOKKIDIO_REAL_SCALAR_CMAKE}
	)

	if (NOT DEFINED KOKKIDIO_NO_GPU)
		message(STATUS
			"Passing preprocessor definitions for using Eigen on GPUs..."
		)
		target_compile_definitions( ${TARGET_NAME} ${TARGET_VISIBILITY}
			EIGEN_USE_GPU
			EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
		)
	endif()

	if (KOKKIDIO_USE_SYCL)
		target_compile_definitions( ${TARGET_NAME} ${TARGET_VISIBILITY}
			EIGEN_USE_SYCL
		)
	endif()

	if(KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
		# suppresses "attribute __host__ does not apply here".
		target_compile_options(${TARGET_NAME} ${TARGET_VISIBILITY} $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1835>)
		# target_compile_options( ${TARGET_NAME} PRIVATE "-diag-suppress 1835" )
	else()
		# nvcc passes the wrong style of line directive to gcc,
		# which creates a hundreds of warnings when compiled with -pedantic.
		target_compile_options( ${TARGET_NAME} ${TARGET_VISIBILITY} -pedantic)
	endif()
	
	if(DEFINED OMP_RPATH)
		message(STATUS "Setting ${TARGET_NAME} RPATH to \"${OMP_RPATH}\"")
		set_target_properties( ${TARGET_NAME} PROPERTIES
			INSTALL_RPATH "${OMP_RPATH}"
			BUILD_WITH_INSTALL_RPATH TRUE
		)
	endif()


	# make sure vectorisation works as intended
	# check a command-line and env var for override

	if(KOKKIDIO_NO_ARCH_NATIVE_CMAKE)
		message(STATUS "${TARGET_NAME}: skipping option \"-march=native\".")
	else()
		message(STATUS "Setting ${TARGET_NAME} option \"-march=native\".")
		target_compile_options(${TARGET_NAME} ${TARGET_VISIBILITY}
			$<$<COMPILE_LANGUAGE:CXX>:-march=native>
		)
	endif()



	# get_target_property(COMP_DEFS ${TARGET_NAME} COMPILE_DEFINITIONS)
	# message(STATUS "Compile definitions for ${TARGET_NAME}: ${COMP_DEFS}")

	# get_target_property(LINKED_LIBS ${TARGET_NAME} LINK_LIBRARIES)
	# message(STATUS "Linked libraries for ${TARGET_NAME}: ${LINKED_LIBS}")
endmacro()