
set_if_defined(KOKKIDIO_REAL_SCALAR_CMAKE KOKKIDIO_REAL_SCALAR)
if (NOT DEFINED KOKKIDIO_REAL_SCALAR_CMAKE)
	message(STATUS
		"KOKKIDIO_REAL_SCALAR not set. Set it to either float or double "
		"to change the underlying data type of all computations. "
		"Using default value \"float\"."
	)
	set(KOKKIDIO_REAL_SCALAR_CMAKE float)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM")
	message(STATUS "Compiler ID \"${CMAKE_CXX_COMPILER_ID}\" "
		"may not build executables with RPATH for OpenMP libraries. "
		"Attempting to add RPATH to executable..."
	)
	set_if_defined(OMP_RPATH OMP_LIB_DIR)
	if(DEFINED OMP_RPATH)
		message(STATUS "Using OpenMP library path \"${OMP_RPATH}\"")
		set(OMP_RPATH_LINE "set(OMP_RPATH \"${OMP_RPATH}\")")
	else()
		message(WARNING
			"OMP_LIB_DIR not provided. "
			"You may need to specify LD_LIBRARY_PATH for libomp[...] "
			"when calling your executable."
		)
	endif()
endif()


set_if_defined(KOKKIDIO_NO_ARCH_NATIVE_CMAKE KOKKIDIO_NO_ARCH_NATIVE)
if(NOT DEFINED KOKKIDIO_NO_ARCH_NATIVE_CMAKE)
	message(STATUS "Using -march=native to enable host code vectorisation.")
	message(STATUS "Set \"KOKKIDIO_NO_ARCH_NATIVE\" to ON to disable.")
	set(KOKKIDIO_NO_ARCH_NATIVE_CMAKE OFF)
endif()
