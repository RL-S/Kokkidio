cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(Kokkidio_bench)

if ( ${PROJECT_SOURCE_DIR} STREQUAL  ${CMAKE_SOURCE_DIR})
	find_package(Kokkidio 1.0 REQUIRED)
endif()

message(STATUS "Kokkidio_DIR: ${Kokkidio_DIR}")

message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")

option(KOKKIDIO_RUN_ALL_TESTS "Run all test implementations" OFF)
message(STATUS "KOKKIDIO_RUN_ALL_TESTS=${KOKKIDIO_RUN_ALL_TESTS}")

macro(set_warnings TARGET_NAME)
	target_compile_options( ${TARGET_NAME} PRIVATE
	-Wall -Wextra -Wno-deprecated-declarations
	)
endmacro()

macro(conf TARGET_NAME)
	set_warnings(${TARGET_NAME})
	kokkidio_configure_target(${TARGET_NAME})
	if(KOKKIDIO_RUN_ALL_TESTS)
		target_compile_definitions(${TARGET_NAME} PRIVATE
			KOKKIDIO_RUN_ALL_TESTS=1
		)
	endif()
endmacro()

include_directories(
	"${PROJECT_SOURCE_DIR}/include"
	#"${CMAKE_SOURCE_DIR}/../../lib"
)
enable_testing()
add_subdirectory(dotProduct)
add_subdirectory(friction)
# add_subdirectory(redux)
add_subdirectory(axpy)
add_subdirectory(norm)
