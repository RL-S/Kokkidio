cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(Kokkidio_examples)
#if ( NOT ${PROJECT_SOURCE_DIR} STREQUAL  ${CMAKE_SOURCE_DIR})
#	list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)
#	list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/)
#	message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
#
#	set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/")
#	include(${CMAKE_SOURCE_DIR}/cmake/KokkidioConfig.cmake)

if ( ${PROJECT_SOURCE_DIR} STREQUAL  ${CMAKE_SOURCE_DIR})
	find_package(Kokkidio 1.0 REQUIRED)
endif()

message(STATUS "Kokkidio_DIR: ${Kokkidio_DIR}")

message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")

macro(set_warnings TARGET_NAME)
	target_compile_options( ${TARGET_NAME} PRIVATE
		-Wall -Wextra -Wno-deprecated-declarations
	)
endmacro()

macro(configure TARGET_NAME)
	set_warnings(${TARGET_NAME})
	kokkidio_configure_target(${TARGET_NAME})
endmacro()

#include_directories(
#	"${CMAKE_CURRENT_SOURCE_DIR}/../../lib"
#)

add_executable(viewmap ViewMap.cpp)
target_link_libraries(viewmap PRIVATE kokkidio)
configure(viewmap)

add_executable(dualviewmap DualViewMap.cpp)
target_link_libraries(dualviewmap PRIVATE kokkidio)
configure(dualviewmap)

add_executable(axpy axpy.cpp)
target_link_libraries(axpy PRIVATE kokkidio)
configure(axpy)

add_subdirectory(colmult)

