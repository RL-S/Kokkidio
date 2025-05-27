
add_executable(colmult "")
target_link_libraries(colmult PRIVATE kokkidio)
target_sources(colmult PRIVATE
	colmult_main.cpp
	colmult_host.cpp
	colmult_device.cpp
)

set_is_cpu(
	colmult_host.cpp
)

configure(colmult)

add_executable(colmult_eigen colmult_eigen.cpp)
target_link_libraries(colmult_eigen PRIVATE kokkidio)
set_is_cpu(
	colmult_eigen.cpp
)

configure(colmult_eigen)
