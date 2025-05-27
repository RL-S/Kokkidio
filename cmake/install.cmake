# installation related commands

set(KOKKIDIO_INST_HEADER_DIR include)
set(KOKKIDIO_INST_LIB_DIR lib)
set(KOKKIDIO_INST_CMAKE_DIR ${KOKKIDIO_INST_LIB_DIR}/cmake/Kokkidio)

install(
	DIRECTORY ${CMAKE_SOURCE_DIR}/include/Kokkidio
	DESTINATION ${KOKKIDIO_INST_HEADER_DIR}
	FILES_MATCHING PATTERN "*.hpp"
)

install(
	FILES ${CMAKE_SOURCE_DIR}/include/Kokkidio.hpp
	DESTINATION ${KOKKIDIO_INST_HEADER_DIR}
)

add_library(kokkidio INTERFACE)
add_library(Kokkidio::kokkidio ALIAS kokkidio)
# Eigen and Kokkos are what we aim to combine
target_link_libraries( kokkidio INTERFACE
	Eigen3::Eigen
	Kokkos::kokkos
)

target_include_directories(kokkidio INTERFACE
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> # for headers when building
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib> # for headers when building
	$<INSTALL_INTERFACE:${KOKKIDIO_INST_HEADER_DIR}>
	$<INSTALL_INTERFACE:${KOKKIDIO_INST_LIB_DIR}>
)

install(
	TARGETS kokkidio
	EXPORT kokkidio
	DESTINATION "${KOKKIDIO_INST_LIB_DIR}"
)
install(
	EXPORT kokkidio
	DESTINATION "${KOKKIDIO_INST_CMAKE_DIR}"
	NAMESPACE Kokkidio::
)

configure_file(
	${PROJ_CONF_DIR}/KokkidioConfigVersion.in.cmake
	${CMAKE_CURRENT_BINARY_DIR}/KokkidioConfigVersion.cmake
	@ONLY
)

configure_file(
	${PROJ_CONF_DIR}/KokkidioVars.in.cmake
	${CMAKE_CURRENT_BINARY_DIR}/KokkidioVars.cmake
	@ONLY
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/KokkidioConfigVersion.cmake
	${CMAKE_CURRENT_BINARY_DIR}/KokkidioVars.cmake
	${PROJ_CONF_DIR}/KokkidioConfig.cmake
	${PROJ_CONF_DIR}/include_conf.cmake
	${PROJ_CONF_DIR}/checkVar.cmake
	${PROJ_CONF_DIR}/sharedOpts.cmake
	${PROJ_CONF_DIR}/checkEigen.cmake
	${PROJ_CONF_DIR}/checkKokkos.cmake
	${PROJ_CONF_DIR}/configureTarget.cmake
	DESTINATION "${KOKKIDIO_INST_CMAKE_DIR}"
)
