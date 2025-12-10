# kokkidio-config.cmake - package configuration file

get_filename_component(PROJ_CONF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
if ( EXISTS ${PROJ_CONF_DIR}/kokkidio.cmake )
    include(${PROJ_CONF_DIR}/kokkidio.cmake)
endif()

include("${PROJ_CONF_DIR}/include_conf.cmake")

enable_language(CXX)
include_conf(KokkidioVars)
include_conf(checkVar)
include_conf(sharedOpts)
include_conf(checkKokkos)
include_conf(checkEigen)
include_conf(configureTarget)
