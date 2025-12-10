# This relies on a variable being set correctly,
# which contains the path to this directory.
# So we check whether the variable is set and whether it points
# to a directory with the name "cmake".
if( NOT EXISTS "${PROJ_CONF_DIR}" )
message(FATAL_ERROR "Config dir does not exist: ${PROJ_CONF_DIR}")
endif()

# get_filename_component(PROJ_CONF_DIR_DIR "${PROJ_CONF_DIR}" NAME)
# if( NOT "${PROJ_CONF_DIR_DIR}" STREQUAL cmake )
# 	message(FATAL_ERROR "Config dir set incorrectly: ${PROJ_CONF_DIR}")
# endif()
# unset(PROJ_CONF_DIR_DIR)


macro(include_conf FILENAME_NOEXT)
	if("${CMAKE_VERSION}" VERSION_LESS_EQUAL "3.19.0")
		get_filename_component(
			CONF_FILE_PATH
			"${PROJ_CONF_DIR}/${FILENAME_NOEXT}.cmake"
			REALPATH
		)
	else()
		file(
			REAL_PATH
			"${PROJ_CONF_DIR}/${FILENAME_NOEXT}.cmake"
			CONF_FILE_PATH
		)
	endif()
	if ( NOT EXISTS "${CONF_FILE_PATH}" )
		message(FATAL_ERROR "Config file does not exist: ${CONF_FILE_PATH}")
	endif()
	include(${CONF_FILE_PATH})
	unset(CONF_FILE_PATH)
endmacro()
