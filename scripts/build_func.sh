
# copied from oneapi
prepend_path() (
	path_to_add="$1"
	path_is_now="$2"

	if [ "" = "${path_is_now}" ] ; then   # avoid dangling ":"
		printf "%s" "${path_to_add}"
	else
		printf "%s" "${path_to_add}:${path_is_now}"
	fi
)


make_title () {
	echo "----------------"
	echo "$1"
	echo "----------------"
}


read_nodefile () {
	echo "Retrieving system-specific variables from \"${env_dir}\" ..."

	source "$env_dir/node_patterns.sh"
	nodefile="$env_dir/nodes/${node_name}.sh"
	if [ -f "$nodefile" ]; then
		echo "Reading node file \"$nodefile\""
		source "$nodefile"
	else
		printf '%s\n%s\n%s\n%s %s %s\n' \
			"Error: Could not find node file to set machine-specific variables." \
			"Creating default node file: " \
			"    $nodefile" \
			"There, you must at least specify the default backend," \
			"and the target architecture." \
			"Rerun this script afterwards. Exiting..." >&2
		cp "$sd/scripts/nodefile_base.sh" "$nodefile"
		exit
	fi
}

check_root () {
	local subj="$1"
	declare rootdir="${subj}_ROOT"
	declare srcdir="${subj}_SRC"

	if [ -n "${!rootdir+x}" ]; then
		if [ -d "${!rootdir}" ]; then
			printf '%s\n%s\n%s\n' \
				"Directory" \
				"    ${rootdir}=\"${!rootdir}\"" \
				"found."
			return 0
		fi
	fi
	return 1
}

check_src () {
	local subj=$1
	local testfile_suffix="$2"
	local required=${3:-false}

	if check_root $subj; then
		echo "Skipping $subj source dir check."
		return 0
	fi

	declare srcdir="${subj}_SRC"
	if [ -z "${!srcdir:+x}" ]; then
		echo "$subj source directory not specified."
		printf '%s%s\n' \
			"Please set the environment variable \"$srcdir\" " \
			"to the $subj source directory!" >&2
		exit
	fi

	local testfile="${!srcdir}/$testfile_suffix"

	echo -n "Checking $subj source directory \"${!srcdir}\"... "
	if [ -f "$testfile" ]; then
		echo "Success."
		return 0
		# echo "$subj source dir: ${!srcdir}"
	else
		printf '%s\n%s\n' \
			"Could not find test file:" \
			"    $testfile" \
			 >&2
		
		if [[ $required == true ]]; then
			printf '%s%s\n' \
				"If source was not downloaded yet, use option -d/--download. " \
				"Exiting...\n" >&2
			exit
		else
			return 1
		fi
	fi
}

check_kokkos_src () {
	local required=${1:-false}
	check_src "Kokkos" "cmake/KokkosConfig.cmake.in" $required
}

check_eigen_src () {
	local required=${1:-false}
	check_src "Eigen" "cmake/Eigen3Config.cmake.in" $required
}

download_lib () {
	local subj="$1"
	local url="$2"
	local hash="${3-:""}"
	declare srcdir="${subj}_SRC"
	declare rootdir="${subj}_ROOT"

	if [ -n "${!srcdir+x}" ]; then
		# echo "yay! running: git clone $url ${!srcdir}"
		# exit
		git clone "$url" "${!srcdir}"
		if [ -n ${hash} ]; then
			cd "${!srcdir}" && git checkout $hash
			cd "${sd}"
		fi
	else
		echo "Download failed: ${subj}_SRC not set! Please set it in your node file, e.g."
		echo "${nodefile}"
		exit
	fi
}


download_libs () {
	if [[ $download_opt != true ]]; then
		return
	fi
	if [[ $buildKokkos == true ]]; then
		if check_kokkos_src; then
			echo "Found Kokkos source or root dir, skipping download"
		else
			download_lib Kokkos https://github.com/kokkos/kokkos.git
		fi
	fi
	if [[ $buildEigen == true ]]; then
		if check_eigen_src; then
			echo "Found Eigen source or root dir, skipping download"
		else
			download_lib Eigen \
				https://gitlab.com/libeigen/eigen.git \
				e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b
		fi
	fi
}


set_backend () {
	if [[ "$backend" == "default" ]]; then
		if [ -n "${backend_default+x}" ]; then
			backend=$backend_default
		else
			printf '%s \n%s %s' \
				"Backend not specified and backend_default not defined." \
				"Please either specify backend as command line option," \
				"or set the variable \"backend_default\" in $nodefile."
			print_help
			exit
		fi
	fi
	echo "Using backend: $backend"
}

# set install prefix according to command line and script vars
set_root () {

	declare varname="$1"
	local defaultval="$2"
	local instdir="${!varname:-$defaultval}"

	if [[ $buildSingle == true ]] && [ -n "$install_prefix" ]; then
		instdir="$install_prefix"
		if [ -n "${!varname+x}" ]; then
			echo "$varname=${!varname}" >&2
			printf '%s \n%s \n%s \n%s \n%s \n' \
				"Both variable " \
				"    ${varname}=\"${!varname}\"" \
				"and command line option " \
				"    --prefix=\"${install_prefix}\"" \
				"found. Command line option takes precedence." \
				>&2
		fi
	fi
	echo "$instdir"
}

set_eigen_root () {
	if [ -n "${Eigen_ROOT+x}" ] || [ -n "${Eigen_SRC+x}" ]; then
		# Eigen_BUILD="${Eigen_BUILD:-$Eigen_SRC/build}"
		# Eigen_INST="${Eigen_INST:-$Eigen_BUILD}"
		Eigen_ROOT="${Eigen_ROOT:-${Eigen_INST:-${Eigen_BUILD:-$Eigen_SRC/build}}}"
	else
		printf '%s\n%s\n%s\n%s\n' \
			"Please define at least one of" \
			"    Eigen_ROOT (build/install directory), or" \
			"    Eigen_SRC  (source directory)" \
			"Exiting..." >&2
		exit
	fi
}

set_kokkos_root () {
	local buildtype=$1
	Kokkos_ROOT="$(set_root Kokkos_INST "$Kokkos_SRC/_install")"
	Kokkos_ROOT+="/$backend/$buildtype"
}

set_kokkidio_root () {
	local buildtype=$1

	Kokkidio_ROOT="$(set_root KOKKIDIO_INST "$sd/_install")"
	Kokkidio_ROOT+="/${backend}/$buildtype"
}

set_root_all () {
	local buildtype=$1
	set_eigen_root
	set_kokkos_root $buildtype
	set_kokkidio_root $buildtype
}


set_gpu_arch() {
	local required=${1:-false}
	if [ -n "${Kokkos_ARCH+x}" ]; then
		if [[ "$Kokkos_ARCH" == "Kokkos_ARCH_SOMEVALUE" ]]; then
			echo "Error: Kokkos_ARCH invalid."
			exit
		fi
		cmakeFlags+=" -D${Kokkos_ARCH}=ON"
		# Kokkos requires this for SYCL
		if [[ "$backend" == "sycl" ]]; then
			cmakeFlags+=" -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON"
		fi
	else
		local arch_help=$(printf '%s\n%s\n%s\n' \
			"Kokkos_ARCH=Kokkos_ARCH_<specifier>" \
			"You can find the specifiers here:" \
			"https://kokkos.org/kokkos-core-wiki/keywords.html#architectures" \
		)
		if [[ $required == true ]]; then
			echo "Please specify the GPU architecture in $nodefile as"
			echo "$arch_help"
			exit
		else
			echo "No GPU architecture specified, falling back to autodetection."
			echo "You may explicitly specify the GPU architecture in $nodefile using"
			echo "$arch_help"
		fi
	fi
}

check_install_vars () {
	local subj="$1"
	declare rootdir="${subj}_ROOT"
	declare instdir="${subj}_INST"
	echo "Checking $subj install..."
	if [ -z "${!rootdir:+x}" ] && [ -z "${instdir:+x}" ]; then
		echo "Variables $rootdir and $instdir both not set or empty." >&2
		return 1
	else
		return 0
	fi
}

check_eigen_install () {
	local required=${1:-false}
	local subj="Eigen"
	declare rootdir="${subj}_ROOT"
	if check_install_vars $subj; then
		local testfile="${Eigen_INST:-$Eigen_ROOT}/Eigen3Config.cmake"
		echo "Checking for $subj cmake config"
		echo -n "    \"$testfile\": "
		if [ -f "$testfile" ]; then
			printf '%s\n%s\n' "Found." "$rootdir dir: ${!rootdir}"
			return 0
		else
			echo "Not found." >&2
		fi
	fi

	if [[ $required == true ]]; then
		printf "Could not find $subj cmake config file!" >&2
		exit
	else
		return 1
	fi
}

check_kokkos_install () {
	local required=${1:-false}
	local subj="Kokkos"
	declare rootdir="${subj}_ROOT"

	if check_install_vars $subj; then
		local testfile_tail="cmake/Kokkos/KokkosConfig.cmake"
		echo "Checking for $subj cmake config"
		for subdir in lib lib64; do
			local testfile="$Kokkos_ROOT/$subdir/$testfile_tail"
			echo -n "    \"$testfile\": "
			if [ -f "$testfile" ]; then
				printf '%s\n%s\n' "Found." "$rootdir dir: ${!rootdir}"
				return 0
			else
				echo "Not found." >&2
			fi
		done
	fi

	if [[ $required == true ]]; then
		printf '%s%s\n' \
			"Could not find $subj cmake config file! " \
			"Did you install after building?"
		exit
	else
		return 1
	fi
}

check_kokkidio_install () {
	local required=${1:-false}
	local subj="Kokkidio"
	declare rootdir="${subj}_ROOT"
	declare instdir="${subj}_INST"

	if check_install_vars $subj; then
		local testfile="${!instdir:-${!rootdir}}/lib/cmake/Kokkidio/KokkidioConfig.cmake"
		echo "Checking for $subj cmake config"
		echo -n "    \"$testfile\": "
		if [ -f "$testfile" ]; then
			printf '%s\n%s\n' "Found." "$rootdir dir: ${!rootdir}"
			return 0
		else
			echo "Not found." >&2
		fi
	fi

	if [[ $required == true ]]; then
		printf "Could not find $subj cmake config file!" >&2
		exit
	else
		return 1
	fi
}

check_install_all () {
	local required=${1:-false}
	check_eigen_install $required
	check_kokkos_install $required
	check_kokkidio_install $required
}

cmake_flags_reset () {
	# reset cmakeFlags between runs/backends etc.
	cmakeFlags=""
	cmakeFlags+=" ${cmakeFlags_add:-""} "
}

cmake_flags_set_shared () {
	cmake_flags_reset
	cmakeFlags+=" -DCMAKE_BUILD_TYPE=$buildtype"
	# cmakeFlags+=" -DCMAKE_CXX_STANDARD=17"
	# Kokkos otherwise emits a warning
	cmakeFlags+=" -DCMAKE_CXX_EXTENSIONS=OFF"
}

cmake_flags_kokkos_backend () {
	# translate to Kokkos naming
	if [[ $backend == "ompt" ]]; then
		local kokkos_backend="openmptarget"
	else
		local kokkos_backend="$backend"
	fi

	if [[ ! $backend =~ cpu ]]; then
		cmakeFlags+=" -DKokkos_ENABLE_${kokkos_backend^^}=ON"
	fi
}


build_cmake () {
	local builddir="$1"
	local srcdir="$2"
	local instdir="${3:-""}"
	echo "Build directory: $builddir"
	echo "Source directory: $srcdir"
	if [ -n "$instdir" ]; then
		echo "Install directory: $instdir"
		cmakeFlags+=" -DCMAKE_INSTALL_PREFIX=$instdir"
	fi
	echo "CMake flags: $cmakeFlags"
	mkdir -p "$builddir"
	cmake -B "$builddir" $cmakeFlags ${srcdir}

	if [[ $noBuild == true ]]; then
		echo "Build with \"cmake --build $builddir -j\""
	else
		echo "Building with CMake..."
		cmake --build "$builddir" -j
		if [[ $install_opt == true ]] && [[ $instdir != "" ]]; then
			cmake --build "$builddir" -- install
		fi
	fi
}


build_eigen () {
	make_title "Building Eigen."

	set_eigen_root

	if check_eigen_install; then
		echo "Found existing Eigen installation, skipping build."
		return
	fi

	check_eigen_src true
	Eigen_BUILD="${Eigen_BUILD:-$Eigen_SRC/_build}"
	cmake_flags_reset

	noBuild_tmp=$noBuild
	noBuild=true
	build_cmake "$Eigen_ROOT" "$Eigen_SRC"
	noBuild=$noBuild_tmp
}


build_kokkos () {
	local buildtype=$1

	make_title "Building Kokkos for ${backend^^}, build type \"$buildtype\"."

	set_kokkos_root $buildtype

	if check_kokkos_install; then
		echo "Found existing Kokkos installation, skipping build."
		return
	fi

	check_kokkos_src true
	Kokkos_BUILD="${Kokkos_BUILD:-$Kokkos_SRC/_build}/${backend}/$buildtype"

	cmake_flags_set_shared
	cmake_flags_kokkos_backend

	cmakeFlags+=" -DKokkos_ENABLE_OPENMP=ON"
	cmakeFlags+=" -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON"
	if [[ $backend =~ cuda|hip ]]; then
		set_gpu_arch $backend true
		if [[ $backend == cuda ]]; then
			cmakeFlags+=" -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
			cmakeFlags+=" -DKokkos_ENABLE_CUDA_LAMBDA=ON"
		fi
	fi
	if [[ $backend =~ sycl|ompt ]]; then
		set_gpu_arch $backend
	fi

	build_cmake "$Kokkos_BUILD" "$Kokkos_SRC" "$Kokkos_ROOT"
}


build_kokkidio () {
	local buildtype=$1

	make_title "Configuring Kokkidio for ${backend^^}, build type \"$buildtype\"."

	set_root_all $buildtype

	# # Kokkidio should get reinstalled if the build script is rerun.
	# # Otherwise, after changing Kokkidio files, one would have to manually
	# # delete the installation directory.
	# if check_kokkidio_install; then
	# 	echo "Found matching Kokkidio installation, skipping."
	# 	return
	# fi

	check_eigen_install true
	check_kokkos_install true

	cmake_flags_set_shared
	cmake_flags_kokkos_backend

	builddir="_build/kokkidio/$backend/$buildtype"
	cmakeFlags+=" -DEigen_ROOT=$Eigen_ROOT"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd" "$Kokkidio_ROOT"
}

build_tests () {
	local buildtype=$1
	local scalar=$2

	make_title "Building tests with ${backend^^}, build type \"$buildtype\", scalar type \"$scalar\"."

	set_root_all $buildtype
	check_install_all true

	export KOKKIDIO_REAL_SCALAR=$scalar
	builddir="_build/tests/$backend/$buildtype/$scalar"

	cmake_flags_set_shared
	cmake_flags_kokkos_backend

	cmakeFlags+=" -DEigen_ROOT=$Eigen_ROOT"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"
	cmakeFlags+=" -DKokkidio_ROOT=$Kokkidio_ROOT"

	if [[ $backend_default == cuda ]] && [[ $backend =~ ompt|sycl ]]; then
		runbuild=Release
	else
		runbuild=Debug
	fi

	if [[ $buildtype == $runbuild ]]; then
		cmakeFlags+=" -DKOKKIDIO_RUN_ALL_TESTS=ON"
	fi

	echo "Running build commands..."
	build_cmake "$builddir" "$sd/src/tests"


	# if [[ $backend != "hip" ]]; then
	# 	cmakeFlags+=" -DCMAKE_INSTALL_PREFIX=${sd}/install"
	# 	build_cmake "$builddir" "$sd" true
	# 	build_cmake "$builddir/tests" "$sd/tests"
	# elif [[ $noBuild == true ]]; then
	# 	export KOKKIDIO_BUILDTYPE=$buildtype
	# 	make -j
	# fi

	echo "Finished compilation for ${backend^^}, build type \"$buildtype\", scalar type \"$scalar\"."

	if [[ $noRun != true ]] && [[ $buildtype == $runbuild ]]; then
		echo "Performing test runs in $buildtype mode..."
		# Without these OMP settings, Kokkos complains about their absence.
		export OMP_PROC_BIND=spread 
		export OMP_PLACES=threads
		$builddir/axpy/axpy             --size 4  50 --runs 2
		$builddir/dotProduct/dotProduct --size 4  50 --runs 2
		$builddir/norm/norm             --size 50    --runs 2
		$builddir/friction/friction     --size 50    --runs 2
		$builddir/rpow/rpow             --size 50    --runs 2
		$builddir/raxpy/raxpy           --size 50    --runs 2
		echo "Test runs in $buildtype mode finished."
	fi
}

build_examples () {
	local buildtype=$1

	make_title "Building examples with ${backend^^}, build type \"$buildtype\"."

	set_root_all $buildtype
	check_install_all true

	export KOKKIDIO_REAL_SCALAR=float
	builddir="_build/examples/$buildtype"

	cmake_flags_set_shared
	cmake_flags_kokkos_backend

	cmakeFlags+=" -DEigen_ROOT=$Eigen_ROOT"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"
	cmakeFlags+=" -DKokkidio_ROOT=$Kokkidio_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd/src/examples"

	echo "Finished compilation for ${backend^^}, build type \"$buildtype\"."
	printf 'Executables can be found in \n\t%s\n' \
		"${builddir}"
}

