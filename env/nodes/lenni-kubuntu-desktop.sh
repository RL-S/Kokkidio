
backend_default=hip
PKGDIR="$HOME/pkg"
source "${env_dir}/pkgdir_shared.sh"

Kokkos_ARCH=Kokkos_ARCH_AMD_GFX1201

if [[ "$backend" == "cpu"* ]]; then
	Kokkos_ARCH=Kokkos_ARCH_NATIVE
else
	ROCM_DIR="/opt/rocm"
	cmakeFlags_add="-DCMAKE_PREFIX_PATH=$ROCM_DIR"

	export PATH=$(prepend_path "$ROCM_DIR/lib/llvm/bin" "${PATH:-}")

	echo "which clang: " `which clang`
	export CXX=`command -v clang++`
	export CC=`command -v clang`
fi