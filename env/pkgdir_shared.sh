
export Eigen_SRC="$PKGDIR/eigen"
export Eigen_ROOT="$Eigen_SRC/build"
Kokkos_BASE="$PKGDIR/kokkos"
# only Kokkos_SRC is required
export Kokkos_SRC="$Kokkos_BASE/src/kokkos-dev"
# Kokkos_BUILD and Kokkos_INST are optional,
# but may be used to specify build/install directories for Kokkos.
export Kokkos_BUILD="$Kokkos_BASE/build"
export Kokkos_INST="$Kokkos_BASE/install"
