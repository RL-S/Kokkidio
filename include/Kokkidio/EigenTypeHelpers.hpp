#ifndef KOKKIDIO_EIGENTYPEHELPERS_HPP
#define KOKKIDIO_EIGENTYPEHELPERS_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/typeAliases.hpp"

namespace Kokkidio
{

template<typename T>
struct is_eigen_matrix : std::false_type {};

template<typename T>
struct is_eigen_array : std::false_type {};

template<typename Scalar, int ... opts>
struct is_eigen_matrix<Eigen::Matrix<Scalar, opts ...>> : std::true_type {};

template<typename Scalar, int ... opts>
struct is_eigen_array<Eigen::Array<Scalar, opts ...>> : std::true_type {};

template<typename T>
inline constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;

template<typename T>
inline constexpr bool is_eigen_array_v = is_eigen_array<T>::value;


template<typename EigenType>
inline constexpr bool is_eigen_dense =
	std::is_base_of_v< Eigen::DenseBase<EigenType>, EigenType >;



template<typename T>
struct is_owning_eigen_type : std::false_type {};

template<typename Scalar, int ... opts>
struct is_owning_eigen_type<Eigen::Matrix<Scalar, opts ...>> : std::true_type {};

template<typename Scalar, int ... opts>
struct is_owning_eigen_type<Eigen::Array<Scalar, opts ...>> : std::true_type {};

template<typename T>
inline constexpr bool is_owning_eigen_type_v = is_owning_eigen_type<T>::value;


template<typename T>
struct is_eigen_map : std::false_type {};

template<typename PlainObjectType, int mapOptions, typename StrideType>
struct is_eigen_map<Eigen::Map<PlainObjectType, mapOptions, StrideType>> :
	std::true_type
{};

template<typename T>
inline constexpr bool is_eigen_map_v = is_eigen_map<T>::value;



template <typename _Derived>
constexpr bool is_contiguous() {
	using Derived = std::remove_const_t<_Derived>;
	static_assert( std::is_base_of_v<Eigen::DenseBase<Derived>, Derived> );
	using T = Eigen::internal::traits<Derived>;
	return
		Eigen::internal::traits<Derived>::InnerStrideAtCompileTime == 1 &&
		// (Derived::Flags & Eigen::LinearAccessBit);
		T::OuterStrideAtCompileTime == T::RowsAtCompileTime;
}


} // namespace Kokkidio

#endif