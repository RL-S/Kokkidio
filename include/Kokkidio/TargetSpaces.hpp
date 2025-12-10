#ifndef KOKKIDIO_TARGETSPACES_HPP
#define KOKKIDIO_TARGETSPACES_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/EigenTypeHelpers.hpp"
#include "Kokkidio/typeHelpers.hpp"
#include "Kokkidio/typeAliases.hpp"
#include "Kokkidio/TargetEnum.hpp"

#include <Kokkos_Core.hpp>

namespace Kokkidio
{

namespace detail
{

template<Target targetArg>
struct ExecutionSpace {
	using Type =
		std::conditional_t<targetArg == Target::host,
			Kokkos::DefaultHostExecutionSpace,
			Kokkos::DefaultExecutionSpace
		>;
};

template<Target targetArg>
struct MemorySpace {
	using Type = typename ExecutionSpace<targetArg>::Type::memory_space;
};


template<typename _PlainObjectType, typename _MemorySpace>
struct ViewType {
public:
	using PlainObjectType = _PlainObjectType;
	using MemorySpace = _MemorySpace;
	using Scalar = transcribe_const_t<PlainObjectType, typename PlainObjectType::Scalar>;
	using P = PlainObjectType;
	static constexpr Index
		RowsAtCompileTime {P::RowsAtCompileTime},
		ColsAtCompileTime {P::ColsAtCompileTime},
		SizeAtCompileTime {P::SizeAtCompileTime};

private:
	/* Builtin array sizes must be greater than zero, and Eigen::Dynamic is -1. 
	 * These sizes only get used when they're greater than zero anyway,
	 * but at least nvcc didn't like negative array sizes,
	 * even in a std::conditional::false_type. */
	static constexpr Index
		Rows {RowsAtCompileTime > 0 ? RowsAtCompileTime : 1},
		Cols {ColsAtCompileTime > 0 ? ColsAtCompileTime : 1};

public:
	static constexpr bool IsFixedSizeAtCompileTime {SizeAtCompileTime != Eigen::Dynamic};
	using DataType = std::conditional_t<IsFixedSizeAtCompileTime,
		Scalar[Rows][Cols],
		Scalar**
	>;
	using Type = Kokkos::View<DataType, Kokkos::LayoutLeft, MemorySpace>;
};

template<Target targetArg>
inline constexpr Target ExecutionTarget { 
	std::is_same_v<
		Kokkos::DefaultExecutionSpace,
		Kokkos::DefaultHostExecutionSpace
	> ? Target::host : targetArg
};

inline constexpr Target DefaultTarget {
	std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace> ?
	Target::host : Target::device
};

template<typename Space>
inline constexpr Target spaceToTarget {
	std::is_same_v<Kokkos::DefaultHostExecutionSpace, Space> ?
	Target::host : DefaultTarget
};

} // namespace detail


template<Target targetArg>
inline constexpr Target ExecutionTarget { detail::ExecutionTarget<targetArg> };

constexpr inline Target DefaultTarget { detail::DefaultTarget };

template<typename Space>
inline constexpr Target spaceToTarget { detail::spaceToTarget<Space> };

template<Target targetArg>
using ExecutionSpace = typename detail::ExecutionSpace<targetArg>::Type;

template<Target targetArg>
using MemorySpace = typename detail::MemorySpace<targetArg>::Type;

template<typename PlainObjectType, typename MemorySpace>
using ViewType = typename detail::ViewType<PlainObjectType, MemorySpace>::Type;


} // namespace Kokkidio


#endif