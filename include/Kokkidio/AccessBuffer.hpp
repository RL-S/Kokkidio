#ifndef KOKKIDIO_ACCESSBUFFER_HPP
#define KOKKIDIO_ACCESSBUFFER_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/DualViewMap.hpp"
#include "Kokkidio/ViewMap.hpp"

namespace Kokkidio
{



/**
 * @brief When many write accesses to the same address are needed in device code,
 * a local variable would be the intuitive solution. 
 * This would inhibit host code vectorisation, 
 * so this class provides a unified abstraction of more optimal patterns.
 * AccessBuffer::get() can be used in place of the EigenRange + ViewMap combination 
 * (EigenRange<...>::operator() (ViewMap<...>)), 
 * so long as after the write accesses, AccessBuffer::write() is called.
 * In device code, AccessBuffer::get() returns a local fixed-size object
 * (an Eigen column vector with size 1 by default), whose values are transferred 
 * to the ViewMap range when calling AccessBuffer::write().
 * In host code, AccessBuffer::get() instead just returns the ViewMap range,
 * and AccessBuffer::write() does nothing. 
 * This relies on caching while skipping an unnecessary copy operation.
 * 
 * Works with ViewMap and DualViewMap.
 * 
 * To instantiate this class, use the factory function
 * 
 * @see make_AccessBuffer<ColType = Array1s, ViewMapType>(
 *   const ViewMapType&, 
 *   const EigenRange<ViewMapType::target>&
 * ).
 * 
 * 
 * 
 * @tparam _ViewMapType is the ViewMap or DualViewMap type
 * @tparam _ColType is the fixed-size column vector type 
 * matching the shape that the _ViewMapType takes in a device kernel.
 * Examples:
 * _ViewMapType is ViewMap<Array3Xs> -> _ColType should be Array3s. 
 * _ViewMapType is ViewMap<ArrayXs>  -> _ColType should be Array1s.
 * If _ViewMapType has a fully dynamic type,
 * but your algorithm still uses a fixed number of dimensions (rows),
 * use that number.
 */
template<typename _ViewMapType, typename _ColType = Array1s>
class AccessBuffer {
public:
	using ViewMapType = _ViewMapType;
	static_assert( is_ViewMap_v<ViewMapType> || is_DualViewMap_v<ViewMapType> );

	using ColType = _ColType;
	static_assert( is_eigen_dense<ColType> );
	static_assert( ColType::ColsAtCompileTime == 1 );
	static_assert( ColType::RowsAtCompileTime != Eigen::Dynamic );

	static constexpr Target target {ViewMapType::target};
	static constexpr bool isHost {target == Target::host};

	class Empty {};
	using Storage = std::conditional_t<isHost, Empty, ColType>;

private:
	/* can't use observer_ptr here, 
	 * because make_observer is not a __device__ function,
	 * and this class gets instantiated inside kernels. */
	const ViewMapType* m_obj {nullptr};
	const EigenRange<target>* m_rng {nullptr};
	Storage m_store;

	KOKKOS_FUNCTION
	auto viewmap() -> const ViewMapType& {
		return assertPtr(m_obj);
	}

	KOKKOS_FUNCTION
	auto range() -> const EigenRange<target>& {
		return assertPtr(m_rng);
	}

public:

	KOKKOS_FUNCTION
	AccessBuffer(const _ViewMapType& obj, const EigenRange<target>& rng)
	{
		this->m_obj = &obj;
		this->m_rng = &rng;
		printdl(
			"AccessBuffer ctor. Viewmap size: %ix%i, "
			"Range: [%i,%i) -> size: %i\n"
			, this->viewmap().rows()
			, this->viewmap().cols()
			, this->range().asIndexRange().start()
			, this->range().asIndexRange().end()
			, this->range().size()
		);
	}

	KOKKOS_FUNCTION
	decltype(auto) get() {
		if constexpr (isHost){
			return this->range()( this->viewmap() );
		} else {
			return (m_store); // parentheses for adding lvalue reference
		}
	}

	KOKKOS_FUNCTION
	void write() {
		if constexpr (isHost){
			printdl("AccessBuffer, on host: Skipping write().\n");
			; // do nothing
		} else {
			printdl("AccessBuffer, on device: Writing back to viewmap.\n");
			this->range()( this->viewmap() ) = m_store;
		}
	}
};

template<typename ColType = Array1s, typename ViewMapType>
KOKKOS_FUNCTION
auto make_AccessBuffer(
	const ViewMapType& obj, const EigenRange<ViewMapType::target>& rng
)
	-> AccessBuffer<ViewMapType, ColType>
{
	return {obj, rng};
}

} // namespace Kokkidio


#endif
