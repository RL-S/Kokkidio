#ifndef KOKKIDIO_DUALVIEWMAP_HPP
#define KOKKIDIO_DUALVIEWMAP_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/ViewMap.hpp"

namespace Kokkidio
{



enum DualViewCopyOnInit {
	DontCopyToTarget = 0,
	CopyToTarget = 1,
};


template<typename _EigenType, Target targetArg = DefaultTarget>
class DualViewMap {
public:
	static constexpr Target target { ExecutionTarget<targetArg> };
	using EigenType_host = _EigenType;

	using ThisType = DualViewMap<EigenType_host, target>;
	using ViewMap_host   = ViewMap<EigenType_host, Target::host>;
	using ViewMap_target = ViewMap<EigenType_host, target>;
	using EigenType_target = typename ViewMap_target::EigenType_target;
	using Scalar = typename ViewMap_target::Scalar;

	using ViewType_host   = typename ViewMap_host  ::ViewType;
	using ViewType_target = typename ViewMap_target::ViewType;
	using ExecutionSpace_target = typename ViewMap_target::ExecutionSpace;

	using MapType_host   = typename ViewMap_host  ::MapType;
	using MapType_target = typename ViewMap_target::MapType;

	static_assert(
		is_owning_eigen_type<std::remove_const_t<EigenType_target>>::value ||
		is_eigen_map        <std::remove_const_t<EigenType_target>>::value
	);

protected:
	ViewMap_host   m_host;
	ViewMap_target m_target;

	void set(Index rows, Index cols){
		m_host = {rows, cols};
		/* When target and host are identical, 
		 * then we can copy-initialise the target View with the host View
		 * (or vice versa), to make them point to the same data.
		 */
		if constexpr (target == Target::host){
			m_target = {m_host};
		} else {
			m_target = {rows, cols};
		}
	}

public:
	DualViewMap(){
		/* For fixed size Eigen types,
		 * the default constructor allocates memory,
		 * like in Eigen itself. */
		using ET = EigenType_host;
		if constexpr( ET::SizeAtCompileTime != Eigen::Dynamic ){
			set(ET::RowsAtCompileTime, ET::ColsAtCompileTime);
		}
		/* For dynamically sized Eigen types, 
		 * the default constructor does nothing */
	}

	DualViewMap(
		EigenType_host& hostObj,
		DualViewCopyOnInit copyToTarget = CopyToTarget
	) :
		m_host  (hostObj),
		m_target(hostObj)
	{
		if ( copyToTarget ){
			this->copyToTarget();
		}
	}

	DualViewMap(Index rows, Index cols){
		set(rows, cols);
	}

	/* For Eigen vector types,
	 * we allow a single size parameter, like in Eigen itself. */
	DualViewMap(Index size) :
		/* DualViewMap(Index, Index) overwrites rows/cols 
		 * if they're known at compile time,
		 * so we could pass any numbers. */
		DualViewMap(size, size)
	{
		static_assert(EigenType_host::IsVectorAtCompileTime);
	}

	void assign( EigenType_host& hostObj ){
		this->m_host   = {hostObj};
		this->m_target = {hostObj};
	}

	void resize( Index rows, Index cols ){
		/* the logic here is analogous to DualViewMap::set */
		this->m_host.resize(rows, cols);
		if constexpr (target == Target::host){
			m_target = {m_host};
		} else {
			this->m_target.resize(rows, cols);
		}
	}

	void resize(Index size){
		static_assert(EigenType_host::IsVectorAtCompileTime);
		this->resize(size, size);
	}

	template<typename EigenObjOrView>
	void resizeLike(const EigenObjOrView& obj){
		this->resize( obj.rows(), obj.cols() );
	}

	KOKKOS_FUNCTION
	bool isAlloc_host() const {
		return this->m_host.isAlloc();
	}

	KOKKOS_FUNCTION
	bool isAlloc_target() const {
		return this->m_target.isAlloc();
	}

	KOKKOS_FUNCTION
	auto get_host() const -> ViewMap_host {
		return this->m_host;
	}

	KOKKOS_FUNCTION
	auto get_target() const -> ViewMap_target {
		return this->m_target;
	}

	template<Target _target>
	KOKKOS_FUNCTION
	auto get() const -> 
		std::conditional<_target == target, ViewMap_target, ViewMap_host>
	{
		if constexpr (_target == target){
			return this->get_target();
		} else {
			static_assert(_target == Target::host);
			return this->get_host();
		}
	}

	KOKKOS_FUNCTION
	auto view_host() const -> ViewType_host {
		assert( this->isAlloc_host() );
		return this->m_host.view();
	}

	KOKKOS_FUNCTION
	auto view_target() const -> ViewType_target {
		assert( this->isAlloc_target() );
		return this->m_target.view();
	}

	KOKKOS_FUNCTION
	auto view() const -> ViewType_target {
		return this->view_target();
	}

	template<Target _target>
	KOKKOS_FUNCTION
	auto view() const -> 
		std::conditional<_target == target, ViewType_target, ViewType_host>
	{
		if constexpr (_target == target){
			return this->view_target();
		} else {
			static_assert(_target == Target::host);
			return this->view_host();
		}
	}

	KOKKOS_FUNCTION
	auto map_host() const -> MapType_host {
		return this->m_host.map();
	}

	KOKKOS_FUNCTION
	auto map_target() const -> MapType_target {
		return this->m_target.map();
	}

	KOKKOS_FUNCTION
	auto map() const -> MapType_target {
		return this->map_target();
	}

	template<Target _target>
	KOKKOS_FUNCTION
	auto map() const -> 
		std::conditional<_target == target, MapType_target, MapType_host>
	{
		if constexpr (_target == target){
			return this->map_target();
		} else {
			static_assert(_target == Target::host);
			return this->map_host();
		}
	}

	KOKKOS_FUNCTION
	Index rows() const {
		return static_cast<Index>( this->view().extent(0) );
	}

	KOKKOS_FUNCTION
	Index cols() const {
		return static_cast<Index>( this->view().extent(1) );
	}

	KOKKOS_FUNCTION
	Index size() const {
		return static_cast<Index>( this->view().size() );
	}

	void copyToTarget(bool async = false){
		if constexpr ( target != Target::host ){
			printd( "Copying from host (n=%i) to target (n=%i)...\n"
				, static_cast<int>( this->view_host  ().size() )
				, static_cast<int>( this->view_target().size() )
			);
			if (async){
				Kokkos::deep_copy( ExecutionSpace_target{},
					this->view_target(), // dst
					this->view_host()    // src
				);
			} else {
				Kokkos::deep_copy(
					this->view_target(), // dst
					this->view_host()    // src
				);
			}
		} else {
			printd( "DualViewMap::copyToTarget, target==host, skipping...\n");
			assert( this->view_target().data() == this->view_host().data() );
			assert( this-> map_target().data() == this-> map_host().data() );
		}
	}

	void copyToHost(bool async = false){
		if constexpr ( target != Target::host ){
			printd( "Copying from target (n=%i) to host (n=%i)...\n"
				, static_cast<int>( this->view_target().size() )
				, static_cast<int>( this->view_host  ().size() )
			);
			if (async){
				Kokkos::deep_copy( ExecutionSpace_target{},
					this->view_host(),  // dst
					this->view_target() // src
				);
			} else {
				Kokkos::deep_copy(
					this->view_host(),  // dst
					this->view_target() // src
				);
			}
		} else {
			printd( "DualViewMap::copyToHost, target==host, skipping...\n");
			assert( this->view_target().data() == this->view_host().data() );
			assert( this-> map_target().data() == this-> map_host().data() );
		}
	}
};

template<typename T>
struct is_DualViewMap : std::false_type {};

template<typename EigenType, Target targetArg>
struct is_DualViewMap<DualViewMap<EigenType, targetArg>> : std::true_type {};

template<typename T>
inline constexpr bool is_DualViewMap_v = is_DualViewMap<T>::value;

template<Target target = DefaultTarget, typename EigenType>
std::enable_if_t<is_eigen_dense<remove_qualifiers<EigenType>>,
	DualViewMap<EigenType, target>
>
dualViewMap(
	EigenType& eigenObj,
	DualViewCopyOnInit copyToTarget = CopyToTarget
){
	return {eigenObj, copyToTarget};
}

#define KOKKIDIO_DUALMAPVIEW_FACTORY \
template<typename EigenType, Target target = DefaultTarget> \
DualViewMap<EigenType, target> dualViewMap

KOKKIDIO_DUALMAPVIEW_FACTORY(){ return {}; }
KOKKIDIO_DUALMAPVIEW_FACTORY(Index vectorSize){ return {vectorSize}; }
KOKKIDIO_DUALMAPVIEW_FACTORY(Index rows, Index cols){ return {rows, cols}; }

#undef KOKKIDIO_DUALMAPVIEW_FACTORY

} // namespace Kokkidio

#endif
