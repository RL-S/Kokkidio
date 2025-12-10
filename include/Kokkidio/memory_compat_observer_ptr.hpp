#ifndef KOKKIDIO_MSVC_COMPAT_OBSERVER_HPP
#define KOKKIDIO_MSVC_COMPAT_OBSERVER_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include <type_traits>
#include <functional>

namespace Kokkidio
{

/* the contents of this file were copied from gcc,
 * and only scope resolution operators 
 * for the std namespace were added */
using nullptr_t = std::nullptr_t;

template <typename _Tp>
class observer_ptr
{
public:
	// publish our template parameter and variations thereof
	using element_type = _Tp;
	using __pointer = std::add_pointer_t<_Tp>;            // exposition-only
	using __reference = std::add_lvalue_reference_t<_Tp>; // exposition-only

	// 3.2.2, observer_ptr constructors
	// default c'tor
	constexpr observer_ptr() noexcept
	: __t()
	{}

	// pointer-accepting c'tors
	constexpr observer_ptr(nullptr_t) noexcept
	: __t()
	{}

	constexpr explicit observer_ptr(__pointer __p) noexcept
	: __t(__p)
	{}

	// copying c'tors (in addition to compiler-generated copy c'tor)
	template <typename _Up,
	typename = typename std::enable_if<
		std::is_convertible<typename std::add_pointer<_Up>::type, __pointer
		>::value
	>::type>
	constexpr observer_ptr(observer_ptr<_Up> __p) noexcept
		: __t(__p.get())
	{}

	// 3.2.3, observer_ptr observers
	constexpr __pointer
	get() const noexcept
	{
		return __t;
	}

	constexpr __reference
	operator*() const
	{
		return *get();
	}

	constexpr __pointer
	operator->() const noexcept
	{
		return get();
	}

	constexpr explicit operator bool() const noexcept
	{
		return get() != nullptr;
	}

	// 3.2.4, observer_ptr conversions
	constexpr explicit operator __pointer() const noexcept
	{
		return get();
	}

	// 3.2.5, observer_ptr modifiers
	constexpr __pointer
	release() noexcept
	{
		__pointer __tmp = get();
		reset();
		return __tmp;
	}

	constexpr void
	reset(__pointer __p = nullptr) noexcept
	{
		__t = __p;
	}

	constexpr void
	swap(observer_ptr& __p) noexcept
	{
		std::swap(__t, __p.__t);
	}

private:
	__pointer __t;
}; // observer_ptr<>

template<typename _Tp>
void
swap(observer_ptr<_Tp>& __p1, observer_ptr<_Tp>& __p2) noexcept
{
	__p1.swap(__p2);
}

template<typename _Tp>
observer_ptr<_Tp>
make_observer(_Tp* __p) noexcept
{
	return observer_ptr<_Tp>(__p);
}

template<typename _Tp, typename _Up>
bool
operator==(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return __p1.get() == __p2.get();
}

template<typename _Tp, typename _Up>
bool
operator!=(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return !(__p1 == __p2);
}

template<typename _Tp>
bool
operator==(observer_ptr<_Tp> __p, nullptr_t) noexcept
{
	return !__p;
}

template<typename _Tp>
bool
operator==(nullptr_t, observer_ptr<_Tp> __p) noexcept
{
	return !__p;
}

template<typename _Tp>
bool
operator!=(observer_ptr<_Tp> __p, nullptr_t) noexcept
{
	return bool(__p);
}

template<typename _Tp>
bool
operator!=(nullptr_t, observer_ptr<_Tp> __p) noexcept
{
	return bool(__p);
}

template<typename _Tp, typename _Up>
bool
operator<(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return std::less<typename std::common_type<typename std::add_pointer<_Tp>::type,
					typename std::add_pointer<_Up>::type
					>::type
			>{}(__p1.get(), __p2.get());
}

template<typename _Tp, typename _Up>
bool
operator>(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return __p2 < __p1;
}

template<typename _Tp, typename _Up>
bool
operator<=(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return !(__p2 < __p1);
}

template<typename _Tp, typename _Up>
bool
operator>=(observer_ptr<_Tp> __p1, observer_ptr<_Up> __p2)
{
	return !(__p1 < __p2);
}

} // namespace Kokkidio

#endif