#ifndef KOKKIDIO_UTIL_HPP
#define KOKKIDIO_UTIL_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include "Kokkidio/typeHelpers.hpp"

namespace Kokkidio
{


/**
 * @brief In debug mode, asserts that the pointer is non-null.
 * Returns a reference to the pointee.
 * 
 * @param PTR 
 * @return *PTR (const or non-const reference)
 */
#define assertPtr(PTR) \
[](auto&& ptr) -> transcribe_const_t<decltype(ptr), decltype(*ptr)>{ \
	assert(ptr); \
	return *ptr; \
}(PTR)


} // namespace Kokkidio

#endif
