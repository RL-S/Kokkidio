#ifndef KOKKIDIO_MEMORY_HPP
#define KOKKIDIO_MEMORY_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#if defined(__clang__) || defined(_MSVC_VER)
    #include "memory_compat_observer_ptr.hpp"
#else

    #include <experimental/memory>

    namespace Kokkidio
    {
    using std::experimental::observer_ptr;
    using std::experimental::make_observer;

    } // namespace Kokkidio

#endif

#endif