#ifndef KOKKIDIO_DONOTOPTIMISE_HPP
#define KOKKIDIO_DONOTOPTIMISE_HPP

namespace Kokkidio
{

/* Copied (modified) from Google Benchmark Lib (Apache 2.0 licence).
 * Original comment:
 * The DoNotOptimize(...) function can be used to prevent a value or
 * expression from being optimized away by the compiler. This function is
 * intended to add little to no overhead.
 * See: https://youtu.be/nXaxk27zwlk?t=2441
 * */
template <class Tp>
KOKKIDIO_INLINE void doNotOptimise(Tp const& value) {
	asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
KOKKIDIO_INLINE void doNotOptimise(Tp& value) {
#if defined(__clang__)
	asm volatile("" : "+r,m"(value) : : "memory");
#else
	asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/* Google Benchmark also has some MSVC section, which may be added if needed. */

} // namespace Kokkidio


#endif