#ifndef KOKKIDIO_MACROS_HPP
#define KOKKIDIO_MACROS_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif


/* tells the compiler that a part of the code is unreachable, e.g. when a 
 * switch covers all enum values. If the compiler doesn't support such a
 * function, a std::exception type is thrown upon reaching the supposedly
 * unreachable code. Make sure that the code is ACTUALLY unreachable! */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202302L) || __cplusplus >= 202302L)
	#include <utility>
	#define UNREACHABLE std::unreachable()
#else
	#if defined(__GNUC__) || defined(__clang__)
		#include <cassert>
		#define UNREACHABLE { \
			assert(false && "Unexpected branch!"); \
			__builtin_unreachable(); }
	#elif defined(_MSC_VER)
		#define UNREACHABLE __assume(false)
	#else
		#include <cassert>
		#define UNREACHABLE { \
			assert(false); }
	#endif
#endif

#define KOKKIDIO_TO_STRING(ARG) _KOKKIDIO_TO_STRING(ARG)
#define _KOKKIDIO_TO_STRING(ARG) #ARG

/* if KOKKIDIO_DEBUG_OUTPUT is defined, calls to printd(...) resolve to printf. */
// #define KOKKIDIO_DEBUG_OUTPUT

/* if KOKKIDIO_DEBUG_OUTPUT_LOOP is defined, printdl resolves to printd. */
// #define KOKKIDIO_DEBUG_OUTPUT_LOOP

#ifdef KOKKIDIO_DEBUG_OUTPUT_LOOP
#ifndef KOKKIDIO_DEBUG_OUTPUT
#define KOKKIDIO_DEBUG_OUTPUT
#endif
#define printdl printd
#warning "Printing to screen inside loops will considerably slow down execution."
#else
#define printdl(...)
#endif

#ifdef KOKKIDIO_DEBUG_OUTPUT
#include <cstdio>
/* Use for debug output outside of loops */
#define printd(...) printf(__VA_ARGS__)
#else
#define printd(...)
#endif



/* forces the compiler to inline a function */
#if defined __GNUC__ || __clang__
#define KOKKIDIO_INLINE __attribute__((always_inline)) inline
#else
#define KOKKIDIO_INLINE inline
#endif

/* IntelLLVM (icpx) doesn't seem to define _OPENMP 
 * when passing -fiopenmp/-qopenmp, 
 * but we pass KOKKIDIO_OPENMP from CMake when linking to OpenMP.
 * However, because something breaks inside icpx' "omp.h" 
 * when _OPENMP is defined before including it (i.e. the normal way...),
 * we instead only use KOKKIDIO_OPENMP as a replacement.
 * The error is a conflicting declaration of omp_is_initial_device:
 * /opt/intel/oneapi/compiler/2024.1/bin/compiler/../../opt/compiler/include/omp.h:533:23: 
 * error: static declaration of 'omp_is_initial_device' follows non-static declaration
 * 533 |     static inline int omp_is_initial_device(void) { return 1; }
 *     |                       ^
 * /opt/intel/oneapi/compiler/2024.1/bin/compiler/../../opt/compiler/include/omp.h:135:40: 
 * note: previous declaration is here
 * 135 |     extern int  __KAI_KMPC_CONVENTION  omp_is_initial_device (void);
 * */
// #undef KOKKIDIO_OPENMP
#if defined(_OPENMP) && !defined(KOKKIDIO_OPENMP)
#define KOKKIDIO_OPENMP
#endif

/* use KOKKIDIO_OMP_PRAGMA(CLAUSES) instead of #ifdef _OPENMP #pragma omp CLAUSES #endif
 * The "stringification" is carried out inside the macro,
 * so you must not use strings to call it.
 */
#ifdef KOKKIDIO_OPENMP
#include <omp.h>
/* as a precaution, we want that people can still query #ifdef _OPENMP,
 * but we have to do that after including <omp.h>, because of *sigh* icpx... */
#ifndef _OPENMP
/* KMP_VERSION_BUILD seems to be what ICPX uses for its OpenMP version symbol */
#if defined(__INTEL_LLVM_COMPILER) && defined(KMP_VERSION_BUILD)
#define _OPENMP KMP_VERSION_BUILD
#endif
#endif

#define KOKKIDIO_PRAGMA_STRINGIFIER(ARG) _Pragma(#ARG)
#define KOKKIDIO_OMP_PRAGMA(...) KOKKIDIO_PRAGMA_STRINGIFIER(omp __VA_ARGS__)
#else
// #ifndef KOKKIDIO_SUPPRESS_IGNORED_OMP_WARNING
#define KOKKIDIO_OMP_PRAGMA(...) printd( \
	"In " __FILE__ ":%i" \
	": Ignoring OpenMP clause \"" KOKKIDIO_TO_STRING(__VA_ARGS__) "\"\n", __LINE__ );
#endif


#ifdef KOKKIDIO_OPENMP
#define KOKKIDIO_OMP_PARALLEL_IF_NOT(CONDITION,OTHER_CLAUSES, ...) \
if ( CONDITION ){ \
	KOKKIDIO_OMP_PRAGMA(OTHER_CLAUSES) \
		__VA_ARGS__ \
} else { \
	KOKKIDIO_OMP_PRAGMA(parallel OTHER_CLAUSES) \
		__VA_ARGS__ \
}
#else
#define KOKKIDIO_OMP_PARALLEL_IF_NOT(CONDITION,OTHER_CLAUSES, ...) __VA_ARGS__
#endif


#ifdef KOKKIDIO_OPENMP
#define KOKKIDIO_OMP_CLAUSE_IF_PARALLEL(CLAUSE, ...) \
if ( omp_in_parallel() ){ \
	KOKKIDIO_OMP_PRAGMA(CLAUSE) \
	{ __VA_ARGS__ } \
} else { __VA_ARGS__ }
#else
#define KOKKIDIO_OMP_CLAUSE_IF_PARALLEL(CLAUSE, ...) __VA_ARGS__
#endif


#ifdef KOKKIDIO_OPENMP
#define KOKKIDIO_OMP_CLAUSE_IF_NOT_PARALLEL(CLAUSE, ...) \
if ( omp_in_parallel() ){ \
	__VA_ARGS__ \
} else { \
	KOKKIDIO_OMP_PRAGMA(CLAUSE) \
	{ __VA_ARGS__ } \
}
#else
#define KOKKIDIO_OMP_CLAUSE_IF_NOT_PARALLEL(...) __VA_ARGS__
#endif


#endif
