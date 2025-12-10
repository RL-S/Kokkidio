#ifndef KOKKIDIO_PARALLELRANGE_CUDA_HPP
#define KOKKIDIO_PARALLELRANGE_CUDA_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

/* source: https://stackoverflow.com/a/11994395 (CC BY-SA 4.0)*/
// Make a FOREACH macro
#define FE_00(func)
#define FE_01(func, X) func(X) 
#define FE_02(func, X, ...) func(X), FE_01(func, __VA_ARGS__)
#define FE_03(func, X, ...) func(X), FE_02(func, __VA_ARGS__)
#define FE_04(func, X, ...) func(X), FE_03(func, __VA_ARGS__)
#define FE_05(func, X, ...) func(X), FE_04(func, __VA_ARGS__)
#define FE_06(func, X, ...) func(X), FE_05(func, __VA_ARGS__)
#define FE_07(func, X, ...) func(X), FE_06(func, __VA_ARGS__)
#define FE_08(func, X, ...) func(X), FE_07(func, __VA_ARGS__)
#define FE_09(func, X, ...) func(X), FE_08(func, __VA_ARGS__)
#define FE_10(func, X, ...) func(X), FE_09(func, __VA_ARGS__)
#define FE_11(func, X, ...) func(X), FE_10(func, __VA_ARGS__)
#define FE_12(func, X, ...) func(X), FE_11(func, __VA_ARGS__)
#define FE_13(func, X, ...) func(X), FE_12(func, __VA_ARGS__)
#define FE_14(func, X, ...) func(X), FE_13(func, __VA_ARGS__)
#define FE_15(func, X, ...) func(X), FE_14(func, __VA_ARGS__)
#define FE_16(func, X, ...) func(X), FE_15(func, __VA_ARGS__)
#define FE_17(func, X, ...) func(X), FE_16(func, __VA_ARGS__)
#define FE_18(func, X, ...) func(X), FE_17(func, __VA_ARGS__)
#define FE_19(func, X, ...) func(X), FE_18(func, __VA_ARGS__)
#define FE_20(func, X, ...) func(X), FE_19(func, __VA_ARGS__)
#define FE_21(func, X, ...) func(X), FE_20(func, __VA_ARGS__)
#define FE_22(func, X, ...) func(X), FE_21(func, __VA_ARGS__)
#define FE_23(func, X, ...) func(X), FE_22(func, __VA_ARGS__)
#define FE_24(func, X, ...) func(X), FE_23(func, __VA_ARGS__)
#define FE_25(func, X, ...) func(X), FE_24(func, __VA_ARGS__)
#define FE_26(func, X, ...) func(X), FE_25(func, __VA_ARGS__)
#define FE_27(func, X, ...) func(X), FE_26(func, __VA_ARGS__)
#define FE_28(func, X, ...) func(X), FE_27(func, __VA_ARGS__)
#define FE_29(func, X, ...) func(X), FE_28(func, __VA_ARGS__)
#define FE_30(func, X, ...) func(X), FE_29(func, __VA_ARGS__)
#define FE_31(func, X, ...) func(X), FE_30(func, __VA_ARGS__)
#define FE_32(func, X, ...) func(X), FE_31(func, __VA_ARGS__)
#define FE_33(func, X, ...) func(X), FE_32(func, __VA_ARGS__)
#define FE_34(func, X, ...) func(X), FE_33(func, __VA_ARGS__)
#define FE_35(func, X, ...) func(X), FE_34(func, __VA_ARGS__)
#define FE_36(func, X, ...) func(X), FE_35(func, __VA_ARGS__)
#define FE_37(func, X, ...) func(X), FE_36(func, __VA_ARGS__)
#define FE_38(func, X, ...) func(X), FE_37(func, __VA_ARGS__)
#define FE_39(func, X, ...) func(X), FE_38(func, __VA_ARGS__)
#define FE_40(func, X, ...) func(X), FE_39(func, __VA_ARGS__)
#define FE_41(func, X, ...) func(X), FE_40(func, __VA_ARGS__)
#define FE_42(func, X, ...) func(X), FE_41(func, __VA_ARGS__)
#define FE_43(func, X, ...) func(X), FE_42(func, __VA_ARGS__)
#define FE_44(func, X, ...) func(X), FE_43(func, __VA_ARGS__)
#define FE_45(func, X, ...) func(X), FE_44(func, __VA_ARGS__)
#define FE_46(func, X, ...) func(X), FE_45(func, __VA_ARGS__)
#define FE_47(func, X, ...) func(X), FE_46(func, __VA_ARGS__)
#define FE_48(func, X, ...) func(X), FE_47(func, __VA_ARGS__)
#define FE_49(func, X, ...) func(X), FE_48(func, __VA_ARGS__)
#define FE_50(func, X, ...) func(X), FE_49(func, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO( \
	_00, \
	_01, \
	_02, \
	_03, \
	_04, \
	_05, \
	_06, \
	_07, \
	_08, \
	_09, \
	_10, \
	_11, \
	_12, \
	_13, \
	_14, \
	_15, \
	_16, \
	_17, \
	_18, \
	_19, \
	_20, \
	_21, \
	_22, \
	_23, \
	_24, \
	_25, \
	_26, \
	_27, \
	_28, \
	_29, \
	_30, \
	_31, \
	_32, \
	_33, \
	_34, \
	_35, \
	_36, \
	_37, \
	_38, \
	_39, \
	_40, \
	_41, \
	_42, \
	_43, \
	_44, \
	_45, \
	_46, \
	_47, \
	_48, \
	_49, \
	_50, \
	NAME,...) NAME 

#define FOR_EACH(action,...) \
	GET_MACRO(	_00, __VA_ARGS__, \
	FE_50, \
	FE_49, \
	FE_48, \
	FE_47, \
	FE_46, \
	FE_45, \
	FE_44, \
	FE_43, \
	FE_42, \
	FE_41, \
	FE_40, \
	FE_39, \
	FE_38, \
	FE_37, \
	FE_36, \
	FE_35, \
	FE_34, \
	FE_33, \
	FE_32, \
	FE_31, \
	FE_30, \
	FE_29, \
	FE_28, \
	FE_27, \
	FE_26, \
	FE_25, \
	FE_24, \
	FE_23, \
	FE_22, \
	FE_21, \
	FE_20, \
	FE_19, \
	FE_18, \
	FE_17, \
	FE_16, \
	FE_15, \
	FE_14, \
	FE_13, \
	FE_12, \
	FE_11, \
	FE_10, \
	FE_09, \
	FE_08, \
	FE_07, \
	FE_06, \
	FE_05, \
	FE_04, \
	FE_03, \
	FE_02, \
	FE_01, \
	FE_00 \
	)(action,__VA_ARGS__)


#define KOKKIDIO_APPLY_FUNC(_APPLY_FUNC, ...) FOR_EACH(_APPLY_FUNC,__VA_ARGS__)

/* printout example */
#if 0

#define XSTR(...) STR(__VA_ARGS__)
#define STR(...) #__VA_ARGS__

#pragma message "The value of ABC: " XSTR( KOKKIDIO_APPLY_FUNC(rng, a, b, c) )

#endif


#endif