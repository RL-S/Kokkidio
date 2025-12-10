#ifndef KOKKIDIO_TYPEHELPERS_HPP
#define KOKKIDIO_TYPEHELPERS_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Kokkidio.hpp instead."
#endif

#include <type_traits>
#include <string_view>

namespace Kokkidio
{

template<class Src, class Dst>
using transcribe_const_t = std::conditional_t<
    std::is_const_v<std::remove_reference_t<Src>>, Dst const, Dst
>;
template<class Src, class Dst>
using transcribe_volatile_t = std::conditional_t<std::is_volatile_v<Src>, Dst volatile, Dst>;
template<class Src, class Dst>
using transcribe_cv_t = transcribe_const_t< Src, transcribe_volatile_t< Src, Dst> >;

template<typename T>
using remove_qualifiers = std::remove_cv_t<std::remove_reference_t<T>>;

/* Use this in constexpr branches that mustn't be called
 * to detect their instantiation, e.g. as
 * static_assert( dependent_false<T>::value, "Not valid for this type" ); */
template<typename T>
struct dependent_false : std::false_type {};


/**
 * @brief Returns non-mangled, human-readable type name.
 * Source: https://stackoverflow.com/a/56766138
 * 
 * @tparam T 
 * @return constexpr std::string_view 
 */
template <typename T>
constexpr std::string_view type_name()
{
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "std::string_view Kokkidio::type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr std::string_view Kokkidio::type_name() [with T = ";
    suffix = "; std::string_view = std::basic_string_view<char>]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "class std::basic_string_view<char,struct std::char_traits<char> > __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}


} // namespace Kokkidio

#endif
