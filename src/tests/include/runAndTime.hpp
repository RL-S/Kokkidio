#ifndef KOKKIDIO_RUN_AND_TIME_HPP
#define KOKKIDIO_RUN_AND_TIME_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <sstream>
#include <chrono>
#include <string_view>
#include <regex>

#include "funcWrapper.hpp"
#include <magic_enum.hpp>
#include "Kokkidio/TargetEnum.hpp"

namespace Kokkidio
{


template<typename T>
struct ReturnWrapper {
private:
	static constexpr bool _isVoid { std::is_same_v<T, void> };
public:
	constexpr bool isVoid() const { return _isVoid; }
	using Type = std::conditional_t<_isVoid, int, T>;
	Type value;
};


template<typename Callable>
using ReturnTypeOf = 
	typename decltype(std::function{std::declval<Callable>()})::result_type;


template<typename Func, typename ... Ts>
ReturnWrapper<ReturnTypeOf<Func>> getResult(Func&& func, Ts&& ... args){
	ReturnWrapper<ReturnTypeOf<Func>> ret;
	if constexpr ( !ret.isVoid() ){
		ret.value = func( std::forward<Ts>(args) ... );
	} else {
		func( std::forward<Ts>(args) ... );
		ret.value = 0;
	}
	return ret;
}


template<typename Check, typename Ret>
bool pass(Check&& check, const ReturnWrapper<Ret>& ret){
	static_assert( std::is_same_v<ReturnTypeOf<Check>, bool> );
	if constexpr ( std::is_invocable_r_v<bool, Check, Ret> ){
		return check(ret.value);
	} else {
		static_assert( std::is_invocable_r_v<bool, Check>,
			"The check function must either be invocable with "
			"the benchmark function's return value, "
			"or take no arguments."
		);
		return check();
	}
}




template<typename Func, typename Check, typename ... Ts>
void runAndTime_single(
	Func&& func,
	Target target,
	std::string comment,
	Check&& check,
	bool gnuplot,
	int runNo,
	Ts&& ... args
){
	using T = Target;
	auto targetStr = [](T t) -> std::string {
		if ( t == T::host ){
			return "CPU";
		} else
		if ( t == T::device ){
			return "GPU";
		}
		return {};
	};

	std::stringstream plotStr;
	// static int plotNo {0};
	auto addPlot = [&](const std::string& comment, auto val){
		if ( comment.find("warmup") != std::string::npos ){
			return;
		}
		auto xlabel = std::regex_replace(comment, std::regex("_"), "-");
		plotStr << runNo << '\t' << xlabel << '\t' << val << '\n';
		// ++plotNo;
	};

	std::string descriptor {
		targetStr(target) + "--" + comment
	};
	if (!gnuplot){
		std::cout << "Run: " << descriptor << "\n";
	}

	auto now = [](){ return std::chrono::high_resolution_clock::now(); };

	auto start = now();
	auto ret { getResult( func, std::forward<Ts>(args) ... ) };
	auto end = now();
	// auto time { (end - start).count() };
	std::chrono::duration<double> elapsed { end - start };

	if ( !pass(std::forward<Check>(check), ret) ){
		std::cerr << "\nFailed test! Test: \"" << descriptor << "\"\n";
		exit(EXIT_FAILURE);
	}
	if (gnuplot){
		addPlot( descriptor, elapsed.count() );
	} else {
		std::cout << "\tComputation time: " << elapsed.count() << " seconds.\n";
	}

	if (gnuplot){
		std::cout << plotStr.str();
	}
}


struct RunOpts {
	std::string groupComment;
	bool skipWarmup {false};
	bool useGnuplot {false};
	int  runNo {0};
};

template<
template<Target, typename ImplEnum, ImplEnum, typename ...>typename FuncWrapper,
 Target target, typename ImplEnum, ImplEnum ... impls, typename Check, typename ... Ts>
void runAndTime(
	RunOpts& opts,
	Check&& check, 
	Ts&& ... args
){
	using magic_enum::enum_name;
	std::size_t maxLen {0};
	([&]{
		maxLen = std::max( maxLen, enum_name(impls).length() );
	}(), ...);
	maxLen += opts.groupComment.length() + 1; // +1 for the "-"
	make_func<FuncWrapper> func;

	int i{0}, nWarmups { target == Target::host ? 1 : 2 };
	([&]{
		std::string comment {opts.groupComment + "-"};
		int repetitions {1};
		if (i == 0 && !opts.skipWarmup){
			comment += "warmup";
			repetitions = nWarmups;
		} else {
			comment += std::string{enum_name(impls)};
			comment += std::string( maxLen - comment.length(), ' ' );
		}
		for (int r=0; r<repetitions; ++r){
			runAndTime_single(
				func.template wrap<target, ImplEnum, impls>(
					std::forward<Ts>(args) ...
				),
				target,
				comment,
				std::forward<Check>(check),
				opts.useGnuplot,
				opts.runNo
			);
		}
		if (i != 0 || opts.skipWarmup){
			++opts.runNo;
		}
		++i;
	}(), ...);
}


} // namespace Kokkidio


#endif
