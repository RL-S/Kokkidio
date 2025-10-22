#ifndef KOKKIDIO_PARSE_OPTS_HPP
#define KOKKIDIO_PARSE_OPTS_HPP

#include "CLI11.hpp"
#include <optional>

namespace Kokkidio
{

struct BenchOpts {
	std::string
		target {"all"},
		group  {"all"},
		impl   {"all"};
	long
		nRuns {1},
		nRows {1},
		nCols {512};
	bool
		gnuplot {false},
		skipWarmup {false};
};

inline void doNothing(CLI::App&){}

namespace detail
{

template<typename ImplEnum>
void appendEnumNames(std::string& enumVals){
	magic_enum::enum_for_each<ImplEnum>([&](auto iter){
		constexpr ImplEnum enumVal = iter;
		enumVals += "\n    ";
		enumVals += magic_enum::enum_name(enumVal);
	});
}

} // namespace detail


template<typename ... ImplEnum>
bool checkImpl(const BenchOpts& opts){
	if (opts.impl == "all"){
		return true;
	}
	bool castSuccess {false};
	([&]{
		if (castSuccess){
			return;
		}
		castSuccess = magic_enum::enum_cast<ImplEnum>(opts.impl).has_value();
	}(), ...);
	if (!castSuccess){
		std::string enumVals;
		([&]{
			detail::appendEnumNames<ImplEnum>(enumVals);
		}(), ...);
		std::cerr
			<< "Unknown implementation \"" 
			<< opts.impl 
			<< "\". Valid options are:"
			<< enumVals
			<< '\n';
	}
	return castSuccess;
}

template<typename Func = void(*)(CLI::App&)>
std::optional<int> parseOpts(
	BenchOpts& opts,
	int argc, char** argv,
	Func&& parseExtra = doNothing
){
	CLI::App app {"Runner for Kokkidio benchmarks"};
	argv = app.ensure_utf8(argv);

	app.add_option(
		"-t,--target", opts.target, "Which target to run on (cpu|gpu|all/both)"
	)->check(
		CLI::IsMember( {"cpu", "gpu", "all", "both"}, CLI::ignore_case )
	);

	std::vector<long> size;
	app.add_option(
		"-s,--size", size,
		"The number of elements to use. "
		"If one argument is provided, it is used as the number of columns. "
		"If two arguments are provided, they are interpreted as rows x cols."
	)->expected(1,2)->check(CLI::PositiveNumber);

	app.add_option(
		"-r,--runs", opts.nRuns, "The number of repetitions"
	)->check(CLI::PositiveNumber);

	app.add_flag(
		"-p,--gnuplot", opts.gnuplot,
		"Whether to format output for piping to gnuplot"
	);

	app.add_flag(
		"-n,--noWarmup", opts.skipWarmup, "Skip warmup runs."
	);

	app.add_option(
		"-g,--group", opts.group,
		"Choose implementation group (\"native\"/\"unified\"), or \"all\"."
	)->check(
		CLI::IsMember( {"native", "unified", "all"}, CLI::ignore_case )
	);

	app.add_option(
		"-i,--impl", opts.impl,
		"Choose specific implementation by name, or \"all\"."
	)->ignore_case();

	parseExtra(app);

	CLI11_PARSE(app, argc, argv);

	if ( size.size() > 1 ){
		opts.nRows = size[0];
		opts.nCols = size[1];
	} else
	if ( size.size() > 0 ){
		opts.nCols = size[0];
	}

	if (!opts.gnuplot){
		std::cout << "Using "
			<< opts.nRows << " rows, "
			<< opts.nCols << " columns, "
			<< opts.nRuns << " iterations, "
			<< "\n"
			<< "target: \"" << opts.target << "\", "
			<< "group: \"" << opts.group << "\", "
			<< "and implementation: \"" << opts.impl << "\""
			<< ".\n";
	}

	return std::nullopt;
}

} // namespace Kokkidio

#endif
