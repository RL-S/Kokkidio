#include <Kokkidio.hpp>
#include <iostream>
#include <string_view>

#include <magic_enum.hpp>


int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	using namespace Kokkidio;
	std::cout << "DefaultTarget: " << magic_enum::enum_name(DefaultTarget) <<'\n';

	int nRows {10}, nCols {20};

	/* existing Eigen object */
	Eigen::ArrayXXd eigenArray {nRows, nCols};

	/*********************************************************/
	/* Construction */
	/*********************************************************/

	/* By default, when initialising with an Eigen object,
	 * the object's data is copied to the target. 
	 * This behaviour be changed with an optional parameter: DontCopyToTarget */
	DualViewMap d1 {eigenArray};
	auto d2 = dualViewMap(eigenArray, DontCopyToTarget);
	/* Otherwise, a DualViewMap can be created in exactly the same ways as a 
	 * ViewMap, so please refer to ViewMap.cpp for more examples. */


	/*********************************************************/
	/* Using DualViewMap */
	/*********************************************************/
	/* with DualViewMap, you can set your values on host, 
	 * then copy them to the target: */
	d2.map_host() = 123;
	d2.copyToTarget();

	auto print = [&](std::string_view descriptor){
		std::cout
			<< "d2, values on host, " << descriptor << ":\n"
			<< d2.map_host() << '\n';
	};
	print("before");

	/* Now you can do some computations on the target, 
	 * then copy the values back */
	parallel_for(d2.cols(), KOKKOS_LAMBDA(ParallelRange<> rng){
		rng(d2) += 1;
	});
	d2.copyToHost();

	print("after");

	return 0;
}