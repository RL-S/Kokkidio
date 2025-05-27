#include <Kokkidio.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <magic_enum.hpp>

template<typename Real, typename ViewMapType>
void printFirstN(
	int nPrint,
	Real a,
	const ViewMapType& x,
	const ViewMapType& y,
	const ViewMapType& z
);

void axpy_kokkidio(){
	using namespace Kokkidio;
	float a {0.1};
	int size {100};
	using FloatArray = DualViewMap<Eigen::ArrayXf>;
	FloatArray x {size}, y {size}, z {size};
	/* fill arrays in some way, which is often easier on the host */
	x.map_host().setRandom();
	y.map_host().setRandom();
	/* then copy the arrays to your compute device (e.g. still host, or a GPU) */
	x.copyToTarget();
	y.copyToTarget();
	/* and now do the computation on that target */
	parallel_for( size, KOKKOS_LAMBDA(ParallelRange<> rng){
		rng(z) = a * rng(x) + rng(y);
	});
	/* After the computation, you may copy the results back to the host */
	z.copyToHost();

	printFirstN(20, a, x, y, z);
}

void axpy_eigen(){
	Eigen::Index size {10};
	float a {0.5};
	Eigen::VectorXf x, y, z;
	auto set = [&](auto& arr){
		arr.resize(size);
		arr.setRandom();
	};
	set(x);
	set(y);
	set(z);

	/* One could either distribute individual SAXPY operations, 
	 * as one might do on a GPU */
	for (int i=0; i<size; ++i){
		z(i) = a * x(i) + y(i);
	}

	/* Or, one could distribute blocks of the arrays to threads and let Eigen
	 * handle the loop over elements, as may be preferable on a CPU.
	 * This can be a lot faster, as it allows Eigen to vectorise the operation. */
	int nCores {4}; // just for illustration
	int sizePerCore {size / nCores}; // not handling remainders

	for (int i=0; i<nCores; ++i){
		int first {i * sizePerCore};
		z.segment(first, sizePerCore) =
			a *
			x.segment(first, sizePerCore) +
			y.segment(first, sizePerCore);
	}
}

int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	using namespace Kokkidio;
	std::cout << "DefaultTarget: " << magic_enum::enum_name(DefaultTarget) <<'\n';

	axpy_kokkidio();

	return 0;
}



template<typename Real, typename ViewMapType>
void printFirstN(
	int nPrint,
	Real a,
	const ViewMapType& x,
	const ViewMapType& y,
	const ViewMapType& z
){
	int size {x.size()};
	nPrint = std::min(nPrint, size);
	std::stringstream str;
	str << std::fixed;
	for (int i{0}; i<nPrint; ++i){
		auto insNum = [&](auto num){
			str << std::setprecision(5) <<  std::setw(8) << num;
		};
		auto getVal = [&](const auto& arr){ return arr.map_host()(i); };
		auto insVal = [&](const auto& arr){ insNum( getVal(arr) ); };
		str << std::setprecision(1) << a << " * ";
		insVal(x);
		str << " + ";
		insVal(y);
		str << " = ";
		insVal(z);
		str << ", expected: ";
		insNum( a * getVal(x) + getVal(y) );
		str << '\n';
	}
	if (nPrint < size){
		str << "...\n";
	}
	std::cout << "Check:\n" << str.str();
}

