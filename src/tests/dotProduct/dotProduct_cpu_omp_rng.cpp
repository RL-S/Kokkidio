// #define PRINT_COLS_PER_THREAD
#ifdef PRINT_COLS_PER_THREAD
#include <sstream>
#include <iostream>
#endif

#include "dotProduct.hpp"
#include "doNotOptimise.hpp"
#include <Kokkidio.hpp>

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

/* Parallel CPU calculation option 1*/

template<>
scalar dotProduct<host, K::eigen_par_colwise_ranged>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	volatile scalar finalResult_cpu;
	for (volatile int iter = 0; iter < iterations; ++iter){
		finalResult_cpu = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto rng = ParallelRange<Target::host>(m1.cols(), 0);
			#ifdef PRINT_COLS_PER_THREAD
			if (iter == 0){
				std::stringstream sstr;
				sstr
					<< "Thread #" << omp_get_thread_num()
					<< ", colBeg=" << rng.get().start
					<< ", nCols=" << rng.get().size
					<< '\n';
				KOKKIDIO_OMP_PRAGMA(for ordered)
				for (int np = 0; np < omp_get_num_threads(); ++np){
					std::cout << sstr.str();
				}
			}
			#endif
			scalar thread_sum {0};
			rng.for_each([&](int col){
				thread_sum += m1.col(col).dot(m2.col(col));
			});
			// #pragma omp for reduction(+ : finalResult_cpu)
			// for (int np = 0; np < omp_get_num_threads(); ++np){
			// 	finalResult_cpu += sum;
			// }
			KOKKIDIO_OMP_PRAGMA(atomic)
			finalResult_cpu += thread_sum;
		}
	}
	return finalResult_cpu;
}

/* Parallel CPU calculation option 2 */
template<>
scalar dotProduct<host, K::eigen_par_arrProd_ranged>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	volatile scalar finalResult_cpu;
	for (volatile int iter = 0; iter < iterations; ++iter){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			KOKKIDIO_OMP_PRAGMA(single)
			{
				finalResult_cpu = 0;
			}
			auto rng = ParallelRange<Target::host>(m1.cols(), 0);
			scalar thread_sum = { ( rng(m1).array() * rng(m2).array() ).sum() };
			// KOKKIDIO_OMP_PRAGMA( for reduction(+ : finalResult_cpu) )
			// for (int np = 0; np < omp_get_num_threads(); ++np){
			// 	finalResult_cpu += sum;
			// }
			KOKKIDIO_OMP_PRAGMA(atomic)
			finalResult_cpu += thread_sum;
		}
	}
	return finalResult_cpu;
}

} // namespace Kokkidio::cpu
