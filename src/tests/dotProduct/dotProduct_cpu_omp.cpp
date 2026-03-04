// #define PRINT_COLS_PER_THREAD
#ifdef PRINT_COLS_PER_THREAD
#include <sstream>
#include <iostream>
#endif

#include "dotProduct.hpp"
#include <doNotOptimise.hpp>
#include <Kokkidio.hpp>

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

template<>
scalar dotProduct<host, K::cstyle_par>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	assert( m1.rows() == m2.rows() );
	assert( m1.cols() == m2.cols() );

	scalar result;

	for (int iter = 0; iter < iterations; ++iter){
		result = 0.0;
		const scalar
			*p1 {m1.data()},
			*p2 {m2.data()};
		Index
			cols {m1.cols()},
			rows {m1.rows()};

		/* this pattern with an intermediate thread-local variable 
		 * was tested to be faster than with direct 
		 * omp parallel for reduction (+:result). */
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			scalar thread_sum{0};
			KOKKIDIO_OMP_PRAGMA(for)
			for (int col = 0; col < cols; ++col){
			for (int row = 0; row < rows; ++row){
				thread_sum += p1[col * rows + row] * p2[col * rows + row];
			}}
			KOKKIDIO_OMP_PRAGMA(atomic)
			result += thread_sum;
		}
		doNotOptimise(result);
	}
	return result;
}


/* Parallel CPU calculation option 1*/

template<>
scalar dotProduct<host, K::eigen_par_colwise>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar result;

	for (int iter = 0; iter < iterations; ++iter){
		result = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto [colBeg, nCols] = ompSegment(m1.cols()).values;
			#ifdef PRINT_COLS_PER_THREAD
			if (iter == 0){
				std::stringstream sstr;
				sstr
					<< "Thread #" << omp_get_thread_num()
					<< ", colBeg=" << colBeg
					<< ", nCols=" << nCols
					<< '\n';
				#pragma omp for ordered
				for (int np = 0; np < omp_get_num_threads(); ++np){
					std::cout << sstr.str();
				}
			}
			#endif
			scalar thread_sum {0};
			for (int col = colBeg; col < colBeg + nCols; ++col){
				thread_sum += m1.col(col).dot(m2.col(col));
			}
			KOKKIDIO_OMP_PRAGMA(atomic)
			result += thread_sum;
		}
		doNotOptimise(result);
	}
	return result;
}

/* Parallel CPU calculation option 2 */

template<>
scalar dotProduct<host, K::eigen_par_arrProd>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar result;
	for (int iter = 0; iter < iterations; ++iter){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			KOKKIDIO_OMP_PRAGMA(single)
			{
				result = 0;
			}
			auto [colBeg, nCols] = ompSegment(m1.cols()).values;
			scalar thread_sum = { (
				m1.middleCols(colBeg, nCols).array() *
				m2.middleCols(colBeg, nCols).array()
			).sum() };
			KOKKIDIO_OMP_PRAGMA(atomic)
			result += thread_sum;
			/* if the iter loop is inside the parallel region,
			 * a barrier is necessary. */
			// KOKKIDIO_OMP_PRAGMA(barrier)
			
			// KOKKIDIO_OMP_PRAGMA( for reduction(+:result) )
			// for (int i=0; i<omp_get_num_threads(); ++i){
			// 	result += thread_sum;
			// }
		}
		doNotOptimise(result);
	}
	return result;
}

} // namespace Kokkidio::cpu
