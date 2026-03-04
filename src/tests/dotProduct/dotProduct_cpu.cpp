#include "dotProduct.hpp"
#include "doNotOptimise.hpp"

#include <cassert>

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

/* Sequential CPU calculation */
template<>
scalar dotProduct<host, K::cstyle_seq>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar result {0};
	for (int iter = 0; iter < iterations; ++iter){
		result = 0;
		const scalar
			*p1 {m1.data()},
			*p2 {m2.data()};
		Index
			cols {m1.cols()},
			rows {m1.rows()};

		for (int col = 0; col < cols; ++col){
		for (int row = 0; row < rows; ++row){
			result += p1[col * rows + row] * p2[col * rows + row];
		}}
		doNotOptimise(result);
	}
	return result;
}

// template<>
// scalar dotProduct<host, K::seq_manual>(
// 	const MatrixXs& m1, const MatrixXs& m2, int iterations
// ){
// 	scalar result {0};
// 	for (int iter = 0; iter < iterations; ++iter){
// 		result = 0;
// 		for (int col = 0; col < m1.cols(); ++col){
// 		for (int row = 0; row < m1.rows(); ++row){
// 			result += m1(row, col) * m2(row, col);
// 		}}
//		doNotOptimise(result);
// 	}
// 	return result;
// }

template<>
scalar dotProduct<host, K::eigen_seq_colwise>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar result {0};
	for (int iter = 0; iter < iterations; ++iter){
		result = 0;
		for (int col = 0; col < m1.cols(); ++col){
			result += m1.col(col).dot(m2.col(col));
		}
		doNotOptimise(result);
	}
	return result;
}

template<>
scalar dotProduct<host, K::eigen_seq_arrProd>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar result {0};
	for (int iter = 0; iter < iterations; ++iter){
		result = ( m1.array() * m2.array() ).sum();
		doNotOptimise(result);
	}
	return result;
}

}   // namespace Kokkidio::cpu
