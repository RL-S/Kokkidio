#include <Kokkidio.hpp>
#include <iostream>
#include <chrono>

#include <magic_enum.hpp>

struct Timer {
	using clk = std::chrono::high_resolution_clock;
	clk::time_point beg, end;
	double& result;

	Timer(double& arg) :
		result {arg}
	{}

	void start(){
		result = 0;
		beg = clk::now();
	}

	void report(){
		end = clk::now();
		std::chrono::duration<double> elapsed { end - beg };
		std::cout
			<< "Result: " << result
			<< ", time: " << elapsed.count() << "s\n";
	}
};

int main(){
	int nRows {4}, nCols {1000000};
	Eigen::MatrixXd a {nRows, nCols}, b;
	b.resizeLike(a);
	/* fill matrices in some way ... */
	a.setRandom();
	b.setRandom();

	double result; // let's sum up the results to not need another array

	Timer t(result);

	/* One could do a nested loop and manually implement the dot product.
	 * We skip that here, because for that you wouldn't use Eigen */

	/* Instead, one could either distribute individual column-multiplications, 
	 * as one might do on a GPU, if nCols >> nRows */
	t.start();
	for (int i=0; i<nCols; ++i){
		result += a.col(i).transpose() * b.col(i);
	}
	t.report();

	t.start();
	for (int i=0; i<nCols; ++i){
		result += a.col(i).dot( b.col(i) );
	}
	t.report();

	/* Or, one could distribute blocks of the matrices to threads and let Eigen
	 * handle the loop over columns, as may be preferable on a CPU.
	 * This can be faster, as it allows Eigen to vectorise the operation. */
	t.start();
	int nCores {4}; // just for illustration
	int nColsPerCore {nCols / nCores}; // not handling remainders
	for (int i=0; i<nCores; ++i){
		int firstCol {i * nColsPerCore};
		result += (
				a.middleCols(firstCol, nColsPerCore).transpose() * 
				b.middleCols(firstCol, nColsPerCore)
			).trace(); // sum of the diagonal
	}
	t.report();

	return 0;
}