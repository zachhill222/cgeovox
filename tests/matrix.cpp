#include "util/matrix.hpp"


template<int n, int m, typename Scalar_t>
gv::util::Matrix<n,m,Scalar_t> create_matrix()
{
	gv::util::Matrix<n,m,Scalar_t> matrix;

	Scalar_t count=0;
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<m; j++)
		{
			matrix.at(i,j) = count;
			count += (Scalar_t) 1;
		}
	}
	return matrix;
}

template<int n, typename Scalar_t>
gv::util::Matrix<n,n,Scalar_t> create_upper()
{
	gv::util::Matrix<n,n,Scalar_t> matrix;

	for (int i=0; i<n; i++)
	{
		for (int j=i; j<n; j++)
		{
			matrix(i,j) = i+j+1;
		}
	}
	return matrix;
}


template<int n, typename Scalar_t>
gv::util::Matrix<n,n,Scalar_t> create_lower()
{
	gv::util::Matrix<n,n,Scalar_t> matrix;

	for (int i=0; i<n; i++)
	{
		for (int j=0; j<=i; j++)
		{
			matrix(i,j) = i+j+1;
		}
	}
	return matrix;
}



int main(int argc, char* argv[])
{
	const int n=3;
	const int m=3;
	const int p=3;

	using Scalar_t = double;

	//create and print matrix
	gv::util::Matrix<n,m,Scalar_t> M = create_matrix<n,m,Scalar_t>();
	std::cout << "M=\n" << M << std::endl;

	gv::util::Matrix<m,p,Scalar_t> A(1);
	std::cout << "A=\n" << A << std::endl;

	//create and print vectors
	gv::util::Point<n,Scalar_t> u(1);
	gv::util::Point<m,Scalar_t> v(1);

	std::cout << "u= " << u << std::endl;
	std::cout << "v= " << v << std::endl;

	//multiply matrix and vectors
	std::cout << "M*v= " << M*v << std::endl;
	std::cout << "u*M= " << u*M << std::endl;

	//multiply matrices
	std::cout << "M*A=\n" << M*A << std::endl;

	//multiply scalar with matrix
	std::cout << "2*M=\n" << 2.0*M << std::endl;

	//add matrices
	std::cout << "M+M=\n" << M+M << std::endl;


	//make upper triangular matrix
	gv::util::Matrix<m,m,Scalar_t> U = create_upper<m,Scalar_t>();
	std::cout << "U=\n" << U << std::endl;

	gv::util::Point<m,Scalar_t> w = gv::util::solve_upper(U,v);
	std::cout << "w:= solve_upper(U,v)= " << w << std::endl;
	std::cout << "U*w= " << U*w << std::endl;

	//make lower triangular matrix
	gv::util::Matrix<m,m,Scalar_t> L = create_lower<m,Scalar_t>();
	std::cout << "L=\n" << L << std::endl;

	w = gv::util::solve_lower(L,v);
	std::cout << "w:= solve_upper(L,v)= " << w << std::endl;
	std::cout << "L*w= " << L*w << std::endl;


	//test QR
	gv::util::Matrix<n,m,Scalar_t> Q;
	gv::util::Matrix<m,m,Scalar_t> R;
	gv::util::partialQR(L,Q,R);

	std::cout << "L=\n"    << L << std::endl;
	std::cout << "Q=\n"    << Q << std::endl;
	std::cout << "R=\n"    << R << std::endl;
	std::cout << "QR=\n"   << Q*R << std::endl;

	std::cout << "Q.tr()*Q=\n" << Q.tr()*Q << std::endl;
	std::cout << "Q*Q.tr()=\n" << Q.tr()*Q << std::endl;


	//test least squares solutions
	w = L/v;
	std::cout << "w:= L/v= " << w <<std::endl;
	std::cout << "L*w-v= " << L*w-v << std::endl;

}