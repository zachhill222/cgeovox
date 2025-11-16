//these classes are for small dense matrices only
#pragma once

#define ZERO_TOL 1E-10 //tolerance for determining if an element is essentially zero (e.g., in QR)

#include "util/point.hpp"

#include "concepts.hpp"

#include <iostream>
#include <cassert>

namespace gv::util
{
	//Scalar is defined in concepts.hpp
	//note that inverses and such will not work for integral types

	template <int n, int m, Scalar Scalar_t>
	class Matrix
	{
	protected:
		Scalar_t _data[n*m] {0};
		int _index(const int i, const int j) const {return n*j+i;}

	public:
		using Row_t = gv::util::Point<m,Scalar_t>;
		using Col_t = gv::util::Point<n,Scalar_t>;

		constexpr Matrix() {}

		Matrix(const Scalar_t a)
		{
			for (int k=0; k<n*m; k++) {_data[k]=a;}
		}


		//element access
		Scalar_t at(int i, int j) const {assert(i<n); assert(j<m); return _data[_index(i,j)];}
		Scalar_t& at(int i, int j) {assert(i<n); assert(j<m); return _data[_index(i,j)];}

		Scalar_t operator()(int i, int j) const {return _data[_index(i,j)];}
		Scalar_t& operator()(int i, int j) {return _data[_index(i,j)];}

		Scalar_t operator[](int k) const {return _data[k];}
		Scalar_t& operator[](int k) {return _data[k];}

		//set identity along the main diagonal and zeros elsewhere
		void eye()
		{
			for (int i=0; i<n; i++)
			{
				for (int j=0; j<m; j++)
				{
					if (i==j) {(*this)(i,j)=1;}
					else {(*this)(i,j)=0;}
				}
			}
		}

		//row and column access
		Row_t row(int i) const
		{
			assert(i<n);
			Row_t result;
			for (int k=0; k<m; k++) {result[k]=_data[_index(i,k)];}
			return result;
		}

		Col_t col(int j) const
		{
			assert(j<m);
			Col_t result;
			for (int k=0; k<n; k++) {result[k]=_data[_index(k,j)];}
			return result;
		}

		//transpose
		Matrix<m,n,Scalar_t> tr() const
		{
			Matrix<m,n,Scalar_t> result;
			for (int i=0; i<n; i++)
			{
				for (int j=0; j<m; j++)
				{
					result(j,i) = (*this)(i,j);
				}
			}

			return result;
		}
	};


	///Print to ostream.
	template <int n, int m, Scalar Scalar_t>
	std::ostream& operator<<(std::ostream& os, const Matrix<n,m,Scalar_t> &matrix)
	{
		for (int i = 0; i < n; i++) {os << "[" << matrix.row(i) << "]\n";}
		return os;
	}

	//scalar-matrix multiplication
	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t> operator*(const Scalar_t a, const Matrix<n,m,Scalar_t> &matrix)
	{
		Matrix<n,m,Scalar_t> result = matrix;
		for (int k=0; k<n*m; k++) {result[k] *= a;}
		return result;
	}


	//matrix-vector multiplication
	template<int n, int m, Scalar Scalar_t>
	gv::util::Point<n,Scalar_t> operator*(const Matrix<n,m,Scalar_t> &matrix, const gv::util::Point<m,Scalar_t> &vector)
	{
		gv::util::Point<n,Scalar_t> result; //all zeros
		for (int j=0; j<m; j++) {result += vector[j]*matrix.col(j);}
		return result;
	}

	//vector-vector outer product
	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t> outer(const gv::util::Point<n,Scalar_t> &left, const gv::util::Point<m,Scalar_t> &right)
	{
		Matrix<n,m,Scalar_t> result;
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<m; j++)
			{
				result(i,j) = left[i]*right[j];
			}
		}
		return result;
	}
	
	template<int n, int m, Scalar Scalar_t>
	gv::util::Point<m,Scalar_t> operator*(const gv::util::Point<n,Scalar_t> &vector, const Matrix<n,m,Scalar_t> &matrix)
	{
		gv::util::Point<m,Scalar_t> result; //all zeros
		for (int j=0; j<m; j++) {result[j] = gv::util::dot(matrix.col(j),vector);}
		return result;
	}

	//matrix-matrix multiplication
	template<int n, int m, int p, Scalar Scalar_t>
	Matrix<n,p,Scalar_t> operator*(const Matrix<n,m,Scalar_t> &left, const Matrix<m,p,Scalar_t> &right)
	{
		Matrix<n,p,Scalar_t> result;
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<p; j++)
			{
				for (int k=0; k<m; k++)
				{
					result.at(i,j) += left.at(i,k)*right.at(k,j);
				}
			}
		}

		return result;
	}

	//matrix least squares solution (must columns must be linearly independent)
	template<int n, int m, Scalar Scalar_t>
	typename Matrix<n,m,Scalar_t>::Row_t operator/(const Matrix<n,m,Scalar_t> A, const typename Matrix<n,m,Scalar_t>::Col_t &b)
	{
		Matrix<n,m,Scalar_t> Q;
		Matrix<m,m,Scalar_t> R;
		partialQR(A,Q,R);

		return solve_upper(R,Q.tr()*b);
	}

	//matrix-matrix addition
	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t> operator+(const Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right)
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k] = left[k]+right[k];}
		return result;
	}

	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t>& operator+=(Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right)
	{
		for (int k=0; k<n*m; k++) {left[k]+=right[k];}
		return left;
	}

	//matrix-matrix subtraction
	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t> operator-(const Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right)
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k] = left[k]-right[k];}
		return result;
	}

	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t>& operator-=(Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right)
	{
		for (int k=0; k<n*m; k++) {left[k]-=right[k];}
		return left;
	}

	template<int n, int m, Scalar Scalar_t>
	Matrix<n,m,Scalar_t> operator-(const Matrix<n,m,Scalar_t> &right)
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k]=-right[k];}
		return result;
	}

	//triangular matrix solve Ux=b with U upper triangular
	template<int n, Scalar Scalar_t>
	gv::util::Point<n,Scalar_t> solve_upper(const Matrix<n,n,Scalar_t> &U, const gv::util::Point<n,Scalar_t> &b)
	{
		gv::util::Point<n,Scalar_t> x=b;
		for (int i=n-1; i>=0; i--)
		{
			assert(U(i,i)!=0);

			for(int j=i+1; j<n; j++)
			{
				x[i] -= U(i,j)*x[j];
			}
			x[i]/= U(i,i);
		}
		return x;
	}

	//triangular matrix solve Lx=b with L lower triangular
	template<int n, Scalar Scalar_t>
	gv::util::Point<n,Scalar_t> solve_lower(const Matrix<n,n,Scalar_t> &L, const gv::util::Point<n,Scalar_t> &b)
	{
		gv::util::Point<n,Scalar_t> x=b;
		for (int i=0; i<n; i++)
		{
			assert(L(i,i)!=0);

			for(int j=0; j<i; j++)
			{
				x[i] -= L(i,j)*x[j];
			}
			x[i]/= L(i,i);
		}
		return x;
	}


	//Partial QR decomposition (modified Gram-Schmidt)
	template<int n, int m, Scalar Scalar_t>
	void partialQR(const Matrix<n,m,Scalar_t> &A, Matrix<n,m,Scalar_t> &Q, Matrix<m,m,Scalar_t> &R)
	{
		//Ensure that R and Q are zeros
		R = Matrix<m,m,Scalar_t>(0);
		Q = A;

		//Modified G-S
		for (int k=0; k<m; k++)
		{
			for (int i=0; i<k; i++)
			{
				R(i,k) = gv::util::dot(Q.col(k), Q.col(i));
				for (int j=0; j<n; j++) {Q(j,k)-= Q(j,i)*R(i,k);}
			}

			R(k,k) = gv::util::norm2(Q.col(k));
			if (!(R(k,k)>ZERO_TOL)) {std::cout << A << std::endl;}
			assert(R(k,k)>ZERO_TOL);

			Scalar_t C = ((Scalar_t) 1)/R(k,k);
			for (int j=0; j<n; j++) {Q(j,k) *= C;}
		}
	}
}