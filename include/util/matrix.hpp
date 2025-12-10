//these classes are for small dense matrices only
#pragma once

#define ZERO_TOL 1E-10 //tolerance for determining if an element is essentially zero (e.g., in QR)

#include "util/point.hpp"

#include "concepts.hpp"

#include <iostream>
#include <cassert>
#include <algorithm>

namespace gv::util
{
	//Scalar is defined in concepts.hpp
	//note that inverses and such will not work for integral types

	template <int n, int m, Scalar Scalar_t>
	class Matrix
	{
	protected:
		Scalar_t _data[n*m];
		static constexpr int _col_major_index(const int i, const int j) noexcept {return n*j+i;}
		static constexpr int _row_major_index(const int i, const int j) noexcept {return m*i+j;}

		using IndexFunction_t = int(*)(int, int);
		IndexFunction_t _index = _col_major_index;

	public:
		using Row_t = gv::util::Point<m,Scalar_t>;
		using Col_t = gv::util::Point<n,Scalar_t>;
		static constexpr bool IS_SQUARE = n==m;

		constexpr Matrix() noexcept : _data{0} {}
		
		constexpr Matrix(const Scalar_t a) noexcept
		{
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					if (i==j) {_data[_index(i,j)] = a;}
					else {_data[_index(i,j)] = Scalar_t{0};}
				}
			}
		}

		constexpr Matrix(const Matrix& other) noexcept
		{
			std::copy(other._data, other._data+n*m, _data);
			_index = other._index;
		}

		constexpr Matrix(Matrix&& other) noexcept
		{
			std::move(other._data, other._data+n*m, _data);
			_index = other._index;
		}

		//element access
		constexpr Scalar_t at(int i, int j) const
		{
			if (0 <= i and i < n) {
				if (0 <= j and j < m) {
					return _data[_index(i,j)];
				}
			}
			
			throw std::runtime_error("Matrix: index out of bounds");
		}

		constexpr Scalar_t& at(int i, int j)
		{
			if (0 <= i and i < n) {
				if (0 <= j and j < m) {
					return _data[_index(i,j)];
				}
			}
			
			throw std::runtime_error("Matrix: index out of bounds");
		}

		constexpr Scalar_t operator()(int i, int j) const noexcept
		{
			assert(0 <= i and i < n);
			assert(0 <= j and j < m);
			return _data[_index(i,j)];
		}

		constexpr Scalar_t& operator()(int i, int j) noexcept
		{
			assert(0 <= i and i < n);
			assert(0 <= j and j < m);
			return _data[_index(i,j)];
		}

		constexpr Scalar_t operator[](int k) const noexcept
		{
			assert(0 <= k and k<m*n);
			return _data[k];
		}

		constexpr Scalar_t& operator[](int k) noexcept
		{
			assert(0 <= k and k<m*n);
			return _data[k];
		}

		//copy and move assignment
		constexpr Matrix<n,m,Scalar_t>& operator=(const Matrix<n,m,Scalar_t>& other) noexcept
		{
			std::copy(other._data, other._data+n*m, _data);
			_index = other._index;
			return *this;
		}

		constexpr Matrix<n,m,Scalar_t>& operator=(Matrix<n,m,Scalar_t>&& other) noexcept
		{
			std::move(other._data, other._data+n*m, _data);
			_index = other._index;
			return *this;
		}

		//set identity along the main diagonal and zeros elsewhere
		constexpr void fill(const Scalar_t val) noexcept {std::fill(_data, _data+n*m, val);}

		//row and column access
		constexpr Row_t row(int i) const noexcept
		{
			assert(0<=i and i<n);
			Row_t result;
			for (int k=0; k<m; k++) {result[k]=_data[_index(i,k)];}
			return result;
		}

		constexpr Col_t col(int j) const noexcept
		{
			assert(0<=j and j<m);
			Col_t result;
			for (int k=0; k<n; k++) {result[k]=_data[_index(k,j)];}
			return result;
		}

		//transpose
		Matrix<m,n,Scalar_t> tr() const noexcept
		{
			Matrix<m,n,Scalar_t> result;
			std::copy(_data, _data+n*m, result._data);
			if (_index == _col_major_index) {result._index = result._row_major_index;}
			else {result._index = result._col_major_index;}
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
	template<int n, int m, Scalar Scalar_t, Scalar Scalar_u>
	constexpr Matrix<n,m,Scalar_t> operator*(const Scalar_u a, const Matrix<n,m,Scalar_t> &matrix) noexcept
	{
		Matrix<n,m,Scalar_t> result = matrix;
		for (int k=0; k<n*m; k++) {result[k] *= static_cast<Scalar_t>(a);}
		return result;
	}


	//matrix-vector multiplication
	template<int n, int m, Scalar Scalar_t>
	constexpr gv::util::Point<n,Scalar_t> operator*(const Matrix<n,m,Scalar_t> &matrix, const gv::util::Point<m,Scalar_t> &vector) noexcept
	{
		gv::util::Point<n,Scalar_t> result; //all zeros
		for (int j=0; j<m; j++) {result += vector[j]*matrix.col(j);}
		return result;
	}

	//vector transpose * matrix
	template<int n, int m, Scalar Scalar_t>
	constexpr gv::util::Point<m,Scalar_t> operator*(const gv::util::Point<n,Scalar_t> &vector, const Matrix<n,m,Scalar_t> &matrix) noexcept
	{
		gv::util::Point<m,Scalar_t> result; //all zeros
		for (int j=0; j<m; j++) {result[j] = gv::util::dot(matrix.col(j),vector);}
		return result;
	}

	//vector-vector outer product
	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t> outer(const gv::util::Point<n,Scalar_t> &left, const gv::util::Point<m,Scalar_t> &right) noexcept
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
	
	//matrix-matrix multiplication
	template<int n, int m, int p, Scalar Scalar_t>
	constexpr Matrix<n,p,Scalar_t> operator*(const Matrix<n,m,Scalar_t> &left, const Matrix<m,p,Scalar_t> &right) noexcept
	{
		Matrix<n,p,Scalar_t> result;
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<p; j++)
			{
				for (int k=0; k<m; k++)
				{
					result(i,j) += left(i,k)*right(k,j);
				}
			}
		}

		return result;
	}

	//matrix least squares solution (must columns must be linearly independent)
	template<int n, int m, Scalar Scalar_t>
	constexpr typename Matrix<n,m,Scalar_t>::Row_t operator/(const Matrix<n,m,Scalar_t> A, const typename Matrix<n,m,Scalar_t>::Col_t &b) noexcept
	{
		Matrix<n,m,Scalar_t> Q;
		Matrix<m,m,Scalar_t> R;
		partialQR(A,Q,R);

		//note b*Q is b^T * Q = (Q.tr()*b)^T. For a vector, b and b^T are both the same C++ type
		return solve_upper(R,b*Q);
	}

	//matrix-matrix addition
	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t> operator+(const Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right) noexcept
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k] = left[k]+right[k];}
		return result;
	}

	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t>& operator+=(Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right) noexcept
	{
		for (int k=0; k<n*m; k++) {left[k]+=right[k];}
		return left;
	}

	//matrix-matrix subtraction
	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t> operator-(const Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right) noexcept
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k] = left[k]-right[k];}
		return result;
	}

	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t>& operator-=(Matrix<n,m,Scalar_t> &left, const Matrix<n,m,Scalar_t> &right) noexcept
	{
		for (int k=0; k<n*m; k++) {left[k]-=right[k];}
		return left;
	}

	template<int n, int m, Scalar Scalar_t>
	constexpr Matrix<n,m,Scalar_t> operator-(const Matrix<n,m,Scalar_t> &right) noexcept
	{
		Matrix<n,m,Scalar_t> result;
		for (int k=0; k<n*m; k++) {result[k]=-right[k];}
		return result;
	}

	//triangular matrix solve Ux=b with U upper triangular
	template<int n, Scalar Scalar_t>
	constexpr gv::util::Point<n,Scalar_t> solve_upper(const Matrix<n,n,Scalar_t> &U, const gv::util::Point<n,Scalar_t> &b) noexcept
	{
		gv::util::Point<n,Scalar_t> x=b;
		for (int i=n-1; i>=0; i--)
		{
			assert(U(i,i)!=Scalar_t{0});

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
	constexpr gv::util::Point<n,Scalar_t> solve_lower(const Matrix<n,n,Scalar_t> &L, const gv::util::Point<n,Scalar_t> &b) noexcept
	{
		gv::util::Point<n,Scalar_t> x=b;
		for (int i=0; i<n; i++)
		{
			assert(L(i,i)!=Scalar_t{0});

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
	constexpr void partialQR(const Matrix<n,m,Scalar_t> &A, Matrix<n,m,Scalar_t> &Q, Matrix<m,m,Scalar_t> &R) noexcept
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