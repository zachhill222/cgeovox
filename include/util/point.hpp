#pragma once

#include <iostream>
#include <initializer_list>
#include <cmath>
#include <limits>

namespace gv::util {
	///Class for points in space.
	/** Points are partially ordered by using the positive quadrant/octant cone. The data type T must be totally ordered, for example double or float.
	 * Note that the initialization Point p {1,2,3} gives p the type Point<3,int> while the initialization Point p {1.0, 2.0, 3.0} give p the type Point<3,double>.*/
	template <int dim=3, typename T=double>
	class Point {
	protected:
		T* _data;

	public:
		//default constructor (all zeros)
		Point () : _data(new T[dim])
		{
			for (int i=0; i<dim; i++) {_data[i] = 0;}
		}

		//constructor for constant value
		Point (const T val) : _data(new T[dim])
		{
			for (int i=0; i<dim; i++) {_data[i] = val;}
		}

		//initialize via braces {1,2,3} and converte types if needed
		template <typename U>
		Point (std::initializer_list<U> init) : _data(new T[dim])
		{
			int i=0;
			for (auto it=init.begin(); it!=init.end(); ++it){
				_data[i++] = (T) *it;
			}
		}
		
		//copy constructor
		Point (const Point<dim,T> &other) : _data(new T[dim])
		{
			for (int i=0; i<dim; i++) {_data[i] = other._data[i];}
		}

		//copy constructor with type conversion if needed
		template <typename U>
		Point (const Point<dim,U> &other) : _data(new T[dim])
		{
			for (int i=0; i<dim; i++) {_data[i] = (T) other[i];}
		}

		//destructor
		~Point()
		{
			if (_data!=nullptr) {delete[] _data; _data=nullptr;}
		}

		//move constructor (must be correct type obviously)
		Point (Point<dim,T>&& other) noexcept : _data(nullptr)
		{
			_data = other._data;
			other._data = nullptr;
		}

		

		//move assignment
		Point& operator=( Point<dim,T>&& other) noexcept
		{
			if (this != &other)
			{
				if (_data!=nullptr) {delete[] _data;}
				_data = other._data;
				other._data = nullptr;
			}
			return *this;
		}

		//copy assignment
		constexpr Point& operator=( const Point<dim,T>& other)
		{
			for (int i=0; i<dim; i++) {_data[i]=other._data[i];}
			return *this;
		}

		T& operator[](const int idx) {return _data[idx];}
		
		T  operator[](const int idx) const {return _data[idx];}
		
		T  at(const int idx) const
		{
			if (idx<0 or idx>=dim) {throw std::runtime_error("INDEX_OUT_OF_RANGE"); return (T) 0;}
			if (_data==nullptr) {throw std::runtime_error("DATA_MOVED"); return (T) 0;}
			return _data[idx];
		}
	};



	// ///Scalar maximum.
	// template <typename T=double>
	// T max(const T &left, const T &right) {return (left > right) ? left : right;}


	// ///Scalar minimum.
	// template <typename T=double>
	// T min(const T &left, const T &right) {return (left < right) ? left : right;}


	///Scalar absolute value.
	template <typename T=double>
	T abs(const T &val) {return (val < 0) ? -val : val;}

	///convenient sign function
	template <typename T=double>
	T sgn(const T& x) {return (x < 0) ? -1 : 1;}

	///Point addition.
	template <int dim=3, typename T=double>
	Point<dim, T> operator+(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result(0);
		for (int i=0; i<dim; i++) {result[i] = left[i]+right[i];}
		return result;
	}


	///In-place addition.
	template <int dim=3, typename T=double>
	Point<dim, T>& operator+=(Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) {left[i] += right[i];}
		return left;
	}


	///Point subtraction.
	template <int dim=3, typename T=double>
	Point<dim, T> operator-(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i]-right[i];}
		return result;
	}


	///In-place subtraction.
	template <int dim=3, typename T=double>
	Point<dim, T>& operator-=(Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) {left[i] -= right[i];}
		return left;
	}


	///Negation.
	template <int dim=3, typename T=double>
	Point<dim, T> operator-(const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = -right[i];}
		return result;
	}


	///Scalar multiplication.
	template <int dim=3, typename T=double>
	Point<dim, T> operator*(const T &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left*right[i];}
		return result;
	}


	///Scalar multiplication with type conversion.
	template <int dim=3, typename T=double, typename S>
	Point<dim,T> operator*(const S &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = ((T) left)*right[i];}
		return result;
	}


	///In-place scalar multiplication
	template <int dim=3, typename T=double>
	Point<dim, T>& operator*=(Point<dim,T> &left, const T &right)
	{
		for (int i=0; i<dim; i++) {left[i] *= right;}
		return left;
	}


	///In-place scalar multiplication with type conversion
	template <int dim=3, typename T=double, typename S>
	Point<dim, T>& operator*=(Point<dim,T> &left, const S &right)
	{
		for (int i=0; i<dim; i++) {left[i] *= (T) right;}
		return left;
	}


	///Component-wise multiplication.
	template <int dim=3, typename T=double>
	Point<dim, T> operator*(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i]*right[i];}
		return result;
	}


	///In-place component-wise multiplication.
	template <int dim=3, typename T=double>
	Point<dim, T>& operator*=(Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) {left[i]*=right[i];}
		return left;
	}


	///Cross-product for dim=3
	template <typename T=double>
	Point<3, T> cross(const Point<3,T> &left, const Point<3,T> &right)
	{
		Point<3, T> result;
		result[0] = left[1]*right[2]-left[2]*right[1];
		result[1] = left[2]*right[0]-left[0]*right[2];
		result[2] = left[0]*right[1]-left[1]*right[0];
		return result;
	}


	///Dot-product
	template <int dim=3, typename T=double>
	T dot(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		T result = 0;
		for (int i=0; i<dim; i++) {result += left[i]*right[i];}
		return result;
	}

	///Component-wise division.
	template <int dim=3, typename T=double>
	Point<dim, T> operator/(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i]/right[i];}
		return result;
	}

	///In-place component-wise division.
	template <int dim=3, typename T=double>
	Point<dim, T>& operator/=(Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) {left[i]/=right[i];}
		return left;
	}


	///Point less than comparison.
	template <int dim=3, typename T=double>
	bool operator<(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) { if (left[i]>=right[i]) {return false;} }
		return true;
	}

	///Point less than or equal to comparison.
	template <int dim=3, typename T=double>
	bool operator<=(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) { if (left[i]>right[i]){return false;} }
		return true;
	}

	///Point greater than comparison.
	template <int dim=3, typename T=double>
	bool operator>(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) { if (left[i]<=right[i]){return false;} }
		return true;
	}

	///Point greater than or equal to comparison.
	template <int dim=3, typename T=double>
	bool operator>=(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		for (int i=0; i<dim; i++) {	if (left[i]<right[i]){return false;} }
		return true;
	}

	///Scalar approximately equal
	template <typename T=double>
	bool approxEqual(const T &left, const T &right)
	{
		T absmax = std::max( gv::util::abs(left), gv::util::abs(right) );
		T delta = gv::util::abs(left-right);
		return  delta <= std::numeric_limits<T>::epsilon() * 2 * absmax;
	}

	///Point equal to comparison.
	template <int dim=3, typename T=double>
	bool operator==(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		// for (int i=0; i<dim; i++)
		// {
		// 	// if (left[i] != right[i]) {return false;}
		// 	if (not approxEqual(left[i], right[i])) {return false;}
		// }
		// return true;
		return approxEqual( squaredNorm(left-right), (T) 0);
	}

	///Point not equal to comparison.
	template <int dim=3, typename T=double>
	bool operator!=(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		return !operator==(left,right);
	}

	///Element-wise maximum.
	template <int dim=3, typename T=double>
	Point<dim, T> elmax(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) { result[i] = std::max(left[i],right[i]);}
		return result;
	}


	///Element-wise minimum.
	template <int dim=3, typename T=double>
	Point<dim, T> elmin(const Point<dim,T> &left, const Point<dim,T> &right)
	{
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = std::min(left[i],right[i]);}
		return result;
	}


	///Element-wise absolute value
	template <int dim=3, typename T=double>
	Point<dim,T> abs(const Point<dim,T> &point)
	{
		Point<dim,T> result;
		for (int i=0; i<dim; i++) {result[i] = abs(point[i]);}
		return result;
	}


	///Maximum element.
	template <int dim=3, typename T=double>
	T max(const Point<dim,T> &point)
	{
		T result = point[0];
		for (int i=1; i<dim; i++) {result = std::max(result,point[i]);}
		return result;
	}


	///Minimum element.
	template <int dim=3, typename T=double>
	T min(const Point<dim,T> &point)
	{
		T result = point[0];
		for (int i=1; i<dim; i++) {result = min(result,point[i]);}
		return result;
	}




	///Print to ostream.
	template <int dim=3, typename T=double>
	std::ostream& operator<<(std::ostream& os, const Point<dim,T> &point)
	{
		for (int i = 0; i < dim-1; i++) {os << point.at(i) << " ";}
		os << point.at(dim-1);
		return os;
	}


	///Squared Norm
	template <int dim, typename T>
	T squaredNorm(const Point<dim,T> &point)
	{
		T result = 0;
		for (int i=0; i<dim; i++) {result+=point[i]*point[i];}
		return result;
	}


	///L2-norm
	template <int dim=3, typename T=double>
	T norm2(const Point<dim,T> &point) {return std::sqrt(squaredNorm(point));}


	///L1-norm
	template <int dim=3, typename T=double>
	T norm1(const Point<dim,T> &point)
	{
		T result = 0;
		for (int i=0; i<dim; i++) {result+=abs(point[i]);}
		return result;
	}


	///L-infinity norm
	template <int dim=3, typename T=double>
	T norminfty(const Point<dim,T> &point) {return max(abs(point));}


	///Normalize (without modification)
	template <int dim, typename T>
	Point<dim,T> normalize(const Point<dim,T> &point)
	{
		T scale = std::sqrt(squaredNorm(point));
		return (1.0/scale) * point;
	}

}