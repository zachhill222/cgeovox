#pragma once

#include <iostream>
#include <initializer_list>
#include <cmath>
#include <limits>
#include <cassert>
#include <concepts>
#include <vector>
#include <algorithm>

#include "concepts.hpp"

namespace gv::util {
	///////////////////////////////////////
	/// Concept for Point-like data types
	///////////////////////////////////////
	template<typename T>
	concept PointLike = requires(T point) {
		typename T::Scalar_t;
		requires Scalar<typename T::Scalar_t>;
		requires T::dimension > 0;
	};

	///Class for points in space.
	/** Points are partially ordered by using the positive quadrant/octant cone. 
	 * The data type T must be totally ordered, for example double or float.
	 * Note that Point p {1,2,3} gives type Point<3,int> while Point p {1.0, 2.0, 3.0} 
	 * gives type Point<3,double>.*/
	template <int dim=3, Scalar T=double>
	class Point {
	private:
		T _data[dim];

	public:
		using Scalar_t = T;
		static constexpr int dimension = dim;

		//============================================================
		// Constructors
		//============================================================
		
		// Default constructor (all zeros)
		constexpr Point() {
			for (int i = 0; i < dim; i++) {
				_data[i] = T{0};
			}
		}

		// Constructor for constant value
		constexpr explicit Point(const T val) {
			for (int i=0; i<dim; i++) {_data[i] = val;}
		}

		// Initialize via braces {1,2,3}
		template <Scalar U>
		constexpr Point(std::initializer_list<U> init) : _data{} {
			int i=0;
			for (auto it=init.begin(); it!=init.end() && i<dim; ++it, ++i) {
				_data[i] = static_cast<T>(*it);
			}
		}

		// Copy constructor (same type)
		constexpr Point(const Point<dim,T> &other) {
			for (int i=0; i<dim; i++) {_data[i] = other[i];}
		}

		// Copy constructor with type conversion
		template <Scalar U>
		constexpr explicit Point(const Point<dim,U> &other) {
			for (int i=0; i<dim; i++) {_data[i] = static_cast<T>(other[i]);}
		}

		// Copy constructor with dimension change (trim/zero pad)
		template<int otherdim, Scalar U>
		constexpr explicit Point(const Point<otherdim,U> &other) {
			constexpr int min_dim = std::min(dim, otherdim);
			for (int i=0; i<min_dim; i++) {_data[i] = static_cast<T>(other[i]);}
			for (int i=min_dim; i<dim; i++) {_data[i] = T{};}
		}

		// Move constructor
		constexpr Point(Point<dim,T>&& other) noexcept {
			for (int i=0; i<dim; i++) {_data[i] = std::move(other._data[i]);}
		}

		// Destructor (default is fine for array member)
		~Point() = default;

		//============================================================
		// Assignment operators
		//============================================================
		
		// Copy assignment
		constexpr Point& operator=(const Point<dim,T>& other) {
			if (this != &other) {
				for (int i=0; i<dim; i++) {_data[i] = other._data[i];}
			}
			return *this;
		}

		// Move assignment
		constexpr Point& operator=(Point<dim,T>&& other) noexcept {
			if (this != &other) {
				for (int i=0; i<dim; i++) {_data[i] = std::move(other._data[i]);}
			}
			return *this;
		}

		//============================================================
		// Element access
		//============================================================
		
		constexpr T& operator[](int idx) {
			assert(0 <= idx && idx < dim); 
			return _data[idx];
		}
		
		constexpr const T& operator[](int idx) const {
			assert(0 <= idx && idx < dim); 
			return _data[idx];
		}
		
		constexpr const T& at(int idx) const {
			if (idx < 0 || idx >= dim) {
				throw std::runtime_error("INDEX_OUT_OF_RANGE");
			}
			return _data[idx];
		}

		constexpr T& at(int idx) {
			if (idx < 0 || idx >= dim) {
				throw std::runtime_error("INDEX_OUT_OF_RANGE");
			}
			return _data[idx];
		}

		//============================================================
		// Utility
		//============================================================
		
		static constexpr Point<dim,T> zero() {
			return Point<dim,T>();
		}
	};

	static_assert(PointLike<Point<3,double>>, "Point<3,double> is not PointLike");
	static_assert(PointLike<Point<2,double>>, "Point<2,double> is not PointLike");
	static_assert(PointLike<Point<3,float>>, "Point<3,float> is not PointLike");
	static_assert(PointLike<Point<2,float>>, "Point<2,float> is not PointLike");

	//============================================================
	// Arithmetic operators with automatic type promotion
	//============================================================

	/// Determine result type for binary operations
	template<Scalar T, Scalar U>
	using promoted_t = decltype(T{} + U{});

	/// Point addition (with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator+(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] + right[i];}
		return result;
	}

	/// In-place addition
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator+=(Point<dim,T> &left, const Point<dim,U> &right) {
		for (int i=0; i<dim; i++) {left[i] += static_cast<T>(right[i]);}
		return left;
	}

	/// Point subtraction (with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator-(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] - right[i];}
		return result;
	}

	/// In-place subtraction
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator-=(Point<dim,T> &left, const Point<dim,U> &right) {
		for (int i=0; i<dim; i++) {left[i] -= static_cast<T>(right[i]);}
		return left;
	}

	/// Negation
	template <int dim, Scalar T>
	constexpr Point<dim,T> operator-(const Point<dim,T> &right) {
		Point<dim,T> result;
		for (int i=0; i<dim; i++) {result[i] = -right[i];}
		return result;
	}

	/// Scalar multiplication (scalar * point, with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator*(const U &left, const Point<dim,T> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left * right[i];}
		return result;
	}

	/// Scalar multiplication (point * scalar, with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator*(const Point<dim,T> &left, const U &right) {
		return right * left;
	}

	/// In-place scalar multiplication
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator*=(Point<dim,T> &left, const U &right) {
		for (int i=0; i<dim; i++) {left[i] *= static_cast<T>(right);}
		return left;
	}

	/// Component-wise multiplication (with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator*(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] * right[i];}
		return result;
	}

	/// In-place component-wise multiplication
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator*=(Point<dim,T> &left, const Point<dim,U> &right) {
		for (int i=0; i<dim; i++) {left[i] *= static_cast<T>(right[i]);}
		return left;
	}

	/// Division by scalar (with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator/(const Point<dim,T> &left, const U &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] / right;}
		return result;
	}

	/// In-place division by scalar
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator/=(Point<dim,T> &left, const U &right) {
		for (int i=0; i<dim; i++) {left[i] /= static_cast<T>(right);}
		return left;
	}

	/// Component-wise division (with type promotion)
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> operator/(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] / right[i];}
		return result;
	}

	/// In-place component-wise division
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator/=(Point<dim,T> &left, const Point<dim,U> &right) {
		for (int i=0; i<dim; i++) {left[i] /= static_cast<T>(right[i]);}
		return left;
	}

	//============================================================
	// Comparison operators
	//============================================================

	/// Scalar approximately equal (using relative epsilon)
	template <Scalar T>
	constexpr bool approxEqual(const T &left, const T &right) {
		if (left == right) {return true;}
		
		T absmax = std::max(std::abs(left), std::abs(right));
		T delta = std::abs(left - right);
		return delta <= std::numeric_limits<T>::epsilon() * 2 * absmax;
	}

	/// Point equality (using approximate comparison on squared norm)
	template <int dim, Scalar T>
	constexpr bool operator==(const Point<dim,T> &left, const Point<dim,T> &right) {
		return approxEqual(squaredNorm(left - right), T{0});
	}

	/// Point inequality
	template <int dim, Scalar T>
	constexpr bool operator!=(const Point<dim,T> &left, const Point<dim,T> &right) {
		return !(left == right);
	}

	/// Point less than (cone ordering)
	template <int dim, Scalar T>
	constexpr bool operator<(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] >= right[i]) {return false;}
		}
		return true;
	}

	/// Point less than or equal to
	template <int dim, Scalar T>
	constexpr bool operator<=(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] > right[i]) {return false;}
		}
		return true;
	}

	/// Point greater than
	template <int dim, Scalar T>
	constexpr bool operator>(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] <= right[i]) {return false;}
		}
		return true;
	}

	/// Point greater than or equal to
	template <int dim, Scalar T>
	constexpr bool operator>=(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] < right[i]) {return false;}
		}
		return true;
	}

	//============================================================
	// Vector operations
	//============================================================

	/// Dot product
	template <int dim, Scalar T, Scalar U>
	constexpr promoted_t<T,U> dot(const Point<dim,T> &left, const Point<dim,U> &right) {
		promoted_t<T,U> result = 0;
		for (int i=0; i<dim; i++) {result += left[i] * right[i];}
		return result;
	}

	/// Cross product (3D only)
	template <Scalar T, Scalar U>
	constexpr Point<3, promoted_t<T,U>> cross(const Point<3,T> &left, const Point<3,U> &right) {
		Point<3, promoted_t<T,U>> result;
		result[0] = left[1]*right[2] - left[2]*right[1];
		result[1] = left[2]*right[0] - left[0]*right[2];
		result[2] = left[0]*right[1] - left[1]*right[0];
		return result;
	}

	/// Squared norm
	template <int dim, Scalar T>
	constexpr T squaredNorm(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {result += point[i] * point[i];}
		return result;
	}

	/// L2-norm
	template <int dim, Scalar T>
	inline T norm2(const Point<dim,T> &point) {
		return std::sqrt(squaredNorm(point));
	}

	/// L1-norm
	template <int dim, Scalar T>
	constexpr T norm1(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {result += std::abs(point[i]);}
		return result;
	}

	/// L-infinity norm
	template <int dim, Scalar T>
	constexpr T norminfty(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {
			result = std::max(result, std::abs(point[i]));
		}
		return result;
	}

	/// Normalize (returns normalized copy)
	template <int dim, Scalar T>
	inline Point<dim,T> normalize(const Point<dim,T> &point) {
		T scale = norm2(point);
		return point / scale;
	}

	//============================================================
	// Element-wise operations
	//============================================================

	/// Element-wise absolute value
	template <int dim, Scalar T>
	constexpr Point<dim,T> abs(const Point<dim,T> &point) {
		Point<dim,T> result;
		for (int i=0; i<dim; i++) {result[i] = std::abs(point[i]);}
		return result;
	}

	/// Element-wise maximum
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> elmax(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = std::max(left[i], right[i]);}
		return result;
	}

	/// Element-wise minimum
	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, promoted_t<T,U>> elmin(const Point<dim,T> &left, const Point<dim,U> &right) {
		Point<dim, promoted_t<T,U>> result;
		for (int i=0; i<dim; i++) {result[i] = std::min(left[i], right[i]);}
		return result;
	}

	/// Maximum element
	template <int dim, Scalar T>
	constexpr T max(const Point<dim,T> &point) {
		T result = point[0];
		for (int i=1; i<dim; i++) {result = std::max(result, point[i]);}
		return result;
	}

	/// Minimum element
	template <int dim, Scalar T>
	constexpr T min(const Point<dim,T> &point) {
		T result = point[0];
		for (int i=1; i<dim; i++) {result = std::min(result, point[i]);}
		return result;
	}

	/// Sum points in careful precision order
	/// W is the input point type
	/// U is the type that the arithmetic should be done in
	/// T is the output type
	template <int dim, Scalar T, Scalar U, Scalar W>
	Point<dim,T> sorted_sum(const std::vector<Point<dim,W>> &points) {
		if (points.empty()) {return Point<dim,T>();}
		
		Point<dim,T> result;
		std::vector<T> component;
		component.reserve(points.size());
		for (int i = 0; i < dim; i++) {
			component.clear();
			for ( const Point<dim,W> &p : points) {
				component.push_back(static_cast<U>(p[i]));
			}

			std::sort(component.begin(), component.end(), [](T a, T b) {
				            return std::abs(a) < std::abs(b);});

			U sum = U{0};
			for (U val : component) {
				sum += val;
			}

			result[i] = static_cast<T>(sum);
		}

		return result;
	}

	template <int dim, Scalar T, Scalar U, Scalar W>
	Point<dim,T> sorted_sum(std::initializer_list<Point<dim,W>> points) {
	    return sorted_sum<dim, T, U, W>(std::vector<Point<dim,W>>(points.begin(), points.end()));
	}

	//============================================================
	// I/O
	//============================================================

	/// Print to ostream
	template <int dim, Scalar T>
	std::ostream& operator<<(std::ostream& os, const Point<dim,T> &point) {
		for (int i = 0; i < dim-1; i++) {os << point[i] << " ";}
		os << point[dim-1];
		return os;
	}

} // namespace gv::util