#pragma once

#include <iostream>
#include <initializer_list>
#include <cmath>
// #include <cstring>
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

	////////////////////////////////////////////////////////////////
	/// Class for points in space.
	/// Points are partially ordered by using the positive quadrant/octant cone.
	/// If floating point comparisons need to be made carefully,
	/// this should be implemented in the Scalar T type.
	///
	/// @tparam dim The number of components.
	/// @tparam T   The scalar type (e.g., float, double, gv::util::FixedPoint)
	////////////////////////////////////////////////////////////////
	template <int dim=3, Scalar T=double>
	struct Point {
		
		T _data[dim];

		using Scalar_t = T;
		static constexpr int dimension = dim;

		//////////////////////////////////////////////////////////
		// Constructors
		//////////////////////////////////////////////////////////
		constexpr Point() noexcept {
			for (int i = 0; i < dim; i++) {
				_data[i] = T{0}; }
		}

		constexpr explicit Point(const T val) noexcept {
			for (int i=0; i<dim; i++) {_data[i] = val;}
		}

		// Initialize via braces {1,2,3}
		template <Scalar U>
		constexpr Point(std::initializer_list<U> init) noexcept : _data{} {
			int i=0;
			for (auto it=init.begin(); it!=init.end() && i<dim; ++it, ++i) {
				_data[i] = static_cast<T>(*it);
			}
		}

		// Copy constructor (same type)
		constexpr Point(const Point<dim,T> &other) noexcept {
			std::copy(other._data, other._data+dim, _data);
		}

		// Copy constructor with type conversion
		template <Scalar U>
		constexpr explicit Point(const Point<dim,T> &other) noexcept {
			for (int i=0; i<dim; i++) {_data[i] = static_cast<T>(other[i]);}
		}

		// Copy constructor with dimension change (trim/zero pad)
		template<int otherdim, Scalar U>
		constexpr explicit Point(const Point<otherdim,U> &other) noexcept {
			constexpr int min_dim = std::min(dim, otherdim);
			for (int i=0; i<min_dim; i++) {_data[i] = static_cast<T>(other[i]);}
			for (int i=min_dim; i<dim; i++) {_data[i] = T{};}
		}

		// Move constructor
		constexpr Point(Point<dim,T>&& other) noexcept {
			std::move(other._data, other._data+dim, _data);
		}

		~Point() = default;

		//////////////////////////////////////////////////////////
		// Assignment operators
		//////////////////////////////////////////////////////////
		constexpr Point& operator=(const Point<dim,T>& other) noexcept {
			if (this != &other) {
				std::copy(other._data, other._data+dim, _data);
			}
			return *this;
		}

		// Move assignment
		constexpr Point& operator=(Point<dim,T>&& other) noexcept {
			if (this != &other) {
				std::move(other._data, other._data+dim, _data);
			}
			return *this;
		}

		//////////////////////////////////////////////////////////
		// Element access
		//////////////////////////////////////////////////////////
		constexpr T& operator[](int idx) noexcept{
			assert(0 <= idx && idx < dim); 
			return _data[idx];
		}
		
		constexpr const T& operator[](int idx) const noexcept {
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
	};

	static_assert(PointLike<Point<3,double>>, "Point<3,double> is not PointLike");
	static_assert(PointLike<Point<2,double>>, "Point<2,double> is not PointLike");
	static_assert(PointLike<Point<3,float>>, "Point<3,float> is not PointLike");
	static_assert(PointLike<Point<2,float>>, "Point<2,float> is not PointLike");

	//////////////////////////////////////////////////////////
	// Arithmetic
	//////////////////////////////////////////////////////////
	template <int dim, Scalar T>
	constexpr Point<dim, T> operator+(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] + right[i];}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim,T>& operator+=(Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {left[i] += right[i];}
		return left;
	}

	template <int dim, Scalar T>
	constexpr Point<dim, T> operator-(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] - right[i];}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim,T>& operator-=(Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {left[i] -= right[i];}
		return left;
	}

	template <int dim, Scalar T>
	constexpr Point<dim,T> operator-(const Point<dim,T> &right) {
		Point<dim,T> result;
		for (int i=0; i<dim; i++) {result[i] = -right[i];}
		return result;
	}

	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, T> operator*(const U &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = static_cast<T>(left) * right[i];}
		return result;
	}

	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, T> operator*(const Point<dim,T> &left, const U &right) {
		return right * left;
	}

	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator*=(Point<dim,T> &left, const U &right) {
		for (int i=0; i<dim; i++) {left[i] *= static_cast<T>(right);}
		return left;
	}

	template <int dim, Scalar T>
	constexpr Point<dim, T> operator*(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] * right[i];}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim,T>& operator*=(Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {left[i] *= right[i];}
		return left;
	}

	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim, T> operator/(const Point<dim,T> &left, const U &right) {
		return left * (T{1} / static_cast<T>(right));
	}

	template <int dim, Scalar T, Scalar U>
	constexpr Point<dim,T>& operator/=(Point<dim,T> &left, const U &right) {
		left *= (T{1} / static_cast<T>(right));
		return left;
	}

	template <int dim, Scalar T>
	constexpr Point<dim, T> operator/(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = left[i] / right[i];}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim,T>& operator/=(Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {left[i] /= right[i];}
		return left;
	}

	//////////////////////////////////////////////////////////
	// Comparison
	//////////////////////////////////////////////////////////
	template <int dim, Scalar T>
	constexpr bool operator==(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i]!=right[i]) {return false;}
		}
		return true;
	}

	template <int dim, Scalar T>
	constexpr bool operator!=(const Point<dim,T> &left, const Point<dim,T> &right) {
		return !(left == right);
	}

	template <int dim, Scalar T>
	constexpr bool operator<(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] >= right[i]) {return false;}
		}
		return true;
	}

	template <int dim, Scalar T>
	constexpr bool operator<=(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] > right[i]) {return false;}
		}
		return true;
	}

	template <int dim, Scalar T>
	constexpr bool operator>(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] <= right[i]) {return false;}
		}
		return true;
	}

	template <int dim, Scalar T>
	constexpr bool operator>=(const Point<dim,T> &left, const Point<dim,T> &right) {
		for (int i=0; i<dim; i++) {
			if (left[i] < right[i]) {return false;}
		}
		return true;
	}

	//////////////////////////////////////////////////////////
	// Vector operations
	//////////////////////////////////////////////////////////
	template <int dim, Scalar T>
	constexpr T dot(const Point<dim,T> &left, const Point<dim,T> &right) {
		T result = 0;
		for (int i=0; i<dim; i++) {result += left[i] * right[i];}
		return result;
	}

	template <Scalar T, Scalar U>
	constexpr Point<3, T> cross(const Point<3,T> &left, const Point<3,T> &right) {
		Point<3, T> result;
		result[0] = left[1]*right[2] - left[2]*right[1];
		result[1] = left[2]*right[0] - left[0]*right[2];
		result[2] = left[0]*right[1] - left[1]*right[0];
		return result;
	}

	template <int dim, Scalar T>
	constexpr T squaredNorm(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {result += point[i] * point[i];}
		return result;
	}

	template <int dim, Scalar T>
	inline T norm2(const Point<dim,T> &point) {
		return std::sqrt(squaredNorm(point));
	}

	template <int dim, Scalar T>
	constexpr T norm1(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {result += std::abs(point[i]);}
		return result;
	}

	template <int dim, Scalar T>
	constexpr T norminfty(const Point<dim,T> &point) {
		T result = 0;
		for (int i=0; i<dim; i++) {
			result = std::max(result, std::abs(point[i]));
		}
		return result;
	}

	template <int dim, Scalar T>
	inline Point<dim,T> normalize(const Point<dim,T> &point) {
		T scale = norm2(point);
		return point / scale;
	}

	//////////////////////////////////////////////////////////
	// Element-wise operations
	//////////////////////////////////////////////////////////
	template <int dim, Scalar T>
	constexpr Point<dim,T> abs(const Point<dim,T> &point) {
		Point<dim,T> result;
		for (int i=0; i<dim; i++) {result[i] = std::abs(point[i]);}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim, T> elmax(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = std::max(left[i], right[i]);}
		return result;
	}

	template <int dim, Scalar T>
	constexpr Point<dim, T> elmin(const Point<dim,T> &left, const Point<dim,T> &right) {
		Point<dim, T> result;
		for (int i=0; i<dim; i++) {result[i] = std::min(left[i], right[i]);}
		return result;
	}

	template <int dim, Scalar T>
	constexpr T max(const Point<dim,T> &point) {
		T result = point[0];
		for (int i=1; i<dim; i++) {result = std::max(result, point[i]);}
		return result;
	}

	template <int dim, Scalar T>
	constexpr T min(const Point<dim,T> &point) {
		T result = point[0];
		for (int i=1; i<dim; i++) {result = std::min(result, point[i]);}
		return result;
	}


	////////////////////////////////////////////////////////////////////////////////
	/// Sum points in careful precision order
	///
	/// @param points A vector of points to add
	///
	/// @tparam W is the input point type
	/// @tparam U is the type that the arithmetic should be done in
	/// @tparam T is the output type
	////////////////////////////////////////////////////////////////////////////////
	template <int dim, Scalar T, Scalar U, Scalar W>
	constexpr Point<dim,T> sorted_sum(const std::vector<Point<dim,W>> &points) noexcept {
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

	////////////////////////////////////////////////////////////////////////////////
	/// Convenient way to call the sorted sum.
	////////////////////////////////////////////////////////////////////////////////
	template <int dim, Scalar T, Scalar U, Scalar W>
	constexpr Point<dim,T> sorted_sum(std::initializer_list<Point<dim,W>> points) noexcept {
	    return sorted_sum<dim, T, U, W>(std::vector<Point<dim,W>>(points.begin(), points.end()));
	}

	/// Print to ostream
	template <int dim, Scalar T>
	std::ostream& operator<<(std::ostream& os, const Point<dim,T> &point) {
		for (int i = 0; i < dim-1; i++) {os << point[i] << " ";}
		os << point[dim-1];
		return os;
	}

}