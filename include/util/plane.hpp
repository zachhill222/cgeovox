#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"

#include <iostream>


template <typename T=double>
using Point_t = gv::util::Point<3,T>;


namespace gv::util{

	template <typename T=double>
	class Plane{
	public:
		Plane(): _origin(Point_t<T>{0,0,0}), _normal(Point_t<T>{0,0,1}) {calcbasis();}
		Plane( const Point_t<T>& origin, const Point_t<T>& normal): _origin(origin), _normal(normalize(normal)) {calcbasis();}
		Plane( const Point_t<T>& p1, const Point_t<T>& p2, const Point_t<T>& p3): _origin(p1), _normal(normalize(cross(p3-p1,p2-p1))) {calcbasis();}

		T dist(const Point_t<T>& point) const; //signed distence to the plane

		Point_t<T> project(const Point_t<T>& point) const; //project point from global coordinates to the plane along the _normal direction. return a 2D point in local coordinates.

		Point_t<T> tolocal(const Point_t<T>& point) const; //write a point in global coordinates in terms of local (_basis[0], _basis[1], _normal) coordinates.

		Point_t<T> toglobal(const Point_t<T>& point) const; //write a local point in the global coordinate system (works for 2D points on the plane or 3D point in local coordinates).

	private:
		Point_t<T> _origin;
		Point_t<T> _normal;
		Point_t<T> _basis[2];

		void calcbasis();
	};

	template <typename T>
	void Plane<T>::calcbasis()
	{
		Point_t<T> vec1, vec2;

		vec1 = cross(_normal,Point_t<T>{1,0,0});
		vec2 = cross(_normal,Point_t<T>{0,1,0});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;
		}

		vec2 = cross(_normal,Point_t<T>{0,0,1});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;}

		_basis[0] = vec1;
		_basis[1] = cross(_normal,_basis[0]);
	}


	template <typename T>
	T Plane<T>::dist(const Point_t<T>& point) const {return dot(point-_origin,_normal);}

	template <typename T>
	Point_t<T> Plane<T>::project(const Point_t<T>& point) const
	{
		Point_t<T> local = tolocal(point); //write point in local coordinates
		return local[0]*_basis[0] + local[1]*_basis[1];
	}

	template <typename T>
	Point_t<T> Plane<T>::tolocal(const Point_t<T>& point) const
	{
		Point_t<T> shift = point - _origin;
		T a = dot(shift,_basis[0]);
		T b = dot(shift,_basis[1]);
		T c = dot(shift,_normal);
		return Point_t<T>(a,b,c);
	}

	template <typename T>
	Point_t<T> Plane<T>::toglobal(const Point_t<T>& point) const
	{
		Point_t<T> result = _origin + point[0]*_basis[0] + point[1]*_basis[1] + point[2]*_normal;
		return result;
	}
}