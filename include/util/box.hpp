#pragma once

#include "util/point.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>

namespace gv::util{

	template <int dim=3, typename T=double>
	class Box{
	protected:
		Point<dim,T> _low;
		Point<dim,T> _high;

	public:
		using Point_t = gv::util::Point<dim,T>;

		Box() : _low(Point_t(-1.0)), _high(Point_t(1.0)) {}
		Box(const T low, const T high) : _low(Point_t(low)), _high(Point_t(high)) {}

		Box(const Point_t &vertex1, const Point_t &vertex2) : _low(elmin(vertex1,vertex2)), _high(elmax(vertex1,vertex2))
		{
			assert(_low<_high);
		}

		//copy constructor
		Box(const Box &other){
			_low = other.low();
			_high = other.high();
		}

		//move constructor
		Box(Box &&other)
		{
			_low = std::move(other._low);
			_high = std::move(other._high);
		}

		//destructor
		~Box() {}


		//move assignment
		Box& operator=(Box &&other) noexcept
		{
			if (this!=&other)
			{
				_low  = std::move(other._low);
				_high = std::move(other._high);
			}
			return *this;
		}

		//copy assignment
		Box& operator=(const Box &other)
		{
			_low  = other._low;
			_high = other._high;
			return *this;
		}
		


		//////////////////////////
		/////// ATTRIBUTES ///////
		//////////////////////////
		inline Point_t low() const {return _low;}
		inline Point_t high() const {return _high;}
		inline Point_t center() const {return 0.5*(_low+_high);}
		inline Point_t sidelength() const {return _high-_low;}
		inline double diameter() const {return gv::util::norm2(_high-_low);}

		///Get i-th vertex in vtk pixel/voxel order.
		Point_t operator[](const int idx) const {
			Point_t vertex;
			int p = idx;
			int r = 0;
			for (int i=0; i<dim; i++){
				r = p%2;
				p = p/2;
				if (r){vertex[i] = _high[i];}
				else {vertex[i] = _low[i];}
			}
			return vertex;
		}


		Point_t voxelijk(const int idx) const {
			Point_t vertex;
			int p = idx;
			int r = 0;
			for (int i=0; i<dim; i++){
				r = p%2;
				p = p/2;
				if (r){vertex[i] = 1.0;}
				else {vertex[i] = 0.0;}
			}
			return vertex;
		}

		///Get i-th vertex in vtk quad/hexahedron order.
		Point_t hexvertex(const int idx) const{
			switch (idx){
			case 2: return operator[](3);
			case 3: return operator[](2);
			case 6: return operator[](7);
			case 7: return operator[](6);
			default: return operator[](idx);
			}
		}

		///Get i-th vertex in vtk pixel/voxel order.
		inline Point_t voxelvertex(const int idx) const {return (*this)[idx];}




		///////////////////////////////////////////////
		/////// CONTAINMENT AND INTERSECTION //////////
		///////////////////////////////////////////////
		///Check if point is in the closed box.
		inline bool contains(const Point_t &point) const {return _low<=point and point<=_high;}
		///Check if point is in the open box.
		inline bool contains_strict(const Point_t &point) const {return _low<point and point<_high;}
		///Check if this box contains the other.
		inline bool contains(const Box<dim> &other) const {return _low<=other.low() and other.high()<=_high;}
		///Check if this box intersects the other.
		bool intersects(const Box<dim> &other) const
		{
			for (int i=0; i<dim; i++)
			{
				bool axis_overlap = _high[i]>=other._low[i] and other._high[i]>=_low[i];
				if (!axis_overlap) {return false;}
			}
			return true;
		}

		///Find a location of the supporting hyperplane with the given direction. This maximizes dot(x,direction) over all points x in the box.
		Point_t support(const Point_t &direction) const{
			double maxdot = dot(direction, (*this)[0]);
			int maxind = 0;

			double tempdot;
			for (int i=1; i<std::pow(2,dim); i++){
				tempdot = dot(direction, (*this)[i]);
				if (tempdot > maxdot){
					maxdot = tempdot;
					maxind = i;
				}
			}
			return operator[](maxind); 
		}


		///////////////////////////////////////////////
		////////// SHIFTING AND SCALING ///////////////
		///////////////////////////////////////////////
		Box<dim>* operator+=(const Point_t &shift){
			_low+=shift;
			_high+=shift;
			return this;
		}
		Box<dim> operator+(const Point_t &shift) const{
			return Box(_low+shift, _high+shift);
		}
		Box<dim>* operator-=(const Point_t &shift){
			_low-=shift;
			_high-=shift;
			return this;
		}
		Box<dim> operator-(const Point_t &shift) const{
			return Box(_low-shift, _high-shift);
		}

		///Scale box towards center.
		Box& operator*=(const double& scale){
			Point_t _center = center();
			_low = _center + scale*(_low-_center);
			_high = _center + scale*(_high-_center);
			return *this;
		}

		Box operator*(const double& scale) const{
			Point_t _center = center();
			return Box(_center+scale*(_low-_center), _center+scale*(_high-_center));
		}

		bool operator==(const Box &other) const {
			return _low==other.low() and _high==other.high();
		}

		inline Box<dim>* operator/=(const double& scale){return operator*=(1.0/scale);}
		inline Box<dim> operator/(const double& scale) const{return operator*(1.0/scale);}

		///Enlarge this box so that it contains the other.
		Box& combine(const Box<dim>& other){
			Point_t _newlow = elmin(_low, other.low());
			Point_t _newhigh = elmax(_high, other.high());
			_low = _newlow;
			_high = _newhigh;
			return *this;
		}

		// std::string str() const{
		// 	std::stringstream ss;
		// 	for (int i=0; i<std::pow(2,dim); i++){
		// 		ss << i << ": " << operator[](i) << "\n";
		// 	}
		// 	return ss.str();
		// }

	
	};

	//LHS scalar multiplication
	template <int dim>
	Box<dim> operator*(const double &scale, const Box<dim> &box)
	{
		Point<dim,double> _low = box.center() + scale*(box.low() - box.center());
		Point<dim,double> _high = box.center() + scale*(box.high() - box.center());
		return Box<dim> {_low, _high};
	}

	//Distance to box
	template <int dim>
	double distance_squared(const Box<dim> &box, const Point<dim,double> &point)
	{
		if (box.contains(point)) {return 0;}

		//get supporting point for tangent plane
		Point<dim,double> normal = (point - box.center()).normalized();
		Point<dim,double> support_point = box.support(normal);

		//get point in supporting plane closest to specified point
		Point<dim,double> closest = point - dot(point,normal)*normal;

		//return distance squared to specified point
		if (box.contains(closest)) {return (closest-point).normSquared();}
		return (support_point-point).normSquared();
	}

	//print to stream
	template<int dim>
	std::ostream& operator<<(std::ostream& os, const Box<dim>& box)
	{
		return os << "(" << box.low() << ") to (" << box.high() << ")";
	}
}



