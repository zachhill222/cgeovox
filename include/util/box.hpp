#pragma once

#include "util/point.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <cmath>

namespace gv::util{
	template <int dim=3>
	class Box{
	public:
		Box() {}

		Box(const Point<dim> &vertex1, const Point<dim> &vertex2){
			_low = elmin(vertex1, vertex2);
			_high = elmax(vertex1, vertex2);
			
			if (!(_low<_high)){
				throw std::invalid_argument("Box: low<high");
			}
		}

		Box(const Box &box1, const Box &box2){
			_low = el_min(box1.low(), box2.low());
			_high = el_max(box1.high(), box2.high());
			if (!(_low<_high)){
				throw std::invalid_argument("Box: low<high");
			}
		}

		Box(const Box &other){
			_low = other.low();
			_high = other.high();
		}


		//////////////////////////
		/////// ATTRIBUTES ///////
		//////////////////////////
		inline Point<dim> low() const {return _low;}
		inline Point<dim> high() const {return _high;}
		inline Point<dim> center() const {return 0.5*(_low+_high);}
		inline Point<dim> sidelength() const {return _high-_low;}

		///Get i-th vertex in vtk pixel/voxel order.
		Point<dim> operator[](const int idx) const {
			Point<dim> vertex;
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

		void setlow(const Point<dim> &newlow){
			Point<dim> _newlow = el_min(newlow, _high);
			Point<dim> _newhigh = el_max(newlow, _high);
			if (_newlow < _newhigh){
				_low = _newlow;
				_high = _newhigh;
			}else{
				throw std::out_of_range("Box: can't move _low to newlow");
			}
		}

		void sethigh(const Point<dim> &newhigh){
			Point<dim> _newlow = el_min(newhigh, _high);
			Point<dim> _newhigh = el_max(newhigh, _high);
			if (_newlow < _newhigh){
				_low = _newlow;
				_high = _newhigh;
			}else{
				std::out_of_range("Box: can't move _high to newhigh");
			}
		}

		///Get i-th vertex in vtk quad/hexahedron order.
		Point<dim> hexvertex(const int idx) const{
			switch (idx){
			case 2: return operator[](3);
			case 3: return operator[](2);
			case 6: return operator[](7);
			case 7: return operator[](6);
			default: return operator[](idx);
			}
		}

		///Get i-th vertex in vtk pixel/voxel order.
		Point<dim> voxelvertex(const int idx) const {return operator[](idx);}


		///////////////////////////////////////////////
		/////// CONTAINMENT AND INTERSECTION //////////
		///////////////////////////////////////////////
		///Check if point is in the closed box.
		inline bool contains(const Point<dim> &point) const {return _low<=point and point<=_high;}
		///Check if point is in the open box.
		inline bool contains_strict(const Point<dim> &point) const {return _low<point and point<_high;}
		///Check if this box contains the other.
		inline bool contains(const Box<dim> &other) const {return _low<=other.low() and other.high()<=_high;}
		///Check if this box intersects the other.
		bool intersects(const Box<dim> &other) const{
			for (int i=0; i<std::pow(2,dim); i++){
				if (this->contains(other[i])){return true;}
				if (other.contains(operator[](i))){return true;}
			}
			return false;
		}

		///Find a location of the supporting hyperplane with the given direction. This maximizes dot(x,direction) over all points x in the box.
		Point<dim> support(const Point<dim> &direction) const{
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
		Box<dim>* operator+=(const Point<dim> &shift){
			_low+=shift;
			_high+=shift;
			return this;
		}
		Box<dim> operator+(const Point<dim> &shift) const{
			return Box(_low+shift, _high+shift);
		}
		Box<dim>* operator-=(const Point<dim> &shift){
			_low-=shift;
			_high-=shift;
			return this;
		}
		Box<dim> operator-(const Point<dim> &shift) const{
			return Box(_low-shift, _high-shift);
		}

		///Scale box towards center.
		Box<dim>* operator*=(const double& scale){
			Point<dim> _center = center();
			_low = _center + scale*(_low-_center);
			_high = _center + scale*(_high-_center);
			return this;
		}

		Box<dim> operator*(const double& scale) const{
			Point<dim> _center = center();
			return Box<dim>(_center+scale*(_low-_center), _center+scale*(_high-_center));
		}

		inline Box<dim>* operator/=(const double& scale){return operator*=(1.0/scale);}
		inline Box<dim> operator/(const double& scale) const{return operator*(1.0/scale);}

		///Enlarge this box so that it contains the other.
		Box<dim>* combine(const Box<dim>& other){
			Point<dim> _newlow = elmin(_low, other.low());
			Point<dim> _newhigh = elmax(_high, other.high());
			_low = _newlow;
			_high = _newhigh;
			return this;
		}

		std::string tostr() const{
			std::stringstream ss;
			for (int i=0; i<std::pow(2,dim); i++){
				ss << i << ": " << operator[](i) << "\n";
			}
			return ss.str();
		}

	private:
		Point<dim,double> _low;
		Point<dim,double> _high;
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
}



