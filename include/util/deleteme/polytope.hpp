#pragma once

#include "util/point.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>


namespace gv::util{
	//POLYTOPE DEFINITION
	template<int dim=3, typename T=double>
	class Polytope{
	public:
		Polytope(const int npts){
			_points.reserve(npts);
		}

		Polytope(std::initializer_list<Point<dim>> list) : _points(list) {}

		Point<dim,T> operator[](int idx) const;
		Point<dim,T>& operator[](int idx);

		int len() const;

		void push_back(const Point<dim,T>& point);

		Point<dim,T> support(const Point<dim,T>& direction) const;

		Point<dim,T> center() const;

		void print(std::ostream& stream) const;

	protected:
		std::vector<Point<dim>> _points;
	};


	//SIMPLEX DEFINITION
	template<int dim=3, typename T=double>
	class Simplex : public Polytope<dim,T> {
	public:
		Simplex(): Polytope<dim,T>(dim+1) {
			this->_points.push_back(Point<dim,T>()); //default constructor is all zeros
			for (int i=1; i<dim+1; i++){
				this->_points.push_back(Point<dim,T>());
				this->_points[i][i-1] = 1.0;
			}
		}

		Simplex(std::initializer_list<Point<dim,T>> list): Polytope<dim>(list) {}

		Simplex(const Simplex<dim>& other): Polytope<dim>(dim+1){
			for (int i=0; i<other.len(); i++){
				this->_points.push_back(other[i]);
			}
		}
	};





	//POLYTOPE IMPLEMENTATION
	template<int dim, typename T>
	Point<dim,T> Polytope<dim,T>::operator[](int idx) const{
		return _points[idx];
	}
	
	template<int dim, typename T>
	Point<dim,T>& Polytope<dim,T>::operator[](int idx){
		return _points[idx%_points.size()];
	}
	
	template<int dim, typename T>
	int Polytope<dim,T>::len() const{
		return _points.size();
	}
	
	template<int dim, typename T>
	void Polytope<dim,T>::push_back(const Point<dim,T>& point){
		_points.push_back(point);
	}
	
	template<int dim, typename T>
	Point<dim,T> Polytope<dim,T>::support(const Point<dim,T>& direction) const{
		double maxdot = dot(direction,_points[0]);
		int maxind = 0;
		double tempdot;

		for (int i=1; i<len(); i++){
			tempdot = dot(direction,_points[i]);
			if (tempdot > maxdot){
				maxdot = tempdot;
				maxind = i;
			}
		}

		return _points[maxind];
	}
	
	template<int dim, typename T>
	Point<dim,T> Polytope<dim,T>::center() const{
		T C = 1.0/_points.size();
		Point<dim,T> result = C*_points[0];

		for (size_t i=0; i<_points.size(); i++){
			result += C*_points[i];
		}

		return result;
	}
	
	template<int dim, typename T>
	void Polytope<dim,T>::print(std::ostream& stream) const{
		for (int i=0; i<len(); i++){
			stream << i << ": ";
			stream << _points[i] << std::endl;
		}
	}
}
