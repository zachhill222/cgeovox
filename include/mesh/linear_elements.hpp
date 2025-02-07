#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <initializer_list>

namespace gv::mesh{
	//typedefs
	template <size_t n_vertex, typename T>
	class Element;

	template <typename T>
	class Voxel;

	template <typename T>
	class Pixel;

	template <typename T>
	class Hexahedron;

	template <typename T>
	class Quad;

	//base element
	template <size_t n_vertex, typename T=double>
	class Element{
	protected:
		gv::util::Point<3,T>* _vertices[n_vertex] {NULL}; //vertex locations in real space, use polynomial mapping from reference cell.

	public:
		Element() {}
		Element(const std::initializer_list<gv::util::Point<3,T>*> &list) : _vertices(list){}

		inline gv::util::Point<3,T>* operator[](int idx){return _vertices[idx];}
		inline const gv::util::Point<3,T>* operator[](int idx) const {return _vertices[idx];}
	};


	template <typename T=double>
	class Voxel : public Element<8,T>{
	public:
		Voxel() : Element<8,T>() {}
		Voxel(const std::initializer_list<gv::util::Point<3,T>*> list) : Element<8,T>(list) {}

		static const int vtkID = 11;

		Pixel<T> getface(const int face) const
		{
			switch (face){
			case 0:
				return Pixel<T> {this->_vertices[1], this->_vertices[0], this->_vertices[3], this->_vertices[2]};
				break;
			case 1:
				return Pixel<T> {this->_vertices[0], this->_vertices[1], this->_vertices[4], this->_vertices[5]};
				break;
			case 2:
				return Pixel<T> {this->_vertices[4], this->_vertices[5], this->_vertices[6], this->_vertices[7]};
				break;
			case 3:
				return Pixel<T> {this->_vertices[6], this->_vertices[7], this->_vertices[2], this->_vertices[3]};
				break;
			case 4:
				return Pixel<T> {this->_vertices[0], this->_vertices[4], this->_vertices[2], this->_vertices[6]};
				break;
			case 5:
				return Pixel<T> {this->_vertices[5], this->_vertices[1], this->_vertices[7], this->_vertices[3]};
				break;
			}
		}
	};


	template <typename T=double>
	class Hexahedron : public Element<8,T>{
	public:
		Hexahedron() : Element<8,T>() {}
		Hexahedron(const std::initializer_list<gv::util::Point<3,T>*> list) : Element<8,T>(list) {}

		static const int vtkID = 12;
		Quad<T> getface(const int face) const
		{
			switch (face){
			case 0:
				return Quad<T> {this->_vertices[1], this->_vertices[0], this->_vertices[3], this->_vertices[2]};
				break;
			case 1:
				return Quad<T> {this->_vertices[0], this->_vertices[1], this->_vertices[5], this->_vertices[4]};
				break;
			case 2:
				return Quad<T> {this->_vertices[4], this->_vertices[5], this->_vertices[6], this->_vertices[6]};
				break;
			case 3:
				return Quad<T> {this->_vertices[7], this->_vertices[6], this->_vertices[2], this->_vertices[3]};
				break;
			case 4:
				return Quad<T> {this->_vertices[0], this->_vertices[4], this->_vertices[7], this->_vertices[3]};
				break;
			case 5:
				return Quad<T> {this->_vertices[5], this->_vertices[1], this->_vertices[2], this->_vertices[6]};
				break;
			}
		}
	};


	template <typename T=double>
	class Pixel : public Element<4,T>{
		Pixel() : Element<4,T>() {}
		Pixel(const std::initializer_list<gv::util::Point<3,T>*> list) : Element<8,T>(list) {}
		Pixel(const Pixel<T>* other)
		{
			for (int i=0; i<4; i++) {this->_vertices[i] = other[i];}
		}

		static const int vtkID = 8;
		gv::util::Point<3,T> normal() const
		{
			gv::util::Point<3,T> u = *this->_vertices[1] - *this->_vertices[0];
			gv::util::Point<3,T> v = *this->_vertices[2] - *this->_vertices[0];
			return gv::util::cross(u,v).normalized();
		}
	};


	template <typename T=double>
	class Quad : public Element<4,T>{
		Quad() : Element<4,T>() {}
		Quad(const std::initializer_list<gv::util::Point<3,T>*> list) : Element<4,T>(list) {}
		Quad(const Quad<T>* other)
		{
			for (int i=0; i<4; i++) {this->_vertices[i] = other[i];}
		}
		Quad(const Pixel<T>* pixel){
			this->_vertices[0] = *pixel[0];
			this->_vertices[1] = *pixel[1];
			this->_vertices[2] = *pixel[3];
			this->_vertices[3] = *pixel[2];
		}

		static const int vtkID = 9;

		gv::util::Point<3,T> normal() const
		{
			gv::util::Point<3,T> u = *this->_vertices[1] - *this->_vertices[0];
			gv::util::Point<3,T> v = *this->_vertices[3] - *this->_vertices[0];
			return gv::util::cross(u,v).normalized();
		}
	};
}
