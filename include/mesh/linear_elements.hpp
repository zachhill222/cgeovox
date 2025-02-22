#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <initializer_list>

namespace gv::mesh{

	//base element
	template <size_t n_vertex>
	class Element{
	protected:
		std::vector<const gv::util::Point<3,double>*> _vertices; //vertex locations in real space, use polynomial mapping from reference cell.

	public:
		Element() {}
		Element(const std::initializer_list<const gv::util::Point<3,double>*> &list) : _vertices(list){}

		inline const gv::util::Point<3,double>* operator[](int idx) const {return _vertices[idx];}

		void print() const
		{
			for (size_t i=0; i<_vertices.size(); i++)
			{
				std::cout << *this->_vertices[i] << " ";
			}
			std::cout << std::endl;
		}
	};

	class Voxel : public Element<8>{
	public:
		Voxel() : Element<8>() {}
		Voxel(const std::initializer_list<const gv::util::Point<3,double>*> list) : Element<8>(list) {}

		static const int vtkID = 11;
	};


	class Hexahedron : public Element<8>{
	public:
		Hexahedron() : Element<8>() {}
		Hexahedron(const std::initializer_list<const gv::util::Point<3,double>*> list) : Element<8>(list) {}

		static const int vtkID = 12;
	};
}
