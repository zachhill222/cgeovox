#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>

namespace gv::mesh {
	/////////////////////////////////////////////////
	/// Interface for VTK element types
	///
	/// @tparam Point_t The type (e.g. gv::util::Point<3,double>) of the coordinates of the vertices in the mesh
	/////////////////////////////////////////////////
	template<typename Point_t>
	class VTK_ELEMENT {
	public:
		VTK_ELEMENT(const BasicElement &elem) : ELEM(elem) {}
		virtual ~VTK_ELEMENT() {}
		const BasicElement &ELEM;
		virtual void split(std::vector<Point_t>& vertex_coords) const = 0;
		virtual void getChildVertices(std::vector<size_t>& child_nodes, const int child_number, const std::vector<size_t>& split_node_numbers) const = 0;
		virtual void getFaceVertices(std::vector<size_t>& face_nodes, const int face_number) const = 0;
		virtual void getSplitFaceVertices(std::vector<size_t>& split_face_nodes, const int face_number, const std::vector<size_t>& split_node_numbers) const = 0;
		BasicElement getFace(const int face_number) const {
			BasicElement face(vtk_face_id(this->ELEM.vtkID));
			getFaceVertices(face.vertices, face_number);
			return face;
		}
		virtual bool isInterior(const std::vector<Point_t>& vertex_coords, const Point_t& coord) const = 0;

		//basis function, gradient, and jacobian methods must also be defined for each element
	};
}