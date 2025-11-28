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
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_ELEMENT {
	public:
		VTK_ELEMENT(const BasicElement &elem) : ELEM(elem) {}
		virtual ~VTK_ELEMENT() {}
		const BasicElement &ELEM;
		virtual void split(std::vector<Point_t> &vertices) const = 0;
		virtual void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const = 0;
		virtual void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const = 0;
		virtual void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const = 0;
		BasicElement getFace(const int face_number) const {
			BasicElement face(vtk_face_id(this->ELEM.vtkID));
			getFaceNodes(face.nodes, face_number);
			return face;
		}
	};
}