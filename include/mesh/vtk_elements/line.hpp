#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>

namespace gv::mesh {
	/* Voxel element node labels
	
		0 ------ 1  
	*/

	/////////////////////////////////////////////////
	/// Line element
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_LINE : public VTK_ELEMENT<Point_t>{
	public:
		VTK_LINE(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = LINE_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));

			using T = typename Point_t::Scalar_t;
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]}));
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

			switch (child_number) {
				case (0):
					child_nodes[0] = split_node_numbers[0];
					child_nodes[1] = split_node_numbers[2];
					break;
				case (1):
					child_nodes[0] = split_node_numbers[2];
					child_nodes[1] = split_node_numbers[1];
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
				}
		}

		void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const override {
			face_nodes.resize(1);
			face_nodes[0] = this->ELEM.nodes[face_number];
		}

		void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const override {
			split_face_nodes.resize(vtk_n_nodes_when_split(vtk_face_id(VTK_ID)));
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));

			switch (face_number) {
				case (0): // Left [0, 2]
				split_face_nodes[0] = split_node_numbers[0];
				split_face_nodes[1] = split_node_numbers[2];
				break;

			case (1): // Right [2, 1]
				split_face_nodes[0] = split_node_numbers[2];
				split_face_nodes[1] = split_node_numbers[1];
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}
	};
}