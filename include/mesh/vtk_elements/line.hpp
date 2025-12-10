#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/matrix.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>
#include <functional>

namespace gv::mesh {
	/* Voxel element node labels
	
		0 ------ 1  
	*/

	/////////////////////////////////////////////////
	/// Line element
	/////////////////////////////////////////////////
	template <typename Vertex_t>
	class VTK_LINE : public VTK_ELEMENT<Vertex_t>{
	public:
		VTK_LINE(const BasicElement &elem) : VTK_ELEMENT<Vertex_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID  = LINE_VTK_ID;
		static constexpr int REF_DIM = 1; //dimension of the reference element
		using Scalar_t    = typename Vertex_t::Scalar_t;
		using RefPoint_t  = gv::util::Point<REF_DIM, Scalar_t>; //type of point in the reference element
		using Matrix_t    = gv::util::Matrix<3,REF_DIM,Scalar_t>; //dimensions of the jacobian matrix (output space is always R3)

		using ScalarFun_t = std::function<Scalar_t(RefPoint_t)>; //function type to evaluate a basis in the element
		using VectorFun_t = std::function<Vertex_t(RefPoint_t)>; //function type to evaluate the gradient of a basis function
		using MatrixFun_t = std::function<Matrix_t(RefPoint_t)>; //function type to evaluate the jacobian of the isoparametric mapping

		void split(std::vector<Vertex_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));

			using T = typename Vertex_t::Scalar_t;
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]}));
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

		bool isInterior(const std::vector<Vertex_t>& vertices, const Vertex_t& coord) const override {
			assert(false); //probably shouldn't be using this...
			using T = typename Vertex_t::Scalar_t;

			//check if coord is a convex combination of vertices[0] and vertices[1].
			//if so, coord = (1-t)*vertices[0] + t*vertices[1] = vertices[0] + t*(vertices[1]-vertices[0])
			T t = (coord[0] - vertices[0][0]) / (vertices[1][0] - vertices[0][0]);
			return coord == vertices[0] + t*(vertices[1]-vertices[0]);
		}
	};
}