#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>


namespace gv::mesh {
	/* Quad element node labels
	
	 			3 ----- 2
				|		|
				|		|
				0 ----- 1
	*/

	/////////////////////////////////////////////////
	/// Quad element
	/// Note that this is an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// but the center of the element must be found by evaluating the shape functions.
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1].
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1) is 0.5(1-e0)*0.5(1+e1) where e0 and e1 are the cartesian coordinates
	/// in the reference element.
	///
	/// Let N0,...N3 be the shape functions at each vertex. Then the mapping from the reference element to the actual element is given by
	///
	/// F(e0,e1) = v0*N0 + v1*N1 + v2*N2 + v3*N3
	///
	/// Where v0,...,v3 are the vertices of the actual/mesh element in R3.
	/// Note e0=e1=0 at the center of the reference element so that N0=N1=N2=N3=0.25 at the center. Mapping into the mesh element gives us the center
	/// 0.25*(v0+v1+v2+v3), which cannot be simplified to the average of any two opposite vertices (v0 may be moved nearly arbitrarily, which changes the
	/// face center but does not change the evaluation of 0.5*(v1+v3).)
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_QUAD : public VTK_ELEMENT<Point_t> {
	public:
		VTK_QUAD(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = QUAD_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

			using T = typename Point_t::Scalar_t;
			
			//edge midpoints
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]})); //4 - bottom (B)
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[2]})); //5 - right (R)
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3]})); //6 - top (T)
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3]})); //7 - left (L)

			//center
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[2],vertices[3]})); //8 (C)
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

			switch (child_number) {
				case (0):
					child_nodes[0] = split_node_numbers[0]; //0
					child_nodes[1] = split_node_numbers[4]; //B
					child_nodes[2] = split_node_numbers[8]; //C
					child_nodes[3] = split_node_numbers[7]; //L
					break;
				case (1):
					child_nodes[0] = split_node_numbers[4]; //B
					child_nodes[1] = split_node_numbers[1]; //1
					child_nodes[2] = split_node_numbers[5]; //R
					child_nodes[3] = split_node_numbers[8]; //C
					break;
				case (2):
					child_nodes[0] = split_node_numbers[8]; //C
					child_nodes[1] = split_node_numbers[5]; //R
					child_nodes[2] = split_node_numbers[2]; //2
					child_nodes[3] = split_node_numbers[6]; //T
					break;
				case (3):
					child_nodes[0] = split_node_numbers[7]; //L
					child_nodes[1] = split_node_numbers[8]; //C
					child_nodes[2] = split_node_numbers[6]; //T
					child_nodes[3] = split_node_numbers[3]; //3
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}
		}

		void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const override {
			face_nodes.resize(2);
			switch (face_number) {
			case (0):
				face_nodes[0] = this->ELEM.nodes[0];
				face_nodes[1] = this->ELEM.nodes[1];
				break;
			case (1):
				face_nodes[0] = this->ELEM.nodes[1];
				face_nodes[1] = this->ELEM.nodes[2];
				break;
			case (2):
				face_nodes[0] = this->ELEM.nodes[2];
				face_nodes[1] = this->ELEM.nodes[3];
				break;
			case (3):
				face_nodes[0] = this->ELEM.nodes[3];
				face_nodes[1] = this->ELEM.nodes[0];
				break;
			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}

		void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const override {
			split_face_nodes.resize(vtk_n_nodes_when_split(vtk_face_id(VTK_ID)));
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));

			switch (face_number) {
				case (0): // Bottom [0, 1]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 1]; //1
				split_face_nodes[2] = split_node_numbers[ 4]; //0-1
				break;

			case (1): // Right [1, 2]
				split_face_nodes[0] = split_node_numbers[ 1]; //1
				split_face_nodes[1] = split_node_numbers[ 2]; //2
				split_face_nodes[2] = split_node_numbers[ 5]; //1-2
				break;

			case (2): // Top [2, 3]
				split_face_nodes[0] = split_node_numbers[ 2]; //2
				split_face_nodes[1] = split_node_numbers[ 3]; //3
				split_face_nodes[2] = split_node_numbers[ 6]; //2-3
				break;

			case (3): // Left [3, 0]
				split_face_nodes[0] = split_node_numbers[ 3]; //3
				split_face_nodes[1] = split_node_numbers[ 0]; //0
				split_face_nodes[2] = split_node_numbers[ 7]; //0-3
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}
	};
}