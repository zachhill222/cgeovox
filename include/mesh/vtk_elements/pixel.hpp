#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>


namespace gv::mesh {
	/* Pixel element node labels
	
	 			2 ----- 3
				|		|
				|		|
				0 ----- 1
	*/

	/////////////////////////////////////////////////
	/// Pixel element
	/// Note that this is not an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// and the center of the element can be found by averaging opposite vertices. Pixel elements must always be a rectangular
	/// subset of a plane that is parallel to a coordinate axis with sides parallel to the remaining two coordinate axes (i.e., faces 
	/// of an axis-aligned bounding box).
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1].
	/// 
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1) is T{0.5}(1-e0)*0.5(1+e1) where e0 and e1 are the cartesian coordinates
	/// in the reference element.
	///
	/// The mapping from the reference element to the actual/mesh element is of the form:
	///
	/// F(e0,e1) = A*[e0, e1]^t + b
	/// 
	/// where A is a 3x2 matrix and b is the location of the center of the mesh element. Because the the map is affine, the Jacobian matrix (J=A) is constant.
	/// In fact sqrt(J^t J) = 0.25*h1*h2 where h1 and h2 are the side-lengths of the mesh element.
	///
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_PIXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_PIXEL(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = PIXEL_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

			using T = typename Point_t::Scalar_t;

			//edge midpoints
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]})); //4 - bottom
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[3]})); //5 - right
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3]})); //6 - top
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[2]})); //7 - left

			//center
			vertices.emplace_back(T{0.5}*(vertices[0]+vertices[3])); //8
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

			switch (child_number) {
				case (0):
					child_nodes[0] = split_node_numbers[0]; //0
					child_nodes[1] = split_node_numbers[4]; //0-1
					child_nodes[2] = split_node_numbers[7]; //0-2
					child_nodes[3] = split_node_numbers[8]; //0-3
					break;
				case (1):
					child_nodes[0] = split_node_numbers[4]; //0-1
					child_nodes[1] = split_node_numbers[1]; //1
					child_nodes[2] = split_node_numbers[8]; //0-3
					child_nodes[3] = split_node_numbers[5]; //1-3
					break;
				case (2):
					child_nodes[0] = split_node_numbers[7]; //0-2
					child_nodes[1] = split_node_numbers[8]; //0-3
					child_nodes[2] = split_node_numbers[2]; //2
					child_nodes[3] = split_node_numbers[6]; //2-3
					break;
				case (3):
					child_nodes[0] = split_node_numbers[8]; //0-3
					child_nodes[1] = split_node_numbers[5]; //1-3
					child_nodes[2] = split_node_numbers[6]; //2-3
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
				face_nodes[1] = this->ELEM.nodes[3];
				break;
			case (2):
				face_nodes[0] = this->ELEM.nodes[3];
				face_nodes[1] = this->ELEM.nodes[2];
				break;
			case (3):
				face_nodes[0] = this->ELEM.nodes[2];
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

			case (1): // Right [1, 3]
				split_face_nodes[0] = split_node_numbers[ 1]; //1
				split_face_nodes[1] = split_node_numbers[ 3]; //3
				split_face_nodes[2] = split_node_numbers[ 5]; //1-3
				break;

			case (2): // Top [3, 2]
				split_face_nodes[0] = split_node_numbers[ 3]; //3
				split_face_nodes[1] = split_node_numbers[ 2]; //2
				split_face_nodes[2] = split_node_numbers[ 6]; //2-3
				break;

			case (3): // Left [2, 0]
				split_face_nodes[0] = split_node_numbers[ 2]; //2
				split_face_nodes[1] = split_node_numbers[ 0]; //0
				split_face_nodes[2] = split_node_numbers[ 7]; //0-2
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}
	};
}