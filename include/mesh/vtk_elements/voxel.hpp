#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>

#include <cassert>


namespace gv::mesh {
	// Voxel element node labels
	//
	// 			2 ------- 3
	//			|\	      |\
	//			| \		  | \
	//			0 -\----- 1  \
	//			 \	\      \  \
	//			  \	 6 ------- 7
	//			   \ |		 \ |
	//				\|        \|
	//				 4 ------- 5
	//

	/////////////////////////////////////////////////
	/// Voxel element
	/// Note that this is not an iso-parametric element.
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1]x[-1,1].
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1,+1) is 0.5(1-e0)*0.5(1+e1)*0.5(1+e2) where e0, e1, and e2
	/// are the cartesian coordinates in the reference element.
	///
	/// Let N0,...,N7 be the shape functions at each vertex. Then the mapping from the reference element to the actual element is given by
	///
	/// F(e0,e1) = v0*N0 + v1*N1 + ... + v7*N7
	///
	/// Where v0,...,v7 are the vertices of the actual/mesh element in R3.
	/// Note e0=e1=e2=0 at the center of the reference element so that N0=N1=N2=N3=0.125 at the center. Mapping into the mesh element gives us the center
	/// 0.125*(v0+...+v7), which cannot be simplified to the average of any two opposite vertices.
	/// Note by a similar line of reasoning, the center of each face must be found by averaging all four vertices.
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_VOXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_VOXEL(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = VOXEL_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));


			using T = typename Point_t::Scalar_t;

			//edge midpoints
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]})); //8  - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[3]})); //9  - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3]})); //10 - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[2]})); //11 - back face

			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[4]})); //12 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[6]})); //13 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[3],vertices[7]})); //14 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[5]})); //15 - connecting edge
			
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[5]})); //16 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[5],vertices[7]})); //17 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[6],vertices[7]})); //18 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[6]})); //19 - front face

			//face midpoints
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[6]})); //20 - left face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[7]})); //21 - right face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[7]})); //22 - top face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[5]})); //23 - bottom face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3]})); //24 - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[7]})); //25 - front face

			//center
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[7]})); //26
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

			switch (child_number) {
				case (0): //voxel element containing original vertex 0
					child_nodes[0] = split_node_numbers[ 0]; //0
					child_nodes[1] = split_node_numbers[ 8]; //0-1
					child_nodes[2] = split_node_numbers[11]; //0-2
					child_nodes[3] = split_node_numbers[24]; //0-3
					child_nodes[4] = split_node_numbers[12]; //0-4
					child_nodes[5] = split_node_numbers[23]; //0-5
					child_nodes[6] = split_node_numbers[20]; //0-6
					child_nodes[7] = split_node_numbers[26]; //0-7
					break;

				case (1): //voxel element containing original vertex 1
					child_nodes[0] = split_node_numbers[ 8]; //0-1
					child_nodes[1] = split_node_numbers[ 1]; //1
					child_nodes[2] = split_node_numbers[24]; //0-3
					child_nodes[3] = split_node_numbers[ 9]; //1-3
					child_nodes[4] = split_node_numbers[23]; //0-5
					child_nodes[5] = split_node_numbers[15]; //1-5
					child_nodes[6] = split_node_numbers[26]; //0-7
					child_nodes[7] = split_node_numbers[21]; //1-7
					break;

				case (2): //voxel element containing original vertex 2
					child_nodes[0] = split_node_numbers[11]; //0-2
					child_nodes[1] = split_node_numbers[24]; //0-3
					child_nodes[2] = split_node_numbers[ 2]; //2
					child_nodes[3] = split_node_numbers[10]; //2-3
					child_nodes[4] = split_node_numbers[20]; //0-6
					child_nodes[5] = split_node_numbers[26]; //0-7
					child_nodes[6] = split_node_numbers[13]; //2-6
					child_nodes[7] = split_node_numbers[22]; //2-7
					break;

				case (3): //voxel element containing original vertex 3
					child_nodes[0] = split_node_numbers[24]; //0-3
					child_nodes[1] = split_node_numbers[ 9]; //1-3
					child_nodes[2] = split_node_numbers[10]; //2-3
					child_nodes[3] = split_node_numbers[ 3]; //3
					child_nodes[4] = split_node_numbers[26]; //0-7
					child_nodes[5] = split_node_numbers[21]; //1-7
					child_nodes[6] = split_node_numbers[22]; //2-7
					child_nodes[7] = split_node_numbers[14]; //3-7
					break;

				case (4): //voxel element containing original vertex 4
					child_nodes[0] = split_node_numbers[12]; //0-4
					child_nodes[1] = split_node_numbers[23]; //0-5
					child_nodes[2] = split_node_numbers[20]; //0-6
					child_nodes[3] = split_node_numbers[26]; //0-7
					child_nodes[4] = split_node_numbers[ 4]; //4
					child_nodes[5] = split_node_numbers[16]; //4-5
					child_nodes[6] = split_node_numbers[19]; //4-6
					child_nodes[7] = split_node_numbers[25]; //4-7
					break;

				case (5): //voxel element containing original vertex 5
					child_nodes[0] = split_node_numbers[23]; //0-5
					child_nodes[1] = split_node_numbers[15]; //1-5
					child_nodes[2] = split_node_numbers[26]; //0-7
					child_nodes[3] = split_node_numbers[21]; //1-7
					child_nodes[4] = split_node_numbers[16]; //4-5
					child_nodes[5] = split_node_numbers[ 5]; //5
					child_nodes[6] = split_node_numbers[25]; //4-7
					child_nodes[7] = split_node_numbers[17]; //5-7
					break;

				case (6): //voxel element containing original vertex 6
					child_nodes[0] = split_node_numbers[20]; //0-6
					child_nodes[1] = split_node_numbers[26]; //0-7
					child_nodes[2] = split_node_numbers[13]; //2-6
					child_nodes[3] = split_node_numbers[22]; //2-7
					child_nodes[4] = split_node_numbers[19]; //4-6
					child_nodes[5] = split_node_numbers[25]; //4-7
					child_nodes[6] = split_node_numbers[ 6]; //6
					child_nodes[7] = split_node_numbers[18]; //6-7
					break;

				case (7): //voxel element containing original vertex 7
					child_nodes[0] = split_node_numbers[26]; //0-7
					child_nodes[1] = split_node_numbers[21]; //1-7
					child_nodes[2] = split_node_numbers[22]; //2-7
					child_nodes[3] = split_node_numbers[14]; //3-7
					child_nodes[4] = split_node_numbers[25]; //4-7
					child_nodes[5] = split_node_numbers[17]; //5-7
					child_nodes[6] = split_node_numbers[18]; //6-7
					child_nodes[7] = split_node_numbers[ 7]; //7
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}
		}

		void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const override {
			face_nodes.resize(4);
			switch (face_number) {
			case (0): //left
				face_nodes[0] = this->ELEM.nodes[0];
				face_nodes[1] = this->ELEM.nodes[4];
				face_nodes[2] = this->ELEM.nodes[2];
				face_nodes[3] = this->ELEM.nodes[6];
				break;
			case (1): //right
				face_nodes[0] = this->ELEM.nodes[1];
				face_nodes[1] = this->ELEM.nodes[3];
				face_nodes[2] = this->ELEM.nodes[5];
				face_nodes[3] = this->ELEM.nodes[7];
				break;
			case (2): //top
				face_nodes[0] = this->ELEM.nodes[2];
				face_nodes[1] = this->ELEM.nodes[6];
				face_nodes[2] = this->ELEM.nodes[3];
				face_nodes[3] = this->ELEM.nodes[7];
				break;
			case (3): //bottom
				face_nodes[0] = this->ELEM.nodes[0];
				face_nodes[1] = this->ELEM.nodes[1];
				face_nodes[2] = this->ELEM.nodes[4];
				face_nodes[3] = this->ELEM.nodes[5];
				break;
			case (4): //back
				face_nodes[0] = this->ELEM.nodes[1];
				face_nodes[1] = this->ELEM.nodes[0];
				face_nodes[2] = this->ELEM.nodes[3];
				face_nodes[3] = this->ELEM.nodes[2];
				break;
			case (5): //front
				face_nodes[0] = this->ELEM.nodes[4];
				face_nodes[1] = this->ELEM.nodes[5];
				face_nodes[2] = this->ELEM.nodes[6];
				face_nodes[3] = this->ELEM.nodes[7];
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
				case (0): // Left face [0, 4, 2, 6]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 4]; //4
				split_face_nodes[2] = split_node_numbers[ 2]; //2
				split_face_nodes[3] = split_node_numbers[ 6]; //6
				split_face_nodes[4] = split_node_numbers[12]; //0-4
				split_face_nodes[5] = split_node_numbers[19]; //4-6
				split_face_nodes[6] = split_node_numbers[13]; //2-6
				split_face_nodes[7] = split_node_numbers[11]; //0-2
				split_face_nodes[8] = split_node_numbers[20]; //left face center
				break;

			case (1): // Right face [1, 3, 5, 7]
				split_face_nodes[0] = split_node_numbers[ 1]; //1
				split_face_nodes[1] = split_node_numbers[ 3]; //3
				split_face_nodes[2] = split_node_numbers[ 5]; //5
				split_face_nodes[3] = split_node_numbers[ 7]; //7
				split_face_nodes[4] = split_node_numbers[ 9]; //1-3
				split_face_nodes[5] = split_node_numbers[14]; //3-7
				split_face_nodes[6] = split_node_numbers[17]; //5-7
				split_face_nodes[7] = split_node_numbers[15]; //1-5
				split_face_nodes[8] = split_node_numbers[21]; //right face center
				break;

			case (2): // Top face [2, 6, 3, 7]
				split_face_nodes[0] = split_node_numbers[ 2]; //2
				split_face_nodes[1] = split_node_numbers[ 6]; //6
				split_face_nodes[2] = split_node_numbers[ 3]; //3
				split_face_nodes[3] = split_node_numbers[ 7]; //7
				split_face_nodes[4] = split_node_numbers[13]; //2-6
				split_face_nodes[5] = split_node_numbers[18]; //6-7
				split_face_nodes[6] = split_node_numbers[14]; //3-7
				split_face_nodes[7] = split_node_numbers[10]; //2-3
				split_face_nodes[8] = split_node_numbers[22]; //top face center
				break;

			case (3): // Bottom face [0, 1, 4, 5]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 1]; //1
				split_face_nodes[2] = split_node_numbers[ 4]; //4
				split_face_nodes[3] = split_node_numbers[ 5]; //5
				split_face_nodes[4] = split_node_numbers[ 8]; //0-1
				split_face_nodes[5] = split_node_numbers[15]; //1-5
				split_face_nodes[6] = split_node_numbers[16]; //4-5
				split_face_nodes[7] = split_node_numbers[12]; //0-4
				split_face_nodes[8] = split_node_numbers[23]; //bottom face center
				break;

			case (4): // Back face [1, 0, 3, 2]
				split_face_nodes[0] = split_node_numbers[ 1]; //1
				split_face_nodes[1] = split_node_numbers[ 0]; //0
				split_face_nodes[2] = split_node_numbers[ 3]; //3
				split_face_nodes[3] = split_node_numbers[ 2]; //2
				split_face_nodes[4] = split_node_numbers[ 8]; //1-0
				split_face_nodes[5] = split_node_numbers[11]; //0-2
				split_face_nodes[6] = split_node_numbers[10]; //2-3
				split_face_nodes[7] = split_node_numbers[ 9]; //3-1
				split_face_nodes[8] = split_node_numbers[24]; //back face center
				break;

			case (5): // Front face [4, 5, 6, 7]
				split_face_nodes[0] = split_node_numbers[ 4]; //4
				split_face_nodes[1] = split_node_numbers[ 5]; //5
				split_face_nodes[2] = split_node_numbers[ 6]; //6
				split_face_nodes[3] = split_node_numbers[ 7]; //7
				split_face_nodes[4] = split_node_numbers[16]; //4-5
				split_face_nodes[5] = split_node_numbers[17]; //5-7
				split_face_nodes[6] = split_node_numbers[18]; //6-7
				split_face_nodes[7] = split_node_numbers[19]; //4-6
				split_face_nodes[8] = split_node_numbers[25]; //front face center
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}
	};
}