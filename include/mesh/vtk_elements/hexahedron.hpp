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
	/* Hexahedron element node labels
	
	 			3 ------- 2
				|\	      |\
				| \		  | \
				0 -\----- 1  \
				 \	\      \  \
				  \	 7 ------- 6
				   \ |		 \ |
					\|        \|
					 4 ------- 5
	*/

	/////////////////////////////////////////////////
	/// Hexahedral element
	/// Note that this is an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// but the center of the element must be found by evaluating the shape functions.
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
	template <typename Vertex_t>
	class VTK_HEXAHEDRON : public VTK_ELEMENT<Vertex_t> {
	public:
		VTK_HEXAHEDRON(const BasicElement &elem) : VTK_ELEMENT<Vertex_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID  = HEXAHEDRON_VTK_ID;
		static constexpr int REF_DIM = 3; //dimension of the reference element
		using Scalar_t    = typename Vertex_t::Scalar_t;
		using RefPoint_t  = gv::util::Point<REF_DIM, Scalar_t>; //type of point in the reference element
		using Matrix_t    = gv::util::Matrix<3,REF_DIM,Scalar_t>; //dimensions of the jacobian matrix (output space is always R3)

		using ScalarFun_t = std::function<Scalar_t(RefPoint_t)>; //function type to evaluate a basis in the element
		using VectorFun_t = std::function<Vertex_t(RefPoint_t)>; //function type to evaluate the gradient of a basis function
		using MatrixFun_t = std::function<Matrix_t(RefPoint_t)>; //function type to evaluate the jacobian of the isoparametric mapping

		void split(std::vector<Vertex_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

			using T = typename Vertex_t::Scalar_t;
			//edge midpoints
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]})); //8  - back face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[2]})); //9  - back face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3]})); //10 - back face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3]})); //11 - back face

			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[4]})); //12 - connecting edge
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[3],vertices[7]})); //13 - connecting edge
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[6]})); //14 - connecting edge
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[5]})); //15 - connecting edge
			
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[5]})); //16 - front face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[5],vertices[6]})); //17 - front face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[6],vertices[7]})); //18 - front face
			vertices.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[7]})); //19 - front face

			//face midpoints
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3],vertices[4],vertices[7]})); //20 - left face  (L)
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[2],vertices[5],vertices[6]})); //21 - right face (R)
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3],vertices[6],vertices[7]})); //22 - up face    (U)
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[4],vertices[5]})); //23 - down face  (D)
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[2],vertices[3]})); //24 - back face  (B)
			vertices.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[5],vertices[6],vertices[7]})); //25 - front face (F)

			//center
			vertices.emplace_back(T{0.125}*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[2],vertices[3]
										,vertices[4],vertices[5],vertices[6],vertices[7]})); //26
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

			switch (child_number) {
				case (0): //hex element containing original vertex 0
					child_nodes[0] = split_node_numbers[ 0]; //0
					child_nodes[1] = split_node_numbers[ 8]; //0-1
					child_nodes[2] = split_node_numbers[24]; //B
					child_nodes[3] = split_node_numbers[11]; //0-3
					child_nodes[4] = split_node_numbers[12]; //0-4
					child_nodes[5] = split_node_numbers[23]; //D
					child_nodes[6] = split_node_numbers[26]; //C
					child_nodes[7] = split_node_numbers[20]; //L
					break;

				case (1): //hex element containing original vertex 1
					child_nodes[0] = split_node_numbers[ 8]; //0-1
					child_nodes[1] = split_node_numbers[ 1]; //1
					child_nodes[2] = split_node_numbers[ 9]; //1-2
					child_nodes[3] = split_node_numbers[24]; //B
					child_nodes[4] = split_node_numbers[23]; //D
					child_nodes[5] = split_node_numbers[15]; //1-5
					child_nodes[6] = split_node_numbers[21]; //R
					child_nodes[7] = split_node_numbers[26]; //C
					break;

				case (2): //hex element containing original vertex 2
					child_nodes[0] = split_node_numbers[24]; //B
					child_nodes[1] = split_node_numbers[ 9]; //1-2
					child_nodes[2] = split_node_numbers[ 2]; //2
					child_nodes[3] = split_node_numbers[10]; //2-3
					child_nodes[4] = split_node_numbers[26]; //C
					child_nodes[5] = split_node_numbers[21]; //R
					child_nodes[6] = split_node_numbers[14]; //2-6
					child_nodes[7] = split_node_numbers[22]; //U
					break;

				case (3): //hex element containing original vertex 3
					child_nodes[0] = split_node_numbers[11]; //0-3
					child_nodes[1] = split_node_numbers[24]; //B
					child_nodes[2] = split_node_numbers[10]; //2-3
					child_nodes[3] = split_node_numbers[ 3]; //3
					child_nodes[4] = split_node_numbers[20]; //L
					child_nodes[5] = split_node_numbers[26]; //C
					child_nodes[6] = split_node_numbers[22]; //U
					child_nodes[7] = split_node_numbers[13]; //3-7
					break;

				case (4): //hex element containing original vertex 4
					child_nodes[0] = split_node_numbers[12]; //0-4
					child_nodes[1] = split_node_numbers[23]; //D
					child_nodes[2] = split_node_numbers[26]; //C
					child_nodes[3] = split_node_numbers[20]; //L
					child_nodes[4] = split_node_numbers[ 4]; //4
					child_nodes[5] = split_node_numbers[16]; //4-5
					child_nodes[6] = split_node_numbers[25]; //F
					child_nodes[7] = split_node_numbers[19]; //4-7
					break;

				case (5): //hex element containing original vertex 5
					child_nodes[0] = split_node_numbers[23]; //D
					child_nodes[1] = split_node_numbers[15]; //1-5
					child_nodes[2] = split_node_numbers[21]; //R
					child_nodes[3] = split_node_numbers[26]; //C
					child_nodes[4] = split_node_numbers[16]; //4-5
					child_nodes[5] = split_node_numbers[ 5]; //5
					child_nodes[6] = split_node_numbers[17]; //5-6
					child_nodes[7] = split_node_numbers[25]; //F
					break;

				case (6): //hex element containing original vertex 6
					child_nodes[0] = split_node_numbers[26]; //C
					child_nodes[1] = split_node_numbers[21]; //R
					child_nodes[2] = split_node_numbers[14]; //2-6
					child_nodes[3] = split_node_numbers[22]; //U
					child_nodes[4] = split_node_numbers[25]; //F
					child_nodes[5] = split_node_numbers[17]; //5-6
					child_nodes[6] = split_node_numbers[ 6]; //6
					child_nodes[7] = split_node_numbers[18]; //6-7
					break;

				case (7): //hex element containing original vertex 7
					child_nodes[0] = split_node_numbers[20]; //L
					child_nodes[1] = split_node_numbers[26]; //C
					child_nodes[2] = split_node_numbers[22]; //U
					child_nodes[3] = split_node_numbers[13]; //3-7
					child_nodes[4] = split_node_numbers[19]; //4-7
					child_nodes[5] = split_node_numbers[25]; //F
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
				face_nodes[2] = this->ELEM.nodes[7];
				face_nodes[3] = this->ELEM.nodes[3];
				break;
			case (1): //right
				face_nodes[0] = this->ELEM.nodes[1];
				face_nodes[1] = this->ELEM.nodes[2];
				face_nodes[2] = this->ELEM.nodes[6];
				face_nodes[3] = this->ELEM.nodes[5];
				break;
			case (2): //top
				face_nodes[0] = this->ELEM.nodes[2];
				face_nodes[1] = this->ELEM.nodes[3];
				face_nodes[2] = this->ELEM.nodes[7];
				face_nodes[3] = this->ELEM.nodes[6];
				break;
			case (3): //bottom
				face_nodes[0] = this->ELEM.nodes[0];
				face_nodes[1] = this->ELEM.nodes[1];
				face_nodes[2] = this->ELEM.nodes[5];
				face_nodes[3] = this->ELEM.nodes[4];
				break;
			case (4): //back
				face_nodes[0] = this->ELEM.nodes[0];
				face_nodes[1] = this->ELEM.nodes[3];
				face_nodes[2] = this->ELEM.nodes[2];
				face_nodes[3] = this->ELEM.nodes[1];
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
			case (0) : //Left face [0, 4, 7, 3]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 4]; //4
				split_face_nodes[2] = split_node_numbers[ 7]; //7
				split_face_nodes[3] = split_node_numbers[ 3]; //3
				split_face_nodes[4] = split_node_numbers[12]; //0-4
				split_face_nodes[5] = split_node_numbers[19]; //4-7
				split_face_nodes[6] = split_node_numbers[13]; //7-3
				split_face_nodes[7] = split_node_numbers[11]; //3-0
				split_face_nodes[8] = split_node_numbers[20]; //L
				break;
			case (1): // Right face [1,2,6,5]
				split_face_nodes[0] = split_node_numbers[ 1]; //1
				split_face_nodes[1] = split_node_numbers[ 2]; //2
				split_face_nodes[2] = split_node_numbers[ 6]; //6
				split_face_nodes[3] = split_node_numbers[ 5]; //5
				split_face_nodes[4] = split_node_numbers[ 9]; //1-2
				split_face_nodes[5] = split_node_numbers[14]; //2-6
				split_face_nodes[6] = split_node_numbers[17]; //6-5
				split_face_nodes[7] = split_node_numbers[15]; //5-1
				split_face_nodes[8] = split_node_numbers[21]; //R
				break;
		        
		    case (2): // Top face [2,3,7,6]
				split_face_nodes[0] = split_node_numbers[ 2]; //2
				split_face_nodes[1] = split_node_numbers[ 3]; //3
				split_face_nodes[2] = split_node_numbers[ 7]; //7
				split_face_nodes[3] = split_node_numbers[ 6]; //6
				split_face_nodes[4] = split_node_numbers[10]; //2-3
				split_face_nodes[5] = split_node_numbers[13]; //3-7
				split_face_nodes[6] = split_node_numbers[18]; //7-6
				split_face_nodes[7] = split_node_numbers[14]; //6-2
				split_face_nodes[8] = split_node_numbers[22]; //U
				break;
		        
		    case (3): // Bottom face [0,1,5,4]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 1]; //1
				split_face_nodes[2] = split_node_numbers[ 5]; //5
				split_face_nodes[3] = split_node_numbers[ 4]; //4
				split_face_nodes[4] = split_node_numbers[ 8]; //0-1
				split_face_nodes[5] = split_node_numbers[15]; //1-5
				split_face_nodes[6] = split_node_numbers[16]; //5-4
				split_face_nodes[7] = split_node_numbers[12]; //4-0
				split_face_nodes[8] = split_node_numbers[23]; //D
				break;
		        
		    case (4): // Back face [0,3,2,1]
				split_face_nodes[0] = split_node_numbers[ 0]; //0
				split_face_nodes[1] = split_node_numbers[ 3]; //3
				split_face_nodes[2] = split_node_numbers[ 2]; //2
				split_face_nodes[3] = split_node_numbers[ 1]; //1
				split_face_nodes[4] = split_node_numbers[11]; //0-3
				split_face_nodes[5] = split_node_numbers[10]; //3-2
				split_face_nodes[6] = split_node_numbers[ 9]; //2-1
				split_face_nodes[7] = split_node_numbers[ 8]; //1-0
				split_face_nodes[8] = split_node_numbers[24]; //B
				break;
		        
		    case (5): // Front face [4,5,6,7]
				split_face_nodes[0] = split_node_numbers[ 4]; //4
				split_face_nodes[1] = split_node_numbers[ 5]; //5
				split_face_nodes[2] = split_node_numbers[ 6]; //6
				split_face_nodes[3] = split_node_numbers[ 7]; //7
				split_face_nodes[4] = split_node_numbers[16]; //4-5
				split_face_nodes[5] = split_node_numbers[17]; //5-6
				split_face_nodes[6] = split_node_numbers[18]; //6-7
				split_face_nodes[7] = split_node_numbers[19]; //7-4
				split_face_nodes[8] = split_node_numbers[25]; //F
				break;
		        
		    default:
		        throw std::out_of_range("face number out of bounds");
		        break;
		    }
		}
		
		bool isInterior(const std::vector<Vertex_t>& vertices, const Vertex_t& coord) const override {
			assert(false);
			return false;
		}
	};
}