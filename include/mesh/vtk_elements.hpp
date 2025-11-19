#pragma once

#include "util/point.hpp"
#include "util/box.hpp"


#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <map>
#include <algorithm>

#include <cassert>
#include <iostream>




namespace gv::mesh
{
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

			//round to lower precision
			for (Point_t &v : vertices) {
				for (int i = 0; i < 3; i++) {
					v[i] = static_cast<double>(static_cast<float>(v[i]));
				}
			}

			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //center
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

		void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const override {}
	};



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
	/// For example, the basis function located at the reference vertex (-1,+1) is 0.5(1-e0)*0.5(1+e1) where e0 and e1 are the cartesian coordinates
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

			//round to lower precision
			for (Point_t &v : vertices) {
				for (int i = 0; i < 3; i++) {
					v[i] = static_cast<double>(static_cast<float>(v[i]));
				}
			}

			//edge midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //4 - bottom
			vertices.emplace_back(0.5*(vertices[1]+vertices[3])); //5 - right
			vertices.emplace_back(0.5*(vertices[2]+vertices[3])); //6 - top
			vertices.emplace_back(0.5*(vertices[0]+vertices[2])); //7 - left

			//center
			vertices.emplace_back(0.5*(vertices[0]+vertices[3])); //8
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

		void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const override {}
	};


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

			//round to lower precision
			for (Point_t &v : vertices) {
				for (int i = 0; i < 3; i++) {
					v[i] = static_cast<double>(static_cast<float>(v[i]));
				}
			}

			//edge midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //4 - bottom (B)
			vertices.emplace_back(0.5*(vertices[1]+vertices[2])); //5 - right (R)
			vertices.emplace_back(0.5*(vertices[2]+vertices[3])); //6 - top (T)
			vertices.emplace_back(0.5*(vertices[0]+vertices[3])); //7 - left (L)

			//center
			vertices.emplace_back(0.25*(vertices[0]+vertices[1]+vertices[2]+vertices[3])); //8 (C)
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

		void getSplitFaceNodes(std::vector<size_t> &split_face_nodes, const int face_number, const std::vector<size_t> &split_node_numbers) const override {}
	};


	/////////////////////////////////////////////////
	/// Voxel element
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
	template <typename Point_t>
	class VTK_HEXAHEDRON : public VTK_ELEMENT<Point_t> {
	public:
		VTK_HEXAHEDRON(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nodes.size()==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = HEXAHEDRON_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

			using T = typename Point_t::Scalar_t;
			//edge midpoints
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1]})); //8  - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[2]})); //9  - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3]})); //10 - back face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3]})); //11 - back face

			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[4]})); //12 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[3],vertices[7]})); //13 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[6]})); //14 - connecting edge
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[5]})); //15 - connecting edge
			
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[5]})); //16 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[5],vertices[6]})); //17 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[6],vertices[7]})); //18 - front face
			vertices.emplace_back(0.5*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[7]})); //19 - front face

			//face midpoints
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[3],vertices[4],vertices[7]})); //20 - left face  (L)
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[1],vertices[2],vertices[5],vertices[6]})); //21 - right face (R)
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[2],vertices[3],vertices[6],vertices[7]})); //22 - up face    (U)
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[4],vertices[5]})); //23 - down face  (D)
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[2],vertices[3]})); //24 - back face  (B)
			vertices.emplace_back(0.25*gv::util::sorted_sum<3,T,T,T>({vertices[4],vertices[5],vertices[6],vertices[7]})); //25 - front face (F)

			//center
			vertices.emplace_back(0.125*gv::util::sorted_sum<3,T,T,T>({vertices[0],vertices[1],vertices[2],vertices[3]
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
	};



	/////////////////////////////////////////////////
	/// Factory to create the appropriate vtk_element type
	/// Note that whenever this function is called, the result must be deleted.
	/// For example auto* VE = VTK_ELEMENT_FACTORY(ELEM); delete VE;
	/////////////////////////////////////////////////
	template <typename Point_t> 
	VTK_ELEMENT<Point_t>* _VTK_ELEMENT_FACTORY(const BasicElement &ELEM) {
		switch (ELEM.vtkID) {
			case LINE_VTK_ID:       return new VTK_LINE<Point_t>(ELEM);
			case PIXEL_VTK_ID:      return new VTK_PIXEL<Point_t>(ELEM);
			case QUAD_VTK_ID:       return new VTK_QUAD<Point_t>(ELEM);
			case VOXEL_VTK_ID:      return new VTK_VOXEL<Point_t>(ELEM);
			case HEXAHEDRON_VTK_ID: return new VTK_HEXAHEDRON<Point_t>(ELEM);
			default: throw std::invalid_argument("unknown element type.");
		}
	};


	/////////////////////////////////////////////////
	/// Function to change an element into its isoparametric variant (e.g. pixel to quad or voxel to hexahedron).
	/////////////////////////////////////////////////
	void makeIsoparametric(BasicElement &ELEM) {
		switch (ELEM.vtkID) {
		case PIXEL_VTK_ID:
			std::swap(ELEM.nodes[2],ELEM.nodes[3]);
			ELEM.vtkID = QUAD_VTK_ID;
			return;
		case VOXEL_VTK_ID:
			std::swap(ELEM.nodes[2],ELEM.nodes[3]);
			std::swap(ELEM.nodes[6],ELEM.nodes[7]);
			ELEM.vtkID = HEXAHEDRON_VTK_ID;
			return;
		}
	}
}