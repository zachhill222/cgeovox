#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
#include <array>
#include <algorithm>

#include <cassert>
#include <iostream>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// Container for tracking element information.
	/////////////////////////////////////////////////
	struct Element {
		std::vector<size_t> nodes;
		size_t nNodes;
		std::vector<size_t> children;
		size_t parent = (size_t) -1;
		size_t color  = (size_t) -1;
		bool is_active = true;
		int vtkID;

		///initialize by vtkID
		Element(const int vtkID) : vtkID(vtkID) {
			switch (vtkID) {
				//linear
				case 3:  nNodes=2;  break; //line
				case 5:  nNodes=3;  break; //triangle
				case 8:  nNodes=4;  break; //pixel
				case 9:  nNodes=4;  break; //quad
				case 10: nNodes=4;  break; //tetra
				case 11: nNodes=8;  break; //voxel
				case 12: nNodes=8;  break; //hexahedron

				//quadratic
				case 21: nNodes=3;  break; //quadratic line
				case 22: nNodes=6;  break; //quadratic triangle
				case 24: nNodes=10; break; //quadratic tetra
				case 28: nNodes=9;  break; //bi-quadratic quad
				case 29: nNodes=27; break; //tri-quadratic hexahedron

				default: throw std::invalid_argument("unkown element type.");
			}

			nodes.resize(nNodes);
		}

		///initialize by vtkID and array of nodes:
		Element(const std::vector<size_t> &nodes, const int vtkID) : Element(vtkID) {assert(nodes.size()== nNodes); this->nodes=nodes;}
	};


	/// Check if two elements are the same (up to orientation)
	// template <int A, int B>
	// bool operator==(const Element<A> &A, const Element<B> &B) {
	// 	if (A.vtkID!=B.vtkID) {return false;}

	// 	std::array<size_t, A> a = A.nodes;
	// 	std::array<size_t, B> b = B.nodes;
	// 	std::sort(a.begin(), a.end());
	// 	std::sort(b.begin(), b.end());

	// 	assert(A.nNodes==B.nNodes);
	// 	for (int i=0; i<nNodes; i++) {
	// 		if (a[i]!=b[i]) {return false;}
	// 	}
	// 	return true;
	// }


	/////////////////////////////////////////////////
	/// Interface for VTK element types
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_ELEMENT {
	public:
		VTK_ELEMENT(const Element &elem) : ELEM(elem) {}
		virtual ~VTK_ELEMENT() {}
		const Element &ELEM;
		virtual void split(std::vector<Point_t> &vertices) const = 0;
		virtual void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const = 0;
		virtual void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const = 0;
		virtual int nChildrenWhenSplit() const = 0;
		virtual int nVerticesWhenSplit() const = 0;
	};


	/////////////////////////////////////////////////
	/// Line element
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_LINE : public VTK_ELEMENT<Point_t>{
	public:
		VTK_LINE(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==N_NODES);}
		static constexpr int VTK_ID = 3;
		static constexpr int N_NODES = 2;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 2;
		static constexpr int N_VERTICES_WHEN_SPLIT = 3;

		int nChildrenWhenSplit() const override {return N_CHILDREN_WHEN_SPLIT;}
		int nVerticesWhenSplit() const override {return N_VERTICES_WHEN_SPLIT;}

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==N_NODES);
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //center
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

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
	};



	/////////////////////////////////////////////////
	/// Pixel element
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_PIXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_PIXEL(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==N_NODES);}
		static constexpr int VTK_ID = 8;
		static constexpr int N_NODES = 4;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 4;
		static constexpr int N_VERTICES_WHEN_SPLIT = 9;

		int nChildrenWhenSplit() const override {return N_CHILDREN_WHEN_SPLIT;}
		int nVerticesWhenSplit() const override {return N_VERTICES_WHEN_SPLIT;}

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==N_NODES);
			vertices.reserve(N_VERTICES_WHEN_SPLIT);

			//edge midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //bottom
			vertices.emplace_back(0.5*(vertices[1]+vertices[3])); //right
			vertices.emplace_back(0.5*(vertices[2]+vertices[3])); //top
			vertices.emplace_back(0.5*(vertices[0]+vertices[2])); //left

			//center
			vertices.emplace_back(0.5*(vertices[0]+vertices[3]));
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

			switch (child_number) {
				case (0):
					child_nodes[0] = split_node_numbers[0];
					child_nodes[1] = split_node_numbers[4];
					child_nodes[2] = split_node_numbers[7];
					child_nodes[3] = split_node_numbers[8];
					break;
				case (1):
					child_nodes[0] = split_node_numbers[4];
					child_nodes[1] = split_node_numbers[1];
					child_nodes[2] = split_node_numbers[8];
					child_nodes[3] = split_node_numbers[5];
					break;
				case (2):
					child_nodes[0] = split_node_numbers[7];
					child_nodes[1] = split_node_numbers[8];
					child_nodes[2] = split_node_numbers[2];
					child_nodes[3] = split_node_numbers[6];
					break;
				case (3):
					child_nodes[0] = split_node_numbers[8];
					child_nodes[1] = split_node_numbers[5];
					child_nodes[2] = split_node_numbers[6];
					child_nodes[3] = split_node_numbers[3];
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
	};


	/////////////////////////////////////////////////
	/// Voxel element
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_VOXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_VOXEL(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==N_NODES);}
		static constexpr int VTK_ID = 11;
		static constexpr int N_NODES = 8;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 8;
		static constexpr int N_VERTICES_WHEN_SPLIT = 27;

		int nChildrenWhenSplit() const override {return N_CHILDREN_WHEN_SPLIT;}
		int nVerticesWhenSplit() const override {return N_VERTICES_WHEN_SPLIT;}

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==N_NODES);
			vertices.reserve(N_VERTICES_WHEN_SPLIT);

			//edge midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //back face
			vertices.emplace_back(0.5*(vertices[1]+vertices[3])); //back face
			vertices.emplace_back(0.5*(vertices[2]+vertices[3])); //back face
			vertices.emplace_back(0.5*(vertices[0]+vertices[2])); //back face

			vertices.emplace_back(0.5*(vertices[0]+vertices[4])); //connecting edge
			vertices.emplace_back(0.5*(vertices[2]+vertices[6])); //connecting edge
			vertices.emplace_back(0.5*(vertices[3]+vertices[7])); //connecting edge
			vertices.emplace_back(0.5*(vertices[1]+vertices[5])); //connecting edge
			
			vertices.emplace_back(0.5*(vertices[4]+vertices[5])); //front face
			vertices.emplace_back(0.5*(vertices[5]+vertices[7])); //front face
			vertices.emplace_back(0.5*(vertices[6]+vertices[7])); //front face
			vertices.emplace_back(0.5*(vertices[4]+vertices[6])); //front face

			//face midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[6])); //left face
			vertices.emplace_back(0.5*(vertices[1]+vertices[7])); //right face
			vertices.emplace_back(0.5*(vertices[2]+vertices[7])); //top face
			vertices.emplace_back(0.5*(vertices[0]+vertices[5])); //bottom face
			vertices.emplace_back(0.5*(vertices[0]+vertices[3])); //back face
			vertices.emplace_back(0.5*(vertices[4]+vertices[7])); //front face

			//center
			vertices.emplace_back(0.5*(vertices[0]+vertices[7]));
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

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
	};



	/////////////////////////////////////////////////
	/// Factory to create the appropriate vtk_element type
	/// Note that whenever this function is called, the result must be deleted.
	/// For example auto* VE = VTK_ELEMENT_FACTORY(ELEM); delete VE;
	/////////////////////////////////////////////////
	template <typename Point_t> 
	VTK_ELEMENT<Point_t>* _VTK_ELEMENT_FACTORY(const Element &ELEM) {
		switch (ELEM.vtkID) {
			case 3:  return new VTK_LINE<Point_t>(ELEM);
			case 8:  return new VTK_PIXEL<Point_t>(ELEM);
			case 11: return new VTK_VOXEL<Point_t>(ELEM);
			default: throw std::invalid_argument("unknown element type.");
		}
	};
	


}