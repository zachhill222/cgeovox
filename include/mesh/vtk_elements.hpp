#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "concepts.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <map>
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
		size_t index  = (size_t) -1;
		bool is_active = true;
		int vtkID;


		///default initializer
		Element() {}

		///initialize by vtkID
		Element(const int vtkID) : nodes(vtk_n_nodes(vtkID)), nNodes(vtk_n_nodes(vtkID)), vtkID(vtkID) {}

		///initialize by vtkID and array of nodes:
		Element(const std::vector<size_t> &nodes, const int vtkID) : nodes(nodes), nNodes(vtk_n_nodes(vtkID)), vtkID(vtkID) {assert(nodes.size()==nNodes);}
	};


	/// Check if two elements are the same (up to orientation)
	bool operator==(const Element &A, const Element &B) {
		if (A.vtkID!=B.vtkID) {return false;}
		if (A.nNodes!=B.nNodes) {return false;}

		std::vector<size_t> a = A.nodes;
		std::vector<size_t> b = B.nodes;
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());

		return a == b;
	}


	/// Element hashing function for use in unordered_set (for example). The order of the element nodes is irrelevent to the hash value.
	struct ElemHashBitPack {
		size_t operator()(const Element& ELEM) const {
			//sort the nodes
			std::vector<size_t> nodes = ELEM.nodes;
			std::sort(nodes.begin(), nodes.end());

			//initialize the hash by getting the last few bits from each node index
			size_t hash = 0;
			size_t bits_per_node;
			if constexpr (sizeof(size_t)==4) {bits_per_node=32/nodes.size();} //32-bit
			else if constexpr (sizeof(size_t)==8) {bits_per_node=64/nodes.size();} //64-bit
			else {bits_per_node=1;}

			size_t mask = (((size_t) 1) << bits_per_node) - 1; //exactly the last bits_per_node bits are 1

			for (size_t i=0; i<nodes.size(); i++) {
				size_t node_bits = nodes[i] & mask;
				hash |= (node_bits << (i*bits_per_node));
			}

			//scramble the hash (MurmurHash3)
			if constexpr (sizeof(size_t)==4) {
				hash ^= hash >> 16;
				hash *= 0x85ebca6b;
				hash ^= hash >> 16;
				hash *= 0xc2b2ae35;
				hash ^= hash >> 16;
			} else if constexpr (sizeof(size_t)==8) {
				hash ^= hash >> 33;
				hash *= 0xff51afd7ed558ccdULL;
				hash ^= hash >> 33;
				hash *= 0xc4ceb9fe1a85ec53ULL;
				hash ^= hash >> 33;
			}

			return hash;
		}
	};


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
		Element getFace(const int face_number) const {
			Element face(vtk_face_id(this->ELEM.vtkID));
			face.color  = this->ELEM.color;
			face.parent = this->ELEM.index;
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
		VTK_LINE(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = LINE_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
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
	};



	/////////////////////////////////////////////////
	/// Pixel element
	/////////////////////////////////////////////////
	template <typename Point_t>
	class VTK_PIXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_PIXEL(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = PIXEL_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

			//edge midpoints
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //bottom
			vertices.emplace_back(0.5*(vertices[1]+vertices[3])); //right
			vertices.emplace_back(0.5*(vertices[2]+vertices[3])); //top
			vertices.emplace_back(0.5*(vertices[0]+vertices[2])); //left

			//center
			vertices.emplace_back(0.5*(vertices[0]+vertices[3]));
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_nodes_when_split(VTK_ID));
			child_nodes.resize(vtk_n_nodes(VTK_ID));

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
		VTK_VOXEL(const Element &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.nNodes==vtk_n_nodes(elem.vtkID));}
		static constexpr int VTK_ID = VOXEL_VTK_ID;

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==vtk_n_nodes(VTK_ID));
			vertices.reserve(vtk_n_nodes_when_split(VTK_ID));

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