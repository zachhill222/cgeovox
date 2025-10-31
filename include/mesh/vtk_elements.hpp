#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
#include <algorithm>

#include <cassert>
#include <iostream>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// The base class for all mesh elements. All points are assumed to be in 3D, no matter the element type, but various levels of precision are allowed.
	/////////////////////////////////////////////////
	template <Float T>
	class BaseElement {
		public:
		using Point_t = gv::util::Point<3,T>;

		/// Constructor when the number of nodes is known
		BaseElement(const int nNodes) : nodes(nNodes), children(0) {}

		/// Constructor when the nodes are known
		BaseElement(const std::vector<size_t> &nodes) : nodes(nodes), children(0) {}

		/// The nodes that define the element
		std::vector<size_t> nodes;

		/// The color of this element in the mesh
		size_t color = (size_t) -1;

		/// Which element is the parent of this element
		size_t parent = (size_t) -1;

		/// Which elements are children of this element
		std::vector<size_t> children;

		/// Track if the element is active
		bool is_active = true;

		/////////////////////////////////////////////////
		/// Method to generate coordinates for refinement
		///
		/// @param vertices The coordinates of the first nNodes vertices. The new vertices will be appended to the end.
		/////////////////////////////////////////////////
		virtual void split(std::vector<Point_t> &vertices) const = 0;

		/////////////////////////////////////////////////
		/// Method construct a refined/child element node indices. Child elements are of the same element type as the parent.
		///
		/// @param child_nodes Where the node indices for the new element will be stored
		/// @param child_number Which child element is being created
		/// @param global_node_numbers The global node numbers corresponding to the vertices generated after split(vertices) is called.
		/////////////////////////////////////////////////
		virtual void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &global_node_numbers) const = 0;

		/// Get the nodes that generate the specified face
		virtual void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const = 0;

		/// common values that are needed
		virtual int vtkID() const = 0;
		virtual int nNodes() const = 0;
		virtual int nChildrenWhenSplit() const = 0;
		virtual int nVerticesWhenSplit() const = 0;
	};

	/////////////////////////////////////////////////
	/// Intermediate class to 
	/////////////////////////////////////////////////

	/// Check if two elements are the same (up to orientation)
	template <Float T>
	bool operator==(const BaseElement<T> &A, const BaseElement<T> &B) {
		if (A.nodes.size()!=B.nodes.size()) {return false;}

		std::vector<size_t> a = A.nodes;
		std::vector<size_t> b = B.nodes;

		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());

		return a==b;
	}



	/////////////////////////////////////////////////
	/// Line element
	/////////////////////////////////////////////////
	template <Float T>
	class VTK_LINE : public BaseElement<T> {
	public:
		using Point_t = BaseElement<T>::Point_t;
		VTK_LINE() : BaseElement<T>(2) {}
		VTK_LINE(const std::vector<size_t> &nodes) : BaseElement<T>(nodes) {assert(nodes.size()==2);}
		
		static constexpr int VTK_ID = 3;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 2;
		static constexpr int N_VERTICES_WHEN_SPLIT = 3;
		static constexpr int N_NODES = 2;

		int vtkID() const override {return VTK_ID;}
		int nNodes() const override {return N_NODES;}
		int nChildrenWhenSplit() const override {return N_CHILDREN_WHEN_SPLIT;}
		int nVerticesWhenSplit() const override {return N_VERTICES_WHEN_SPLIT;}

		void split(std::vector<Point_t> &vertices) const override {
			assert(vertices.size()==N_NODES);
			vertices.emplace_back(0.5*(vertices[0]+vertices[1])); //center
		}

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &global_node_numbers) const override {
			assert(global_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

			switch (child_number) {
				case (0):
					child_nodes[0] = global_node_numbers[0];
					child_nodes[1] = global_node_numbers[2];
					break;
				case (1):
					child_nodes[0] = global_node_numbers[2];
					child_nodes[1] = global_node_numbers[1];
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
				}
		}

		void getFaceNodes(std::vector<size_t> &face_nodes, const int face_number) const override {
			face_nodes.resize(1);
			face_nodes[0] = this->nodes[face_number];
		}
	};



	/////////////////////////////////////////////////
	/// Pixel element
	/////////////////////////////////////////////////
	template <Float T>
	class VTK_PIXEL : public BaseElement<T> {
	public:
		using Point_t = BaseElement<T>::Point_t;
		VTK_PIXEL() : BaseElement<T>(4) {};
		VTK_PIXEL(const std::vector<size_t> &nodes) : BaseElement<T>(nodes) {assert(nodes.size()==4);}
		static constexpr int VTK_ID = 8;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 4;
		static constexpr int N_VERTICES_WHEN_SPLIT = 9;
		static constexpr int N_NODES = 4;

		int vtkID() const override {return VTK_ID;}
		int nNodes() const override {return N_NODES;}
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

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &global_node_numbers) const override {
			assert(global_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

			switch (child_number) {
				case (0):
					child_nodes[0] = global_node_numbers[0];
					child_nodes[1] = global_node_numbers[4];
					child_nodes[2] = global_node_numbers[7];
					child_nodes[3] = global_node_numbers[8];
					break;
				case (1):
					child_nodes[0] = global_node_numbers[4];
					child_nodes[1] = global_node_numbers[1];
					child_nodes[2] = global_node_numbers[8];
					child_nodes[3] = global_node_numbers[5];
					break;
				case (2):
					child_nodes[0] = global_node_numbers[7];
					child_nodes[1] = global_node_numbers[8];
					child_nodes[2] = global_node_numbers[2];
					child_nodes[3] = global_node_numbers[6];
					break;
				case (3):
					child_nodes[0] = global_node_numbers[8];
					child_nodes[1] = global_node_numbers[5];
					child_nodes[2] = global_node_numbers[6];
					child_nodes[3] = global_node_numbers[3];
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
				face_nodes[0] = this->nodes[0];
				face_nodes[1] = this->nodes[1];
				break;
			case (1):
				face_nodes[0] = this->nodes[1];
				face_nodes[1] = this->nodes[3];
				break;
			case (2):
				face_nodes[0] = this->nodes[3];
				face_nodes[1] = this->nodes[2];
				break;
			case (3):
				face_nodes[0] = this->nodes[2];
				face_nodes[1] = this->nodes[0];
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
	template <Float T>
	class VTK_VOXEL : public BaseElement<T> {
	public:
		using Point_t = BaseElement<T>::Point_t;
		VTK_VOXEL() : BaseElement<T>(8) {};
		VTK_VOXEL(const std::vector<size_t> &nodes) : BaseElement<T>(nodes) {assert(nodes.size()==N_NODES);}
		
		static constexpr int VTK_ID = 11;
		static constexpr int N_CHILDREN_WHEN_SPLIT = 8;
		static constexpr int N_VERTICES_WHEN_SPLIT = 27;
		static constexpr int N_NODES = 8;

		int vtkID() const override {return VTK_ID;}
		int nNodes() const override {return N_NODES;}
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

		void getChildNodes(std::vector<size_t> &child_nodes, const int child_number, const std::vector<size_t> &global_node_numbers) const override {
			assert(global_node_numbers.size()==N_VERTICES_WHEN_SPLIT);
			child_nodes.resize(N_NODES);

			switch (child_number) {
				case (0): //voxel element containing original vertex 0
					child_nodes[0] = global_node_numbers[ 0]; //0
					child_nodes[1] = global_node_numbers[ 8]; //0-1
					child_nodes[2] = global_node_numbers[11]; //0-2
					child_nodes[3] = global_node_numbers[24]; //0-3
					child_nodes[4] = global_node_numbers[12]; //0-4
					child_nodes[5] = global_node_numbers[23]; //0-5
					child_nodes[6] = global_node_numbers[20]; //0-6
					child_nodes[7] = global_node_numbers[26]; //0-7
					break;

				case (1): //voxel element containing original vertex 1
					child_nodes[0] = global_node_numbers[ 8]; //0-1
					child_nodes[1] = global_node_numbers[ 1]; //1
					child_nodes[2] = global_node_numbers[24]; //0-3
					child_nodes[3] = global_node_numbers[ 9]; //1-3
					child_nodes[4] = global_node_numbers[23]; //0-5
					child_nodes[5] = global_node_numbers[15]; //1-5
					child_nodes[6] = global_node_numbers[26]; //0-7
					child_nodes[7] = global_node_numbers[21]; //1-7
					break;

				case (2): //voxel element containing original vertex 2
					child_nodes[0] = global_node_numbers[11]; //0-2
					child_nodes[1] = global_node_numbers[24]; //0-3
					child_nodes[2] = global_node_numbers[ 2]; //2
					child_nodes[3] = global_node_numbers[10]; //2-3
					child_nodes[4] = global_node_numbers[20]; //0-6
					child_nodes[5] = global_node_numbers[26]; //0-7
					child_nodes[6] = global_node_numbers[13]; //2-6
					child_nodes[7] = global_node_numbers[22]; //2-7
					break;

				case (3): //voxel element containing original vertex 3
					child_nodes[0] = global_node_numbers[24]; //0-3
					child_nodes[1] = global_node_numbers[ 9]; //1-3
					child_nodes[2] = global_node_numbers[10]; //2-3
					child_nodes[3] = global_node_numbers[ 3]; //3
					child_nodes[4] = global_node_numbers[26]; //0-7
					child_nodes[5] = global_node_numbers[21]; //1-7
					child_nodes[6] = global_node_numbers[22]; //2-7
					child_nodes[7] = global_node_numbers[14]; //3-7
					break;

				case (4): //voxel element containing original vertex 4
					child_nodes[0] = global_node_numbers[12]; //0-4
					child_nodes[1] = global_node_numbers[23]; //0-5
					child_nodes[2] = global_node_numbers[20]; //0-6
					child_nodes[3] = global_node_numbers[26]; //0-7
					child_nodes[4] = global_node_numbers[ 4]; //4
					child_nodes[5] = global_node_numbers[16]; //4-5
					child_nodes[6] = global_node_numbers[19]; //4-6
					child_nodes[7] = global_node_numbers[25]; //4-7
					break;

				case (5): //voxel element containing original vertex 5
					child_nodes[0] = global_node_numbers[23]; //0-5
					child_nodes[1] = global_node_numbers[15]; //1-5
					child_nodes[2] = global_node_numbers[26]; //0-7
					child_nodes[3] = global_node_numbers[21]; //1-7
					child_nodes[4] = global_node_numbers[16]; //4-5
					child_nodes[5] = global_node_numbers[ 5]; //5
					child_nodes[6] = global_node_numbers[25]; //4-7
					child_nodes[7] = global_node_numbers[17]; //5-7
					break;

				case (6): //voxel element containing original vertex 6
					child_nodes[0] = global_node_numbers[20]; //0-6
					child_nodes[1] = global_node_numbers[26]; //0-7
					child_nodes[2] = global_node_numbers[13]; //2-6
					child_nodes[3] = global_node_numbers[22]; //2-7
					child_nodes[4] = global_node_numbers[19]; //4-6
					child_nodes[5] = global_node_numbers[25]; //4-7
					child_nodes[6] = global_node_numbers[ 6]; //6
					child_nodes[7] = global_node_numbers[18]; //6-7
					break;

				case (7): //voxel element containing original vertex 7
					child_nodes[0] = global_node_numbers[26]; //0-7
					child_nodes[1] = global_node_numbers[21]; //1-7
					child_nodes[2] = global_node_numbers[22]; //2-7
					child_nodes[3] = global_node_numbers[14]; //3-7
					child_nodes[4] = global_node_numbers[25]; //4-7
					child_nodes[5] = global_node_numbers[17]; //5-7
					child_nodes[6] = global_node_numbers[18]; //6-7
					child_nodes[7] = global_node_numbers[ 7]; //7
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
				face_nodes[0] = this->nodes[0];
				face_nodes[1] = this->nodes[4];
				face_nodes[2] = this->nodes[2];
				face_nodes[3] = this->nodes[6];
				break;
			case (1): //right
				face_nodes[0] = this->nodes[1];
				face_nodes[1] = this->nodes[3];
				face_nodes[2] = this->nodes[5];
				face_nodes[3] = this->nodes[7];
				break;
			case (2): //top
				face_nodes[0] = this->nodes[2];
				face_nodes[1] = this->nodes[6];
				face_nodes[2] = this->nodes[3];
				face_nodes[3] = this->nodes[7];
				break;
			case (3): //bottom
				face_nodes[0] = this->nodes[0];
				face_nodes[1] = this->nodes[1];
				face_nodes[2] = this->nodes[4];
				face_nodes[3] = this->nodes[5];
				break;
			case (4): //back
				face_nodes[0] = this->nodes[1];
				face_nodes[1] = this->nodes[0];
				face_nodes[2] = this->nodes[3];
				face_nodes[3] = this->nodes[2];
				break;
			case (5): //front
				face_nodes[0] = this->nodes[4];
				face_nodes[1] = this->nodes[5];
				face_nodes[2] = this->nodes[6];
				face_nodes[3] = this->nodes[7];
				break;
			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}
	};

}