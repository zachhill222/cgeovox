#pragma once

#include "concepts.hpp"
#include "util/octree_parallel.hpp"
#include "util/octree_util.hpp"
#include "mesh/mesh_basic.hpp"

#include <string>
#include <iostream>
#include <vector>
#include <functional>

namespace gv::util {

	template<typename Data_t, bool SINGLE_DATA, int DIM, int N_DATA, Scalar T>
	void makeOctreeLeafMesh(const BasicParallelOctree<Data_t, SINGLE_DATA, DIM, N_DATA, T> &octree, const std::string filename) {
		using Vertex_t  = gv::util::Point<3,T>;
		using Node_t    = gv::mesh::BasicNode<Vertex_t>;
		using Face_t    = gv::mesh::BasicElement;
		using Element_t = gv::mesh::BasicElement;
		using Mesh_t    = gv::mesh::BasicMesh<Node_t,Element_t,Face_t>;

		Mesh_t mesh(octree.bbox());
		std::vector<int> nIdx;

		using OctreeNode_t = typename BasicParallelOctree<Data_t, SINGLE_DATA, DIM, N_DATA, T>::Node_t;
		std::function<void(const OctreeNode_t* node)> recursive_add_leafs = [&](const OctreeNode_t* node) {
			if (node==nullptr) {return;}

			if (isLeaf(node)) {
				std::vector<Vertex_t> vertices(OctreeNode_t::N_CHILDREN);
				for (int c = 0; c < OctreeNode_t::N_CHILDREN; c++) {
					vertices[c] = node->bbox.voxelvertex(c);
				}

				static_assert(DIM==3 or DIM==2, "The octree must be in 2 or 3 dimensions");
				if constexpr (DIM==3) {
					mesh.constructElement_Locked(vertices, 11);
				}
				if constexpr (DIM==2) {
					mesh.constructElement_Locked(vertices, 8);
				}



				nIdx.push_back(node->cursor);
			} else {
				for (int c = 0; c < OctreeNode_t::N_CHILDREN; c++) {
					recursive_add_leafs(node->children[c]);
				}
			}
		};

		recursive_add_leafs(octree._root);
		mesh.save_as(filename, false, true); //no details, ascii format

		
	}
	
}



	