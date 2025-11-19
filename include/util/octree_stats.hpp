#pragma once

#include "util/octree_parallel.hpp"
#include "util/octree_util.hpp"
#include <iostream>

namespace gv::util {

	// template<typename Data_t, bool SINGLE_DATA, int DIM, int N_DATA, Float T>
	// std::ostream& operator<<(std::ostream &os, const BasicParallelOctree<Data_t, SINGLE_DATA,DIM,N_DATA,T> &octree) {
	// 	treeSummary(os, octree);
	// 	return os;
	// }

	/// Print tree summary information to the specified ostream
	template<typename Data_t, bool SINGLE_DATA, int DIM, int N_DATA, Float T>
	void treeSummary(std::ostream &os, const BasicParallelOctree<Data_t, SINGLE_DATA,DIM,N_DATA,T> &octree) {
		//get the number of nodes and total data indices stored in the nodes
		size_t nNodes{0}, nIdx{0}, nIdxCap{0}, nLeafs{0};
		int maxDepth{0};

		_recursive_node_properties(octree._root, nNodes, nIdx, nIdxCap, nLeafs, maxDepth);


		os << "number of data stored: " << octree.size() << "\n";
		os << "number of nodes: " << nNodes << "\n";
		os << "number of leafs: " << nLeafs << "\n";
		os << "number of stored indices: " << nIdx << " (capacity= " << nIdxCap << ")\n";
		os << "total tree depth: " << maxDepth - octree._root->depth << "\n";
	}


	template<int DIM=3, int N_DATA=16, Float T=float>
	void _recursive_node_properties(const OctreeParallelNode<DIM,N_DATA,T>* node, size_t &n_nodes, size_t &n_idx, size_t &n_idx_cap, size_t &n_leafs, int &max_depth) {
		if (node == nullptr) {return;}
		n_nodes++;
		
		if (isLeaf(node)) {n_leafs++;}

		if (node->data_idx != nullptr) {
			n_idx += node->cursor;
			n_idx_cap += N_DATA;
		}

		max_depth = std::max(max_depth, node->depth);

		for (int c = 0; c < OctreeParallelNode<DIM,N_DATA,T>::N_CHILDREN; c++) {
			_recursive_node_properties<DIM,N_DATA,T>(node->children[c], n_nodes, n_idx, n_idx_cap, n_leafs, max_depth);
		}
	}



	/// Print memory management to the specified stream
	template<typename Data_t, bool SINGLE_DATA, int DIM, int N_DATA, Float T>
	void memorySummary(std::ostream &os, const BasicParallelOctree<Data_t, SINGLE_DATA,DIM,N_DATA,T> &octree) {
		
	}
}



	