#pragma once

#include "compile_constants.hpp"

#include <cassert>
#include <array>

namespace gv::util
{
	/*
	//////////// LEVEL INDEX GUIDE (SAME AS VTK VOXEL NODES) for dim=3 (dim=2 is the same with the k-level removed)
	// node 1 is hidden
	// 
	// i=0: 0,2,4,6 (x=-1)
	// i=1: 1,3,5,7 (x= 1)
	// 
	// j=0: 0,1,4,5 (y=-1)
	// j=1: 2,3,6,7 (y= 1)
	//
	// k=0: 0,1,2,3 (z=-1)
	// k=1: 4,5,6,7 (z= 1)
	//
	//		2------3
	//		|\      \
	//		| \      \
	//		0  6------7
	//		 \ |      |
	//		  \|      |
	//		   4------5
	//
	////////////////////////////////////////////////////////
	*/


	///class for refering to octree paths
	template <int dim=3>
	class OctreePath
	{
	private:
		static const int max_depth = gv::constants::OCTREE_MAX_DEPTH;

		///store depth of the node in octree
		int depth = 0;

		///store path in octree to the node
		std::array<std::array<bool,dim>, max_depth> path;

		///cartesian index of the node at its finest level. each index is between 0 and 2^dim-1
		size_t idx[dim] {0};
		
		///global node number
		///each possible node at each level of the octree can be listed. this function returns a unique index for each distinct node
		///there are L_n=(2^n)^dim possible nodes on level n
		///there are (L_n-1)/(2^dim - 1) possible nodes on levels strictly less than n
		size_t global_node_number = 0;

	public:
		//defaults to root node parameters
		OctreePath() {}

		//constructor for use in octrees (i.e. splitting a parent node)
		OctreePath(const OctreePath &parent, bool child[dim]) : depth(parent.depth+1), path(parent.path)
		{
			assert((this->depth <= max_depth) && void("OCTREE_MAX_DEPTH exceeded"));

			for (int k=0; k<dim; k++)
			{
				path[k][depth-1] = child[k]; //append to path
				idx[k] = 2*parent.idx[k] + ((size_t) child[k]); //update ijk index
			}

			global_node_number = (std::pow(2,depth*dim)-1)/(std::pow(2,dim)-1); //offset by the number of nodes on previous levels, smallest possible index for any node at current depth
			size_t step_size = std::pow(2,depth-1); //number of nodes to increment by on the current level
			size_t local_node_number = idx[dim]; //add to offset
			for (int k=dim; k>=0; k--)
			{
				local_node_number += idx[k] + step_size*local_node_number;
			}
			global_node_number += local_node_number;
		}
	};


}