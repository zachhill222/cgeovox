#pragma once

// #include "mesh/homo_mesh.hpp" //for writing vtk files
// #include "mesh/vtkVoxel.hpp"

#include "util/octree.hpp"

#include <cassert>
#include <vector>

namespace gv::util
{

	//create a friend class to the base octree to avoid putting utility/debugging methods in the main class
	template<typename Octree_t>
	class OctreeInspector
	{
	public:
		OctreeInspector(Octree_t& octree) : octree(octree) {}

		void save_octree_structure(const std::string filename="octree_structure.vtk") const;
		void print() const;

	private:
		Octree_t& octree;
	};


	template<typename Octree_t>
	void OctreeInspector<Octree_t>::print() const
	{
		//get all leaf nodes
		std::vector<typename Octree_t::Node_t*> nodes = octree._get_node(octree.bbox());

		//print data from all leaf nodes
		for (size_t i=0; i<nodes.size(); i++)
		{
			if (nodes[i]->data_cursor==0) {continue;}

			std::cout << "===== node " << i << " =====\n";
			std::cout << "\tbbox= " << nodes[i]->bbox << "\n";
			std::cout << "\tdepth= " << nodes[i]->depth << "\n";
			std::cout << "\tdata: (" << nodes[i]->data_cursor << ")\n";
			for (int j=0; j<nodes[i]->data_cursor; j++)
			{
				std::cout << "\t\tdata_idx[" << j <<"]= " << nodes[i]->data_idx[j];
				std::cout << " (" << octree[nodes[i]->data_idx[j]] << ")\n";
			}
			std::cout << std::endl;
		}
	}




	// ///function for viewing octree structure, only works in 3 dimensions
	// template <typename Octree_t>
	// void OctreeInspector<Octree_t>::save_octree_structure(const std::string filename="octree_structure.vtk")
	// {	
	// 	using Node_t = Octree_t::Node_t;
	// 	std::vector<Node_t*> leaves = octree._get_node(octree.root->bbox); //put all leaf nodes into a vector

	// 	//initialize mesh
	// 	gv::mesh::HomoMesh<gv::mesh::Voxel> octree_mesh;
	// 	octree_mesh.set_bbox(octree.bbox());
	// 	octree_mesh.reserve(nElems);

	// 	//create an element for each node
	// 	for (size_t idx=0; idx<nElems; idx++)
	// 	{
	// 		if (not octree.isLeaf(idx)) {continue;}

	// 		//make element
	// 		gv::util::Point<3,double> element[8];
	// 		gv::util::Box<3> box = octree.bbox(idx);


	// 		for (int j=0; j<8; j++)
	// 		{
	// 			element[j] = box[j];
	// 		}

	// 		//add element to mesh
	// 		octree_mesh.add_element(element);
	// 	}

	// 	//print mesh to file
	// 	std::ofstream meshfile(filename);

	// 	if (not meshfile.is_open()){
	// 		std::cout << "Couldn't write to " << filename << std::endl;
	// 		meshfile.close();
	// 		return;
	// 	}

	// 	octree_mesh.vtkprint(meshfile);

	// 	//print number of data in each node to meshfile
	// 	std::stringstream buffer;

	// 	buffer << "CELL_DATA " << octree_mesh.nElems() << std::endl;
	// 	buffer << "SCALARS nData integer\n";
	// 	buffer << "LOOKUP_TABLE default\n";
	// 	for (size_t i=0; i<octree_mesh.nElems(); i++)
	// 	{
	// 		buffer << octree.nData(i) << "\n";
	// 	}
	// 	buffer << "\n";
	// 	meshfile << buffer.rdbuf();
	// 	buffer.str("");

	// 	//close file
	// 	meshfile.close();
	// }



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
	// template <int dim=3>
	// class OctreePath
	// {
	// private:
	// 	static const int max_depth = gv::constants::OCTREE_MAX_DEPTH;

	// 	///store depth of the node in octree
	// 	int depth = 0;

	// 	///store path in octree to the node
	// 	std::array<std::array<bool,dim>, max_depth> path;

	// 	///cartesian index of the node at its finest level. each index is between 0 and 2^dim-1
	// 	size_t idx[dim] {0};
		
	// 	///global node number
	// 	///each possible node at each level of the octree can be listed. this function returns a unique index for each distinct node
	// 	///there are L_n=(2^n)^dim possible nodes on level n
	// 	///there are (L_n-1)/(2^dim - 1) possible nodes on levels strictly less than n
	// 	size_t global_node_number = 0;

	// public:
	// 	//defaults to root node parameters
	// 	OctreePath() {}

	// 	//constructor for use in octrees (i.e. splitting a parent node)
	// 	OctreePath(const OctreePath &parent, bool child[dim]) : depth(parent.depth+1), path(parent.path)
	// 	{
	// 		assert((this->depth <= max_depth) && void("OCTREE_MAX_DEPTH exceeded"));

	// 		for (int k=0; k<dim; k++)
	// 		{
	// 			path[k][depth-1] = child[k]; //append to path
	// 			idx[k] = 2*parent.idx[k] + ((size_t) child[k]); //update ijk index
	// 		}

	// 		global_node_number = (std::pow(2,depth*dim)-1)/(std::pow(2,dim)-1); //offset by the number of nodes on previous levels, smallest possible index for any node at current depth
	// 		size_t step_size = std::pow(2,depth-1); //number of nodes to increment by on the current level
	// 		size_t local_node_number = idx[dim]; //add to offset
	// 		for (int k=dim; k>=0; k--)
	// 		{
	// 			local_node_number += idx[k] + step_size*local_node_number;
	// 		}
	// 		global_node_number += local_node_number;
	// 	}
	// };




}