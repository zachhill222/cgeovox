#pragma once

#include "mesh/homo_mesh.hpp" //for writing vtk files
#include "mesh/vtk_voxel.hpp"

#include "util/octree.hpp"

#include <cassert>
#include <vector>
#include <iostream>

namespace gv::util
{
	///function for viewing octree structure, only works in 3 dimensions
	template <typename Octree_t>
	void view_octree_vtk(const Octree_t &octree, const std::string filename="octree_structure.vtk")
	{	
		std::vector<const typename Octree_t::Node_t*> nodes = octree._get_node(octree.bbox()); //get nodes from the octree with data

		const size_t nElems = nodes.size(); //each node will be a voxel element in the visualization

		//initialize mesh
		gv::mesh::HomoMesh<gv::mesh::Voxel> octree_mesh(octree.bbox());
		octree_mesh.reserve_elems(nElems);
		octree_mesh.reserve_nodes(8*nodes.size());
		//create an element for each node
		for (size_t idx=0; idx<nElems; idx++)
		{
			//make element
			gv::util::Point<3,double> element[8];

			for (int j=0; j<8; j++)
			{
				element[j] = nodes[idx]->bbox[j];
			}

			//add element to mesh
			octree_mesh.add_element(element);
		}

		//print mesh to file
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}

		octree_mesh.vtkprint(meshfile);

		//print number of data in each node to meshfile
		std::stringstream buffer;

		buffer << "CELL_DATA " << octree_mesh.nElems() << std::endl;
		buffer << "SCALARS nData integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (size_t i=0; i<octree_mesh.nElems(); i++)
		{
			buffer << nodes[i]->data_cursor << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();
		buffer.str("");

		//close file
		meshfile.close();
	}
}