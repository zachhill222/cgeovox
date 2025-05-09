#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/homo_mesh.hpp"
#include "mesh/vtkVoxel.hpp"


namespace gv::util
{
	///function for viewing octree structure, only works in 3 dimensions
	template <typename Octree_t>
	void view_octree_vtk(const Octree_t &octree, const std::string filename="octree_structure.vtk")
	{	
		const size_t nElems = octree.nNodes();

		//initialize mesh
		gv::mesh::HomoMesh<gv::mesh::Voxel> octree_mesh;
		octree_mesh.set_bbox(octree.bbox());
		octree_mesh.reserve(nElems);

		//create an element for each node
		for (size_t idx=0; idx<nElems; idx++)
		{
			if (not octree.isLeaf(idx)) {continue;}

			//make element
			gv::util::Point<3,double> element[8];
			gv::util::Box<3> box = octree.bbox(idx);


			for (int j=0; j<8; j++)
			{
				element[j] = box[j];
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
			buffer << octree.nData(i) << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();
		buffer.str("");

		//close file
		meshfile.close();
	}
}