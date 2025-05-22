#pragma once

//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder

#include "fem/charms_Q1_element.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

namespace gv::fem
{
	class CharmsQ1_3DMesh
	{
		using Element_t = Q1_3D_element;
		using Basis_t   = Q1_3D_basis;
		using Point_t   = gv::util::Point<3,double>;
		using Box_t     = gv::util::Box<3>;

		

		Q1_3D_element_list active_elements;
		Q1_3D_basis_list active_basis;
		gv::util::PointOctree<3> mesh_nodes;

		//add nodes to mesh and update element reference if necessary
		void add_to_mesh(Element_t* elem)
		{
			for (int i=0; i<elem->n_nodes; i++)
			{
				Point_t	vertex = elem->bbox.voxelvertex(i);
				assert(mesh_nodes.push_back(vertex)!=-1); //-1 means that there was an error
				elem->global_nodes[i] = mesh_nodes.find(vertex);
			}
		}

		//divide an element
		void divide(Element_t* elem)
		{
			//divide element
			elem->divide();

			//ensure mesh data is correct
			for (int i=0; i<elem->n_children; i++) {add_to_mesh(elem->children[i]);}
		}

		///save mesh to file
		void save_as(std::string filename) const;
		void vtkprint(std::ostream& os) const;
	}













	//save mesh to file (vtk format)
	void CharmsQ1_3DMesh::save_as(std::string filename) const
	{
		//open and check file
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}

		//print mesh to file
		vtkprint(meshfile);
		meshfile.close();
	}

	void CharmsQ1_3DMesh::vtkprint(std::ostream& os) const
{
	//write to buffer and flush buffer to the stream
	std::stringstream buffer;

	//HEADER
	buffer << "# vtk DataFile Version 2.0\n";
	buffer << "Mesh Data\n";
	buffer << "ASCII\n\n";
	buffer << "DATASET UNSTRUCTURED_GRID\n";

	//POINTS
	buffer << "POINTS " << nNodes() << " float\n";
	for (size_t i=0; i<nNodes(); i++) { buffer << _nodes[i] << "\n";}
	buffer << "\n";
	stream << buffer.rdbuf();
	buffer.str("");

	//ELEMENTS
	buffer << "CELLS " << nElems() << " " << (1+referenceElement.nNodes)*nElems() << "\n";
	for (size_t i=0; i<nElems(); i++)
	{
		buffer << referenceElement.nNodes << " ";
		for (size_t j=0; j<referenceElement.nNodes; j++)
		{
			buffer << _elem2node[referenceElement.nNodes*i + j] << " ";
		}
		buffer << "\n";
	}
	buffer << "\n";
	stream << buffer.rdbuf();
	buffer.str("");

	//VTK IDs
	buffer << "CELL_TYPES " << nElems() << "\n";
	for (size_t i=0; i<nElems(); i++) {buffer << referenceElement.vtkID << " ";}
	buffer << "\n\n";
	stream << buffer.rdbuf();
	buffer.str("");
}


}