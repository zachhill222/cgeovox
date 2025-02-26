#pragma once

#include "util/octree.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

namespace gv::mesh
{
	//mesh for one element type
	template <size_t nodes_per_element=8>
	class Mesh{
	private:
		gv::util::PointOctree<3,double,32> _nodes;
		std::vector<size_t> _elem2node;
		
	public:
		Mesh() {}
		
		//get global node numbers for a particular element
		void get_element(const size_t idx, size_t (&element)[nodes_per_element]) const
		{
			size_t start = idx*nodes_per_element;
			for (size_t i=0; i<nodes_per_element; i++) {element[i] = _elem2node[start+i];}
		}

		//add an element
		void add_element(const gv::util::Point<3,double> (&element)[nodes_per_element])
		{
			for (size_t i=0; i<nodes_per_element; i++)
			{
				_nodes.push_back(element[i]);
				size_t global_idx = _nodes.find(element[i]);
				_elem2node.push_back(global_idx);

			}
		}

		//reserve space for _elem2node
		void reserve(size_t nElems) {_elem2node.reserve(_elem2node.size()+nodes_per_element*nElems);}

		//set bounds for nodes
		void set_bbox(const gv::util::Box<3> &bbox) {_nodes.set_bbox(bbox);}

		//print mesh structure to ostream
		void vtkprint(std::ostream &stream) const
		{
			size_t nElems = _elem2node.size()/nodes_per_element;
			std::stringstream buffer;

			//HEADER
			buffer << "# vtk DataFile Version 2.0\n";
			buffer << "Mesh Data\n";
			buffer << "ASCII\n\n";
			buffer << "DATASET UNSTRUCTURED_GRID\n";

			//POINTS
			buffer << "POINTS " << _nodes.size() << " float\n";
			for (size_t i=0; i<_nodes.size(); i++) { buffer << _nodes[i] << "\n";}
			buffer << "\n";
			stream << buffer.rdbuf();
			buffer.str("");

			//ELEMENTS
			buffer << "CELLS " << nElems << " " << (1+nodes_per_element)*nElems << "\n";
			for (size_t i=0; i<nElems; i++)
			{
				buffer << nodes_per_element << " ";
				for (size_t j=0; j<nodes_per_element; j++)
				{
					buffer << _elem2node[nodes_per_element*i + j] << " ";
				}
				buffer << "\n";
			}
			buffer << "\n";
			stream << buffer.rdbuf();
			buffer.str("");

			//VTK IDs
			buffer << "CELL_TYPES " << nElems << "\n";
			for (size_t i=0; i<nElems; i++) {buffer << vtkID << " ";}
			buffer << "\n\n";
			stream << buffer.rdbuf();
			buffer.str("");
		}

		//vtkID for element type
		int vtkID = 11; //voxel default
	};


	///function for viewing octree structure, only works in 3 dimensions
	template <typename Octree_t>
	void view_octree_vtk(const Octree_t &octree, const std::string filename="octree_structure.vtk")
	{	
		const int n_children = octree.n_children;
		const size_t nElems = octree.nNodes();

		//initialize mesh
		gv::mesh::Mesh<n_children> octree_mesh;
		octree_mesh.set_bbox(octree.bbox());
		octree_mesh.reserve(nElems);

		octree_mesh.vtkID = 11; //VTK_VOXEL

		//create an element for each node
		for (size_t idx=0; idx<nElems; idx++)
		{
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

		buffer << "CELL_DATA " << nElems << std::endl;
		buffer << "SCALARS nData integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (size_t i=0; i<nElems; i++)
		{
			buffer << octree.nData(i) << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();

		buffer << "SCALARS isLeaf integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (size_t i=0; i<nElems; i++)
		{
			buffer << octree.isLeaf(i) << "\n";
		}
		buffer << "\n";
		meshfile << buffer.rdbuf();

		//close file
		meshfile.close();
	}
}