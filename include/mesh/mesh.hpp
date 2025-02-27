#pragma once

#include "util/octree.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <algorithm>

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
		std::vector<size_t> _node2elem_start_idx; //each node belongs to an unknown number of elements. track where node2element begins for each node.
		std::vector<size_t> _node2elem;

	public:
		Mesh() {}

		//vtkID for element type
		int vtkID = 11; //voxel default
		
		//get global node numbers for a particular element
		void get_element(const size_t idx, size_t (&element)[nodes_per_element]) const
		{
			size_t start = idx*nodes_per_element;
			for (size_t i=0; i<nodes_per_element; i++) {element[i] = _elem2node[start+i];}
		}

		//add a single element
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
		void reserve(size_t nNewElems) {_elem2node.reserve(_elem2node.size()+nodes_per_element*nNewElems);}

		//convenient size functions
		size_t nElems() const {return _elem2node.size()/nodes_per_element;}
		size_t nNodes() const {return _nodes.size();}

		//convenient access functions
		size_t elem2node(size_t elem, size_t localnode) const {return _elem2node[elem*nodes_per_element+localnode];}
		size_t local_nElem(size_t node) const
		{
			if (node==nNodes()-1) {return _node2elem.size()-_node2elem_start_idx[node];}
			return _node2elem_start_idx[node+1]-_node2elem_start_idx[node];
		}
		size_t node2elem(size_t node, size_t localelem) const
		{
			//return max size_t as an error
			if (localelem >= local_nElem(node)) {return (size_t) -1;}
			return _node2elem[_node2elem_start_idx[node] + localelem];
		}


		//compute mesh connectivity
		void compute_connectivity()
		{
			//compute number of elements for each node
			std::vector<size_t> node_count(nNodes(), 0);
			for (size_t el=0; el<nElems(); el++)
			{
				for (size_t i=0; i<nodes_per_element; i++)
				{
					size_t node = elem2node(el,i);
					node_count[node] += 1;
				}
			}

			//convert number of elements to start index
			_node2elem_start_idx.resize(nNodes(),0);
			_node2elem_start_idx[0]=0;

			for (size_t n=1; n<nNodes(); n++)
			{
				_node2elem_start_idx[n] = _node2elem_start_idx[n-1] + node_count[n-1];
			}

			//compute node2element
			_node2elem.resize(_node2elem_start_idx[nNodes()-1] + node_count[nNodes()-1]);
			std::fill(node_count.begin(), node_count.end(), 0); //use node_count for tracking current index for each node
			for (size_t el=0; el<nElems(); el++)
			{
				for (size_t i=0; i<nodes_per_element; i++)
				{
					size_t node = elem2node(el,i);
					size_t idx = _node2elem_start_idx[node] + node_count[node];
					_node2elem[idx] = el;
					node_count[node] += 1;
				}
			}
		}

		//set bounds for nodes
		void set_bbox(const gv::util::Box<3> &bbox) {_nodes.set_bbox(bbox);}

		//print mesh structure to ostream
		void vtkprint(std::ostream &stream) const
		{
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
			buffer << "CELLS " << nElems() << " " << (1+nodes_per_element)*nElems() << "\n";
			for (size_t i=0; i<nElems(); i++)
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
			buffer << "CELL_TYPES " << nElems() << "\n";
			for (size_t i=0; i<nElems(); i++) {buffer << vtkID << " ";}
			buffer << "\n\n";
			stream << buffer.rdbuf();
			buffer.str("");
		}

		//print mesh connectivity for error checking
		void print_connectivity() const
		{
			std::cout << "ELEMENT2NODE:\n";
			for (size_t el=0; el<nElems(); el++)
			{
				std::cout << "element " << el << "\t: ";
				for (size_t n=0; n<nodes_per_element; n++)
				{
					std::cout << elem2node(el,n) << " ";
				}
				std::cout << std::endl;
			}


			std::cout << "\nNODE2ELEMENT:\n";
			for (size_t n=0; n<nNodes(); n++)
			{
				std::cout << "node " << n << "\t: ";
				for (size_t el=0; el<local_nElem(n); el++)
				{
					std::cout << node2elem(n,el) << " ";
				}
				std::cout << std::endl;
			}
		}
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

		////// PRINT MESH STRUCTURE TO CONSOLE
		// octree_mesh.compute_connectivity();
		// octree_mesh.print_connectivity();
		//////////////////////////////////////

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
		
		// buffer << "SCALARS elementID integer\n";
		// buffer << "LOOKUP_TABLE default\n";
		// for (size_t i=0; i<octree_mesh.nElems(); i++)
		// {
		// 	buffer << i << "\n";
		// }
		// buffer << "\n";
		// meshfile << buffer.rdbuf();
		// buffer.str("");

		// buffer << "POINT_DATA " << octree_mesh.nNodes() << std::endl;
		// buffer << "SCALARS nodeID integer\n";
		// buffer << "LOOKUP_TABLE default\n";
		// for (size_t i=0; i<octree_mesh.nNodes(); i++)
		// {
		// 	buffer << i << "\n";
		// }
		// buffer << "\n";
		// meshfile << buffer.rdbuf();
		// buffer.str("");

		//close file
		meshfile.close();
	}
}