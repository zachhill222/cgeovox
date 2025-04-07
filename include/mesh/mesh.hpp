#pragma once

#include "util/octree.hpp"
#include "util/point.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>
#include <armadillo>

namespace gv::mesh
{
	//mesh for one element type
	template <typename Element_t>
	class Mesh{
	protected:
		//sincle reference element
		static const Element_t referenceElement;

		gv::util::PointOctree<3,double,32> _nodes;
		std::vector<size_t> _elem2node;
		std::vector<size_t> _node2elem_start_idx; //each node belongs to an unknown number of elements. track where node2element begins for each node.
		std::vector<size_t> _node2elem;
		std::vector<std::vector<size_t>> _boundary; //assign which nodes belong to the boundary. allow for multiple boundaries.

	protected:
		//convert a pair of local node indices to a linear index. used for creating mass and stiffness matrices.
		static size_t _ij2lin(const size_t i, const size_t j) {return referenceElement.nNodes*i + j;}

	public:
		Mesh() {}
		
		std::vector<int> elem_marker;

		///// MESH MANIPULATION
		//get global node numbers for a particular element
		void get_element(const size_t idx, size_t (&element)[referenceElement.nNodes]) const
		{
			size_t start = idx*referenceElement.nNodes;
			for (size_t i=0; i<referenceElement.nNodes; i++) {element[i] = _elem2node[start+i];}
		}

		//add a single element and nodes
		void add_element(const gv::util::Point<3,double> (&element)[referenceElement.nNodes])
		{
			for (size_t i=0; i<referenceElement.nNodes; i++)
			{
				_nodes.push_back(element[i]);
				
				size_t global_idx;
				bool success = _nodes.find(element[i], global_idx);
				if (!success) {throw std::runtime_error("ERROR: can't find node while adding element to mesh. Possibly need to set the bbox for the _nodes octree.");}

				_elem2node.push_back(global_idx);
			}
		}

		//add a single element with known nodes
		void add_element(const size_t (&element)[referenceElement.nNodes])
		{
			for (size_t i=0; i<referenceElement.nNodes; i++)
			{
				_elem2node.push_back(element[i]);
			}
		}

		//add a single node. used if nodes are known before constructing mesh avoid error in doubles arithmetic for small mesh sizes.
		void add_node(const gv::util::Point<3,double> &node)
		{
			_nodes.push_back(node);
		}

		//get index for a node
		size_t node_idx(const gv::util::Point<3,double> &node) const
		{
			return _nodes.find(node);
		}

		//add node to a boundary
		void add_to_boundary(const size_t node_idx, const size_t boundary_idx = 0) {_boundary[boundary_idx].push_back(node_idx);}
		size_t create_new_boundary()
		{
			_boundary.push_back(std::vector<size_t> {});
			return _boundary.size()-1;
		}

		//reserve space for _elem2node
		void reserve(size_t nNewElems)
		{
			_elem2node.reserve(_elem2node.size()+referenceElement.nNodes*nNewElems);
		}

		//convenient size functions
		size_t nElems() const {return _elem2node.size()/referenceElement.nNodes;}
		size_t nNodes() const {return _nodes.size();}

		//convenient access functions
		size_t elem2node(size_t elem, size_t localnode) const {return _elem2node[elem*referenceElement.nNodes+localnode];}
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
				for (size_t i=0; i<referenceElement.nNodes; i++)
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
				for (size_t i=0; i<referenceElement.nNodes; i++)
				{
					size_t node = elem2node(el,i);
					size_t idx = _node2elem_start_idx[node] + node_count[node];
					_node2elem[idx] = el;
					node_count[node] += 1;
				}
			}
		}


		//get mesh sub-domain
		void get_subdomain(const int mkr, Mesh<Element_t> &out_mesh)
		{
			//TODO: make external and internal boundaries?

			//check that all elements are marked
			if (elem_marker.size() != nElems())
			{
				throw std::runtime_error("Not all elements of the original mesh are marked.");
			}

			//count number of elements
			size_t n_elems_out = 0;
			for (size_t el=0; el<nElems(); el++)
			{
				if (elem_marker[el]==mkr) {n_elems_out+=1;}
			}

			if (n_elems_out==0) {return;}


			//ensure bounding box for out_mesh is sufficiently large (_nodes octree)
			out_mesh.set_bbox(this->_nodes.bbox());

			//add elements
			for (size_t el=0; el<nElems(); el++)
			{
				if (elem_marker[el] == mkr)
				{
					//construct new element
					gv::util::Point<3,double> new_elem[referenceElement.nNodes];
					for (size_t i=0; i<referenceElement.nNodes; i++)
					{
						new_elem[i] = this->_nodes[this->elem2node(el,i)];
					}

					//add new element to out_mesh
					out_mesh.add_element(new_elem);
					out_mesh.elem_marker.push_back(mkr);
				}
			}
		}


		//set bounds for nodes
		void set_bbox(const gv::util::Box<3> &bbox) {_nodes.set_bbox(bbox);}


		////// MESH OUTPUTS
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

			//ELEMENT MARKERS
			if (elem_marker.size() == nElems())
			{
				buffer << "CELL_DATA " << nElems() << "\n";
				buffer << "SCALARS elem_marker integer\n";
				buffer << "LOOKUP_TABLE default\n";
				for (size_t el=0; el<nElems(); el++) {buffer << elem_marker[el] << " ";}
				buffer << "\n\n";
			}
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
				for (size_t n=0; n<referenceElement.nNodes; n++)
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

		//save mesh to file
		void saveas(std::string filename) const
		{
			//////////////// OPEN FILE ////////////////
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

		//////MASS MATRIX
		void makeMassMatrix(arma::sp_mat &massMat) const;
	};


	template<typename Element_t>
	void Mesh<Element_t>::makeMassMatrix(arma::sp_mat &massMat) const
	{
		//set up index tracking to allow parallel looping over elements when computing integrals
		arma::umat locations(2,referenceElement.nNodes*referenceElement.nNodes*nElems());
		arma::vec values(referenceElement.nNodes*referenceElement.nNodes*nElems());

		//integrate over each element
		#pragma omp parallel
		for (size_t el=0; el<nElems(); el++)
		{
			//set parameters for this element
			gv::util::Point<3,double> H = _nodes[elem2node(el,7)] - _nodes[elem2node(el,0)]; //size of voxel
			size_t start = el*referenceElement.nNodes*referenceElement.nNodes; //start of this element's block in locations and values.

			//compute contributions to mass matrix
			for (size_t i=0; i<8; i++)
			{
				//global node number for local node i
				size_t global_i = elem2node(el,i);

				//diagonal (i,i) entry
				values.at(start + _ij2lin(i,i)) = referenceElement.integrate_mass(i,i,H);
				locations.at(0,start + _ij2lin(i,i)) = global_i;
				locations.at(1,start + _ij2lin(i,i)) = global_i;

				//off-diagonal entries
				for (size_t j=i+1; j<8; j++)
				{
					//global node number for local node j
					size_t global_j = elem2node(el,j);

					//get value
					double val = referenceElement.integrate_mass(i,j,H);
					
					//store location 1
					values[_ij2lin(i,j)] = val;
					locations.at(0,start + _ij2lin(i,j)) = global_i;
					locations.at(1,start + _ij2lin(i,j)) = global_j;

					//store location 2
					values[_ij2lin(j,i)] = val;
					locations.at(0,start + _ij2lin(j,i)) = global_j;
					locations.at(1,start + _ij2lin(j,i)) = global_i;
				}
			}
		}

		//construct matrix
		massMat = arma::sp_mat(true, locations, values, nNodes(), nNodes(), true, false);
	}



	
}