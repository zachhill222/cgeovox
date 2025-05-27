#pragma once

//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder


#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "util/octree.hpp"
#include "compile_constants.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

namespace gv::fem
{

	///// DEFINE BASIS ////
	template<int dim>
	class Basis_Q1
	{
	public:
		Basis_Q1(const int depth, const Index_t& index) : depth(depth), index(index) {}

		using Box_t   = gv::util::Box<dim>;
		using Point_t = gv::util::Point<dim,double>;
		using Index_t = gv::util::Point<dim,size_t>;

		int depth = 0;
		Index_t index; // 0 <= i,j,k <= 2^depth
		bool is_active = false;
		std::vector<size_t> support_idx;

		//track domain
		Box_t* domain;
		
		//get coordinate associated with this basis function
		Point_t coord() const
		{
			Point_t H = std::pow(0.5,depth) * domain->sidelength();
			Point_t result;
			for (int k=0; k<dim; k++) {result[k] = domain->low()[k] + H[k] * (double) index[k];}
			return result;
		}

		//check if the corresponding vertex has occured at a lower depth
		bool is_odd() const
		{
			for (int k=0; k<dim; k++)  {if (index[k]%2) {return true}}
			return false;
		}

		bool operator==(const Basis_Q1& other) const
		{
			if (depth!=other.depth) {return false;}
			return index==other.index;
		}
	}

	//octree for storing basis function
	template<int dim=3>
	class BasisOctree_Q1 : public gv::util::BasicOctree<Basis_Q1<dim>, dim, false, 32>
	{
	public:
		BasisOctree_Q1(const gv::util::Box<dim> &bbox) : gv::util::BasicOctree<Basis_Q1<dim>, dim, false, 32>(bbox) {}
	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const Basis_Q1<dim> &data) const override  {return box.contains(data.coord());}
	}


	///// DEFINE ELEMENTS ////
	template<int dim=3>
	class Element_Q1
	{
	public:
		Element_Q1(const int depth, const Index_t& index) : depth(depth), index(index) {}

		using Point_t = gv::util::Point<dim,double>;
		using Index_t = gv::util::Point<dim,size_t>;
		using Box_t   = gv::util::Box<dim>;

		Box_t* domain = nullptr;

		static const Box_t index_helper(0,1); //helpful for converting between ijk[dim] and vertex number

		int depth = 0;
		size_t Index_t index; //0 <= i,j,k < 2^depth
		bool is_active = false;

		std::vector<size_t> ancestor_basis;
		std::vector<size_t> natural_basis;
		
		bool operator==(const Element_Q1& other) const
		{
			if (depth!=other.depth) {return false;}
			for (int k=0; k<dim; k++) {if (index[k]!=other.index[k]) {return false;}}
			return true;
		}

		Box_t bbox() const
		{
			Point_t H = std::pow(0.5,depth) * domain->sidelength();
			Point_t low;
			for (int k=0; k<dim; k++) {low[k] = domain->low()[k] + H[k] * (double) index[k];}
			return Box_t(low, low+H);
		}

		Element_Q1 child(const int n) const
		{
			//initialize
			Element_Q1 _child;
			_child.domain = domain;
			_child.depth = depth+1;

			//set index
			for (int k=0; k<dim; k++) {_child.index[k] = 2*index[k] + (size_t) index_helper.voxelvertex(n)[k];}
			
			//set basis
			for (size_t k=0; k<ancestor_basis.size(); k++) {_child.ancestor_basis.push_back(ancestor_basis[k]);}
			for (size_t k=0; k<natural_basis.size(); k++)  {_child.ancestor_basis.push_back(natural_basis[k]);}
		}
	}

	//octree for storing elements efficiently (searching for elements is needed)
	template<int dim=3, size_t n_data=32>
	class ElementOctree_Q1 : public gv::util::BasicOctree<Element_Q1<dim>, dim, true, n_data>
		{
	public:
		ElementOctree_Q1(const gv::util::Box<dim> &bbox) : gv::util::BasicOctree<Element_Q1<dim>, dim, true, n_data>(const gv::util::Box<dim> &bbox) {}
	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const Element_Q1<dim> &data) const override {return box.intersects(data.bbox());}
	};



	



	// class for interacting with the CHARMS mesh
	class CharmsQ1_3DMesh
	{
	public:
		CharmsQ1_3DMesh(const Box_t& domain) : domain(domain), vertices(domain), elements(domain), basis(domain)
		{
			//create initial element
			Element_t elem(0, Index_t(0,0,0));
			elem.domain = &domain;
			elem.is_active = true;
			elements.push_back(elem);

			//create initial basis functions
			for (int k=0; k<8; k++)
			{
				Basis_t fun(0, domain.voxelvertex(k));
				fun.is_active = true;
				fun.domain = &domain;
				fun.support_idx.push_back(0); //first element index is 0
				basis.push_back(fun); //add basis function to list of all basis functions
				elements[0].natural_basis.push_back(k); //this basis function is the k-th basis function
			}
		}

		static const size_t NAN = (size_t) -1;
		using Index_t   = gv::util::Point<3,size_t>;
		using Point_t   = gv::util::Point<3,double>;
		using Box_t     = gv::util::Box<3>;
		using Basis_t   = Basis_Q1<3>;
		using Element_t = Element_Q1<3>;
		
		///domain bounding box
		const Box_t domain; //physical bounding box for the domain

		///mesh storage
		gv::util::PointOctree<3> vertices; //track nodes by their ijk and depth
		ElementOctree_Q1<3> elements; //track elements
		BasisOctree_Q1<3> basis; //track basis functions

		///activate a given basis function
		void activate(const int depth, const Index_t &fun_index)
		{
			//initialize basis
			Basis_t fun(depth,fun_index);
			fun.domain = &domain;

			//attempt to add basis
			int flag = basis.push_back(fun);
			assert(flag!=-1); //basis is outside of the domain
			
			size_t f_idx = basis.find(fun);
			if (basis[f_idx].is_active) {return;} //basis was already active. nothing to do.

			//mark basis as active
			basis[f_idx].is_active = true;

			//assign support elements
			for (size_t i=0; i<2; i++)
			{
				if (fun_index[0]==0 and i==1) {continue;} //element would be outside the domain
				for (size_t j=0; j<2; j++)
				{
					if (fun_index[2]==0 and j==1) {continue;} //element would be outside the domain
					for (size_t k=0; k<2; k++)
					{
						if (fun_index[2]==0 and k==1) {continue;} //element would be outside the domain
						
						Index_t elem_index = fun_index - Index_t(i,j,k);
						Element_t elem(depth, elem_index);
						elem.domain = &domain;

						size_t e_idx = elements.find(elem); //attempt to find the element
						
					}
				}
			}


		}
	};



	// //erase all nodes and re-create unstructured mesh
	// void CharmsQ1_3DMesh::remake_mesh()
	// {
	// 	//clear current nodes
	// 	mesh_nodes.clear();

	// 	//add active elements to unstructured mesh
	// 	for (size_t idx=0; idx<nElems(); idx++)
	// 	{
	// 		Element_t* elem = active_elements[idx];
	// 		for (size_t n=0; n<8; n++)
	// 		{
	// 			Point_t vertex = elem->bbox.voxelvertex(n);
	// 			mesh_nodes.push_back(vertex);
	// 			elem->global_nodes[n] = mesh_nodes.find(vertex);
	// 		}
	// 	}
	// }

	// //save mesh to file (vtk format)
	// void CharmsQ1_3DMesh::save_as(std::string filename) const
	// {
	// 	//open and check file
	// 	std::ofstream meshfile(filename);

	// 	if (not meshfile.is_open()){
	// 		std::cout << "Couldn't write to " << filename << std::endl;
	// 		meshfile.close();
	// 		return;
	// 	}

	// 	//print mesh to file
	// 	vtkprint(meshfile);
	// 	meshfile.close();
	// }

	// void CharmsQ1_3DMesh::vtkprint(std::ostream& os) const
	// {
	// 	//write to buffer and flush buffer to the ostream
	// 	std::stringstream buffer;

	// 	//HEADER
	// 	buffer << "# vtk DataFile Version 2.0\n";
	// 	buffer << "Mesh Data\n";
	// 	buffer << "ASCII\n\n";
	// 	buffer << "DATASET UNSTRUCTURED_GRID\n";

	// 	//POINTS
	// 	buffer << "POINTS " << nNodes() << " float\n";
	// 	for (size_t i=0; i<nNodes(); i++) { buffer << mesh_nodes[i] << "\n";}
	// 	buffer << "\n";
	// 	os << buffer.rdbuf();
	// 	buffer.str("");

	// 	//ELEMENTS
	// 	buffer << "CELLS " << nElems() << " " << (1+8)*nElems() << "\n";
	// 	for (size_t i=0; i<nElems(); i++)
	// 	{
	// 		buffer << 8 << " ";
	// 		for (size_t j=0; j<8; j++)
	// 		{
	// 			buffer << active_elements[i].global_nodes[j] << " ";
	// 		}
	// 		buffer << "\n";
	// 	}
	// 	buffer << "\n";
	// 	os << buffer.rdbuf();
	// 	buffer.str("");

	// 	//VTK IDs
	// 	buffer << "CELL_TYPES " << nElems() << "\n";
	// 	for (size_t i=0; i<nElems(); i++) {buffer << active_elements[i]->vtk_id << " ";}
	// 	buffer << "\n\n";
	// 	os << buffer.rdbuf();
	// 	buffer.str("");

	// 	//ACTIVE BASIS FUNCTIONS
	// 	int basis_depth[nBasis()];
	// 	std::fill_n(basis_depth, nBasis(), -1);
	// 	for (size_t idx=0; idx<nBasis(); idx++)
	// 	{
	// 		size_t node_idx = mesh_nodes.find(active_basis[idx]->coord);
	// 		if (node_idx != (size_t) -1)
	// 		{
	// 			basis_depth[node_idx] = active_basis[idx]->depth;
	// 		}
	// 	}

	// 	buffer << "POINT_DATA " << nNodes() << "\n";
	// 	buffer << "SCALARS depth int\n";
	// 	buffer << "LOOKUP_TABLE default\n";
	// 	for (size_t i=0; i<nNodes(); i++) {buffer << basis_depth[i] << " ";}
	// 	buffer << "\n\n";
	// 	os << buffer.rdbuf();
	// 	buffer.str("");
	// }


}