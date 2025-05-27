#pragma once

//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder

#include "fem/charms_basis.hpp"
#include "fem/charms_element.hpp"

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cassert>

namespace gv::fem
{
	template<int dim>
	struct Voxel_t
	{
		gv::util::Box<dim>* domain = nullptr; //bounding box for the domain that is meshed. needed to get box for this voxel.
		int depth = -1;
		
	}














	//// Q1 ELEMENTS IN 3D ////
	// class Q1_3D_element;
	// using Q1_3D_basis = CharmsBasisFun<3,Q1_3D_element>;
	
	// using Q1_3D_element_list = CharmsElementOctree<3,Q1_3D_element>; //to be used in the mesh class
	// using Q1_3D_basis_list = CharmsBasisFunOctree<3,Q1_3D_basis>; //to be used in the mesh class

	// class Q1_3D_element : public CharmsElement<3,8,8,Q1_3D_basis>
	// {
	// public:
	// 	using Basis_t = Q1_3D_basis;

	// 	Q1_3D_element(const Box_t& box) : CharmsElement<3,8,8,Q1_3D_basis>(), parent(nullptr), bbox(box) {} //for constructing the root element
	// 	Q1_3D_element(Q1_3D_element* const parent, int idx) : CharmsElement<3,8,8,Q1_3D_basis>(parent),
	// 		parent(parent), bbox(parent->bbox.voxelvertex(idx), parent->bbox.center()) {}
	// 	~Q1_3D_element()
	// 	{
	// 		if (is_divided)
	// 		{
	// 			for (int i=0; i<n_children; i++) {delete children[i];}
	// 		}
	// 		delete this;
	// 	}

	// 	//tree structure
	// 	Q1_3D_element* children[n_children] {nullptr};
	// 	Q1_3D_element* const parent;
	// 	void divide();

	// 	//physical location information
	// 	const Box_t bbox; //convenient access instead of getting information from global_nodes[0] and global_nodes[7] from an external vertex list
	// 	inline static const Box_t reference_element{-1,1}; //for evaluating basis functions
		
	// 	//mesh information
	// 	size_t global_nodes[n_nodes] {(size_t) -1};
	// 	static const int vtk_id = 11; //Q1 voxel

	// 	//geometry testing
	// 	bool contains(const Point_t& location) const override {return bbox.contains(location);}
	// 	bool intersects(const Box_t& box) const override {return bbox.intersects(box);}

	// 	//basis function evaluation
	// 	double evaluate(const Point_t& location, int node) const override;
	// 	Point_t evaluate_grad(const Point_t& location, int node) const override;

	// 	void activate(Basis_t* basis)
	// 	{
	// 		assert(contains(basis->coord));
	// 		assert(depth == basis->depth);

	// 		//find local node number
	// 		for (size_t n=0; n<n_nodes; n++)
	// 		{
	// 			if (basis->coord == bbox.voxelvertex(n))
	// 			{
	// 				basis->add_support_element(this,n);
	// 				basis_same.push_back(basis);
	// 				break;
	// 			}
	// 		}
	// 	}
	// };


	// ///evaluate basis functions for Q1 voxel elements
	// double Q1_3D_element::evaluate(const Q1_3D_element::Point_t& location, int node) const
	// {
	// 	Point_t local_coord = (location-bbox.center())/bbox.sidelength();
	// 	Point_t coefs       = reference_element.voxelvertex(node);
	// 	double val = 0.125;
	// 	for (int i=0; i<3; i++) {val*=(1.0+coefs[i]*local_coord[i]);}
	// 	return val;
	// }

	// ///evaluate gradients of basis functions for Q1 voxel elements
	// Q1_3D_element::Point_t Q1_3D_element::evaluate_grad(const Q1_3D_element::Point_t& location, int node) const
	// {
	// 	Point_t local_coord = (location-bbox.center())/bbox.sidelength();
	// 	Point_t coefs       = reference_element.voxelvertex(node);
	// 	Point_t value       = 0.125 * coefs/bbox.sidelength();
	// 	for (int i=0; i<3; i++)
	// 	{
	// 		if (i==node) {continue;}
	// 		value *= (1.0+coefs[i]*local_coord[i]);
	// 	}
	// 	return value;
	// }

	// ///divide Q1 element
	// void Q1_3D_element::divide()
	// {
	// 	assert(!is_divided);
	// 	//create children (constructor copies basis functions and creates bounding boxes)
	// 	for (int i=0; i<n_children; i++) {children[i] = new Q1_3D_element(this, i);}
	// }

	


	// ////////// Logic for handling all elements ///////////////
	// class CharmsQ1_3D_active_elements
	// {
	// public:
		
	// }



























	///print Q1 element
	// std::ostream& operator<<(std::ostream& os, const Q1_3D_element& elem)
	// {
	// 	std::string pad = "";
	// 	for (int k=0; k<elem.depth; k++)
	// 	{
	// 		pad += "    ";
	// 		os  << "----";
	// 	}
		
	// 	os << "|depth= " << elem.depth << "\n";
	// 	os << pad << "|is_divided= " << elem.is_divided << "\n";
	// 	os << pad << "|bbox= " << elem.bbox << "\n";

	// 	os << pad << "|global_nodes= [" << elem.global_nodes[0];
	// 	for (int i=1; i<elem.n_nodes; i++) {os << ", " << elem.global_nodes[i];}
	// 	os << "]\n";

	// 	os << pad << "|basis_ancestor= " << elem.basis_ancestor << "\n";
	// 	os << pad << "|basis_same= " << elem.basis_same << "\n";
	// 	return os << "\n";
	// }
}