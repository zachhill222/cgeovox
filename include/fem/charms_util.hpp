#pragma once

#include <iostream>
#include <string>

//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder

#include <vector>
#include <set>
#include <cassert>

#include "util/box.hpp"
#include "util/point.hpp"

namespace gv::fem
{
	template<int dim>
	int bool2number(const bool child[dim])
	{
		sibling_number = (int) child[dim-1];
		for (int k=dim-1; k>=0; k--)
		{
			sibling_number += ((int) child[k]) + 2*sibling_number;
		}
		return sibling_number;
	}


	///Data type to store elements in an octree structure.
	template<int dim>
	class CharmsElement
	{
	public:
		CharmsElement(const gv::util::Box<dim>& bbox) : parent(nullptr), depth(0), ID(0), bbox(bbox)
		{
			for (int k=0; k<dim; k++) {idx[k]=0;}
			
			total_descendents = 0;
			index = 0;
		}

		CharmsElement(const CharmsElement* parent, const bool child[dim]) : parent(parent), depth(parent->depth+1) {}
		// {
		// 	//set child number
		// 	sibling_number = bool2number(child);

		// 	//set idx
		// 	for (int k=0; k<dim; k++)
		// 	{
		// 		idx[k] = 2*(parent->idx[k]) + (size_t) child[k];
		// 	}

		// 	//set ID
		// 	ID = (std::pow(2,dim*depth)-1)/(std::pow(2,dim)-1);

		// 	//update parent
		// 	parent->is_divided = false;


		// 	//add parent basis functions to ancestor list
		// 	std::merge(std::begin(parent->basis_a), std::end(parent->basis_a),
		// 		std::begin(parent->basis_s), std::end(parent->basis_s),
		// 		std::back_inserter(this->basis_a))
		// }

		//basis function tracking.
		std::set<size_t> basis_a; //basis functions on coarser levels
		std::set<size_t> basis_s; //basis functions on same level

		//method to test if the element contains a given point.
		virtual bool contains(const gv::util::Point<dim,double>& point)  {return bbox.contains(point);}

		//method to test if the element intersects a bounding box (used in is_data_valid method for octree)
		virtual bool intersects(const gv::util::Box<dim> &box)  {return bbox.insersects(box);}

		//tree structure
		static const uint max_children = std::pow(2,dim);
		CharmsElement* children[max_children] {nullptr};
		const CharmsElement* parent;
		const uint depth; //root is depth 0
		bool is_divided = false;
		int sibling_number = -1;
		size_t idx[dim];
		gv::util::Box<dim> bbox; //extent of the element
		size_t GlobalID; //each possible element has a unique ID. this is independent of the order of refinement.

		//tree traversal logic
		size_t total_descendents;
		size_t index;
	}

	//debug
	template<int dim>
	std::ostream& operator<<(std::ostream& os, const CharmsElement<dim>& elem)
	{
		std::string padding = "";
		for (int k=0; k<depth; k++)
		{
			padding += "  ";
			os      << "--";
		}


		os << padding << "|ID= " << elem.ID << "\n";
		
		os << padding << "|idx= ";
		for (int k=0; k<dim; k++) {os << idx[k] << " ";}
		os << padding << "\n";
		
		os << padding << "|depth= " << elem.depth << "\n";
		os << padding << "|sibling_number= " << elem.sibling_number << "\n";
		os << padding << "|is_divided= " << elem.is_divided << "\n";
		os << padding << "basis_a= " << elem.basis_a << "\n";
		os << padding << "basis_s= " << elem.basis_s << "\n";
	}






	///Octree structure for elements
	template<int dim>
	class CharmsActiveElements
	{
	public:
		CharmsElementOctree() : root(gv::util::Box<dim>(gv::util::Point<dim,double>(0.0), gv::util::Point<dim,double>(1.0))) {}
		CharmsElementOctree(const gv::util::Box<dim> &bbox) : root(bbox) {}

		const gv::util::Box<dim>& bbox() const {return root->bbox;}
	private:
		const CharmsElement* root;

		//setting the flag on each element is useful (e.g. tree traversal)
		void set_all_flags(const int flag) const {recursive_set_all_flags(flag, root);}
		void recursive_set_all_flags(const int flag, CharmsElement* elem) const
		{
			assert(elem!=nullptr);
			elem->flag = flag;

			if (elem->is_divided)
			{
				for (int k=0; k<elem->max_children; k++)
				{
					recursive_set_all_flags(flag, elem->children[k]);
				}
			}
		}

		//increment total number of descendents
		void increment_descendents(const int incr, CharmsElement* elem) const
		{
			assert(elem!=nullptr);
			assert(elem->total_descendents>=-incr);

			elem->total_descendents += incr;
			if (elem->parent!=nullptr) {increment_descendents(incr, elem->parent);}
		}

		//get element by linear index
		CharmsElement* operator[](const size_t idx)
		{
			assert(idx<root->total_descendents);
			return recursive_find_element(idx, root, 0);
		}
		CharmsElement* recursive_find_element(const size_t idx, CharmsElement* elem, size_t accumulator)
		{
			assert(elem!=nullptr);
			if (elem->index == idx) {return elem;}
			for ()
		}
	}


	///Basis function class for CHARMS method
	template<typename ActiveElements_t>
	class CharmsBasisFun
	{
	private:
		const ActiveElements_t& active_element_list;

	public:
		CharmsBasisFun(const ActiveElements_t& elements) : active_element_list(elements) {}

		std::vector<size_t> support; //elements which intersect the support of this basis function, on same level only
		std::vector<int> local_node_number; //which local basis function this function corresponds to on each element
		int depth;

		///evaluate the basis function at the specified point
		double evaluate(const gv::util::Point<dim,double>& point) const
		{
			assert(support.size()>0);

			for (size_t k=0; k<support.size(); k++)
			{
				//return result on first element that work. basis functions must be continuous.
				if (active_element_list[support[k]].contains(point))
				{
					return active_element_list[support[k]].evaluate(point, local_node_number[k]);
				}
			}
			return 0.0;
		};

		///evaluate the gradient of the basis function at the specified point
		gv::util::Point<dim,double> gradient(const gv::util::Point<dim,double>& point) const
		{
			assert(support.size()>0);

			for (size_t k=0; k<support.size(); k++)
			{
				//return result on first element that work. gradients are not necessarily continuous.
				//this method should only be called for points that are interior to a support element.
				if (active_element_list[support[k]].contains(point))
				{
					return active_element_list[support[k]].gradient(point, local_node_number[k]);
				}
			}
			return gv::util::Point<dim,double>(); //default constructor is all zeros
		}
	};
}