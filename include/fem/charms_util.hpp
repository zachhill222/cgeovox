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
#include <algorithm>
// #include <iterator>
#include <cassert>

#include "util/box.hpp"
#include "util/point.hpp"

namespace gv::fem
{
	///Data type to store elements in an octree structure.
	template<int dim>
	class CharmsElement
	{
	public:
		CharmsElement(const gv::util::Box<dim>& bbox) : parent(nullptr), depth(0), bbox(bbox), total_descendents(0) {}

		CharmsElement(CharmsElement<dim>* parent, const int sibling_number) :
			parent(parent),
			depth(parent->depth+1),
			sibling_number(sibling_number), 
			bbox(parent->bbox.center(), parent->bbox[sibling_number])
		{
			assert(parent!=nullptr);
			assert(!parent->is_divided);
			assert(sibling_number>-1);

			//add parent basis functions to ancestor list
			std::merge(parent->basis_a.begin(), parent->basis_a.end(),
				parent->basis_s.begin(), parent->basis_s.end(),
				std::inserter(this->basis_a, this->basis_a.begin()));
		}

		~CharmsElement()
		{
			if (is_divided)
			{
				for (int k=0; k<max_children; k++) {delete children[k];}
			}

			if (parent!=nullptr) {parent->increment_descendents(-max_children);}

			delete this;
		}

		//basis function tracking.
		std::set<size_t> basis_a; //basis functions on coarser levels
		std::set<size_t> basis_s; //basis functions on same level

		//method to test if the element contains a given point.
		virtual bool contains(const gv::util::Point<dim,double>& point)  {return bbox.contains(point);}

		//method to test if the element intersects a bounding box (used in is_data_valid method for octree)
		virtual bool intersects(const gv::util::Box<dim> &box)  {return bbox.intersects(box);}

		//tree structure
		static const int max_children = std::pow(2,dim);
		CharmsElement<dim>* children[max_children] {nullptr};
		CharmsElement<dim>* parent;
		const int depth;
		bool is_divided = false;
		int sibling_number = -1;
		gv::util::Box<dim> bbox; //extent of the element

		//tree traversal logic
		size_t total_descendents = 0;

		//increment total number of descendents
		void increment_descendents(const int incr)
		{
			total_descendents += incr;
			assert(total_descendents>=0);
			if (parent!=nullptr) {parent->increment_descendents(incr);}
		}
	};








	///Octree structure for elements
	template<int dim>
	class CharmsActiveElements
	{
	public:
		CharmsActiveElements() : root(new Element_t(gv::util::Box<dim>(0.0, 1.0))) {}
		CharmsActiveElements(const gv::util::Box<dim>& bbox) : root(new Element_t(bbox)) {}

		const gv::util::Box<dim>& bbox() const {return root->bbox;}
		std::ostream& str(std::ostream& os) const {return recursive_str(os, root);}
		void refine_at(const gv::util::Point<dim,double>& point) {divide(get_leaf(point));}

	private:
		using Element_t = CharmsElement<dim>;
		Element_t* root;

		//division
		void divide(Element_t* elem)
		{
			assert(elem!=nullptr);
			assert(!elem->is_divided);
			for (int k=0; k<elem->max_children; k++)  {elem->children[k] = new Element_t(elem, k);}
			
			elem->increment_descendents(elem->max_children);
			elem->is_divided = true;
		}

		//access elements
		Element_t* get_leaf(const gv::util::Point<dim,double>& point)
		{
			return recursive_get_leaf(point, root);
		}

		Element_t* recursive_get_leaf(const gv::util::Point<dim,double>& point, Element_t* elem)
		{
			assert(elem->bbox.contains(point));
			if (elem->is_divided)
			{
				for (int k=0; k<elem->max_children; k++)
				{
					if (elem->children[k]->bbox.contains(point)) {return recursive_get_leaf(point, elem->children[k]);}
				}
			}
			return elem;
		}

		//print and debug
		std::ostream& recursive_str(std::ostream& os, Element_t* elem) const
		{
			assert(elem!=nullptr);
			os << *elem;
			if (elem->is_divided)
			{
				for (int k=0; k<elem->max_children; k++)  {recursive_str(os, elem->children[k]);}
			}
			return os;
		}
	};


	///Basis function class for CHARMS method
	// template <int dim>
	// class CharmsBasisFun
	// {
	// private:
	// 	const ActiveElements_t& active_element_list;

	// public:
	// 	CharmsBasisFun(const ActiveElements_t& elements) : active_element_list(elements) {}

	// 	std::vector<size_t> support; //elements which intersect the support of this basis function, on same level only
	// 	std::vector<int> local_node_number; //which local basis function this function corresponds to on each element
	// 	int depth;

	// 	///evaluate the basis function at the specified point
	// 	double evaluate(const gv::util::Point<dim,double>& point) const
	// 	{
	// 		assert(support.size()>0);

	// 		for (size_t k=0; k<support.size(); k++)
	// 		{
	// 			//return result on first element that work. basis functions must be continuous.
	// 			if (active_element_list[support[k]].contains(point))
	// 			{
	// 				return active_element_list[support[k]].evaluate(point, local_node_number[k]);
	// 			}
	// 		}
	// 		return 0.0;
	// 	};

	// 	///evaluate the gradient of the basis function at the specified point
	// 	gv::util::Point<dim,double> gradient(const gv::util::Point<dim,double>& point) const
	// 	{
	// 		assert(support.size()>0);

	// 		for (size_t k=0; k<support.size(); k++)
	// 		{
	// 			//return result on first element that work. gradients are not necessarily continuous.
	// 			//this method should only be called for points that are interior to a support element.
	// 			if (active_element_list[support[k]].contains(point))
	// 			{
	// 				return active_element_list[support[k]].gradient(point, local_node_number[k]);
	// 			}
	// 		}
	// 		return gv::util::Point<dim,double>(); //default constructor is all zeros
	// 	}
	// };




	//debug
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::set<T>& s)
	{
		os << "{";
		bool o{};
		for (const auto&e : s)
			os << (o ? ", " : (o=1, " ")) << e;
		return os << " } [" << s.size() << "]";
	}

	template<int dim>
	std::ostream& operator<<(std::ostream& os, const CharmsElement<dim>& elem)
	{
		std::string pad = "";
		for (int k=0; k<elem.depth; k++)
		{
			pad += "    ";
			os  << "----";
		}
		
		os << "|depth= " << elem.depth << "\n";
		os << pad << "|total_descendents= " << elem.total_descendents << "\n";
		os << pad << "|sibling_number= " << elem.sibling_number << "\n";
		os << pad << "|bbox= " << elem.bbox << "\n";
		os << pad << "|is_divided= " << elem.is_divided << "\n";
		os << pad << "|basis_a= " << elem.basis_a << "\n";
		os << pad << "|basis_s= " << elem.basis_s << "\n";
		return os << "\n";
	}

	template<int dim>
	std::ostream& operator<<(std::ostream& os, const CharmsActiveElements<dim>& elem_octree)
	{
		return elem_octree.str(os);
	}
}