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
#include <cassert>

#include "util/box.hpp"
#include "util/point.hpp"
#include "util/octree.hpp"

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
	template<int dim, typedef Basis_t>
	class CharmsMesh
	{
	public:
		using Element_t = CharmsElement<dim>;

		CharmsMesh() : root(new Element_t(gv::util::Box<dim>(0.0, 1.0))), _basis_funtions(gv::util::Box<dim>(0.0, 1.0))
		{
			_elements.push_back(root);
		}
		CharmsMesh(const gv::util::Box<dim>& bbox) : root(new Element_t(bbox)), _basis_funtions(bbox) {_elements.push_back(root);}

		const gv::util::Box<dim>& bbox() const {return root->bbox;}
		std::ostream& str(std::ostream& os) const {return recursive_str(os, root);}
		void refine_at(const gv::util::Point<dim,double>& point) {divide(get_leaf(point));}

		//access elements
		const Element_t& operator[](const size_t idx) const {return *_elements[idx];}
		Element_t& operator[](const size_t idx) {return *_elements[idx];}
		size_t size() const {assert(_elements.size()==root->total_descendents+1); return _elements.size();}

	private:
		//data
		Element_t* root;
		std::vector<Element_t*> _elements;
		CharmsBasisOctree<dim, Basis_t> _basis_funtions;

		//division
		void divide(Element_t* elem)
		{
			assert(elem!=nullptr);
			assert(!elem->is_divided);
		
			//create children
			for (int k=0; k<elem->max_children; k++)
			{
				elem->children[k] = new Element_t(elem, k);
				_elements.push_back(elem->children[k]);
			}
			
			//update element and ancestors
			elem->increment_descendents(elem->max_children);
			elem->is_divided = true;
		}

		//get first leaf that contains the specified point
		Element_t* get_leaf(const gv::util::Point<dim,double>& point) {return recursive_get_leaf(point, root);}
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

		//get all elements that contain the specified point
		std::vector<Element_t*> get_elements(const gv::util::Point<dim,double>& point)
		{
			std::vector<Element_t*> el_list;
			recursive_get_elements(point, el_list, root);
			return el_list;
		}
		void recursive_get_elements(const gv::util::Point<dim,double>& point, std::vector<Element_t*>& el_list, Element_t* elem)
		{
			assert(elem!=nullptr);
			assert(elem->bbox.contains(point));

			//current node belongs in the list
			el_list.push_back(elem);

			//check if children belong in the list
			if (elem->is_divided)
			{
				for (int k=0; k<elem->max_children; k++)
				{
					if (elem->children[k]->bbox.contains(point)) {recursive_get_elements(point, el_list, elem->children[k]);}
				}
			}
		}

		//get all elements that contain the specified point at the specified depth
		std::vector<Element_t*> get_elements_at_depth(const gv::util::Point<dim,double>& point, const int depth)
		{
			std::vector<Element_t*> el_list;
			recurseve_get_elements_at_depth(point, depth, el_list, root);
			return el_list;
		}
		void recurseve_get_elements_at_depth(const gv::util::Point<dim,double>& point, const int depth, std::vector<Element_t*>& el_list, Element_t* elem)
		{
			assert(elem!=nullptr);
			assert(elem->bbox.contains(point));
			assert(elem->depth <= depth);

			//if the current element belongs in the list, then its children do not
			if (elem->depth == depth)
			{
				el_list.push_back(elem);
				return;
			}

			//check if its children belong in the list
			if (elem->is_divided)
			{
				for (int k=0; k<elem->max_children; k++)
				{
					if (elem->children[k]->bbox.contains(point)) {recursive_get_elements(point, depth, el_list, elem->children[k]);}
				}
			}
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


	///Basis function class for CHARMS method.
	template <int dim>
	class CharmsBasisFun
	{
	private:
		using Element_t = CharmsElement<dim>;
		static constexpr const gv::util::Box<dim> reference_box(-1,1);

	public:
		CharmsBasisFun(const gv::util::Point<dim,double>& coord, int depth) : coord(coord), depth(depth) {}
		gv::util::Point<dim,double> coord;
		int depth;
		
		std::vector<Element_t*> support;
		std::vector<int> local_node; //track which node this basis function corresponds to on each support element
	};


	//allow basis functions to be stored in an octree structure
	template <int dim, typedef Basis_t>
	class CharmsBasisOctree : public gv::util::BasicOctree<Basis_t, dim, false, 32>
	{
	public:
		CharmsBasisOctree() : gv::util::BasicOctree<Basis_t, dim, false, 32>() {}
		CharmsBasisOctree(const gv::util::Box<dim>& bbox) : gv::util::BasicOctree<Basis_t, dim, false, 32>(bbox) {}
	private:
		bool is_data_valid(const gv::util::Box<dim>& box, const Basis_t& data) const override {return box.contains(data.coord);}
	}





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