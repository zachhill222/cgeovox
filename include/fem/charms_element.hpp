#pragma once



//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder

#include "fem/charms_basis.hpp"
#include "util/points.hpp"
#include "util/box.hpp"

#include <vector>
#include <array>
#include <functional>
#include <cassert>

namespace gv::fem
{	
	////Base class
	template<int dim, int nodes_per_element, int max_children>
	class CharmsElement
	{
		using Point_t = gv::util::Point<dim,double>;
		using Box_t = gv::util::Point<dim,double>;


		CharmsElement() : depth(0) {}
		template<typename Element_t> //Element_t must be a derived class
		CharmsElement(const Element_t* const parent) : depth(1+parent->depth)
		{

			//pass basis functions from parent to this element
			basis_ancestor.reserve(parent->basis_ancestor.size()+parent->basis_same.size());
			for (size_t idx=0; idx<parent->basis_ancestor.size(); idx++)
			{
				this->basis_ancestor.push_back(parent->basis_ancestor[idx]);
			}
			for (size_t idx=0; idx<parent.basis_same->size(); idx++)
			{
				this->basis_ancestor.push_back(parent->basis_same[idx]);
			}
		}

		//constant information
		static const int n_nodes = nodes_per_element;
		static const int n_children = max_children;

		//refinement level
		const int depth;

		//check if this element is divided or not
		bool is_divided = false;

		//basis function tracking. the two basis sets are always disjoint
		std::vector<size_t> basis_ancestor; //track basis functions in coarser levels that this element is contained in the support of
		std::vector<size_t> basis_same; //track the basis function on the same level that this element is in the natural support of

		//methods that must be overridden in each derived element class
		virtual bool contains(const Point_t& location) const {return false;} //determine if the point is contained in this element
		virtual double evaluate(const Point_t& location, int node) const {return 0.0;} //evaluate basis function (assume location contained)
		virtual Point_t evaluate_grad(const Point_t& location, int node) const {return Point_t(0.0);} //evaluate gradient of specified basis function (assume location is an interior point)
		virtual intersects(const Box_t& box) const {return false;} //check if this element intersects a box. used for octree storage.
	};

	template<typename Element_t>
	class CharmsElementOctree : public gv::util::BasicOctree<Element_t, dim, true, 16>
	{
	public:
		CharmsElementOctree() : gv::util::BasicOctree<Element_t, dim, true, 16>() {}
		CharmsElementOctree(const gv::util::Box<dim>& bbox) : gv::util::BasicOctree<Element_t, dim, true, 16>(bbox) {}
	private:
		bool is_data_valid(const gv::util::Box<dim>& box, const Element_t& data) const override {return data.intersects(box);}
	};



	//// Q1 ELEMENTS IN 3D ////
	class Q1_3D_element;
	using Q1_3D_basis = CharmsBasisFun<3,Q1_3D_element>;

	class Q1_3D_element : public CharmsElement<3, 8, 8>
	{
		Q1_3D_element(const Box_t& box) : CharmsElement<3,8,8>(0), bbox(box) {} //for constructing the root element
		Q1_3D_element(const Q1_3D_element* const parent, int idx) : CharmsElement<3,8,8>(parent),
			parent(parent), bbox(parent->bbox.voxel_index(idx), parent->bbox.center()) {}

		Q1_3D_element* children[n_children] {nullptr};
		Q1_3D_element* const parent;

		const Box_t bbox;

		bool contains(const Point_t& location) const override {return bbox.contains(location);}
		bool intersects(const Box_t& box) const override {return bbox.intersects(box);}


	};
}