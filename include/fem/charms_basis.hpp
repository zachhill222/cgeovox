#pragma once



//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include <vector>
#include <cassert>

namespace gv::fem
{
	///base class. stores general information. evaluation must be implemented in the element type.
	template<int dim, typename Element_t>
	class CharmsBasisFun
	{
	public:
		using Point_t = gv::util::Point<dim,double>;

		CharmsBasisFun(const Point_t& coord, const int depth) : coord(coord), depth(depth) {}
		
		const Point_t coord; //coordinate of vertex associated with this basis function
		const int depth; //depth that this basis function occurs at in the mesh (i.e., refinement level)
		bool is_active = false; //create all possible basis functions when dividing elements. only act

		//logic for determining if two basis functions are equal. There can only be one basis function at any vertex.
		//templated to avoid re-implementing it in every class that inherits from this one.
		template<typename Basis_t>
		bool operator==(const Basis_t& other) const {return coord==other.coord;}

		///evaluation shortcut. the following methods must exist:
		///bool Element_t::contains(const gv::util::Point<dim,double>&) const 
		///double Element_t::evaluate(const gv::util::Point<dim,double>&) const 
		double evaluate(const Point_t& location) const
		{
			assert(support.size()==local_node_numbers.size());

			//basis functions must be continuous accross elements, so it is ok to evaluate at the first element.
			for (size_t el_idx=0; el_idx<support.size(); el_idx++)
			{
				if (support[el_idx]->contains(location)) {return support[el_idx]->evaluate(location, local_node_numbers[el_idx]);}
			}

			//if no element contains the specified point, then it is not in the support of this basis function and evaluates to 0.
			return 0.0;
		}

		///gradient evaluation shortcut. the following methods must exist:
		///bool Element_t::contains(const gv::util::Point<dim,double>&) const 
		///gv::util::Point<dim,double> Element_t::evaluate_grad(const gv::util::Point<dim,double>&) const 
		double evaluate_grad(const Point_t& location) const
		{
			assert(support.size()==local_node_numbers.size());

			//gradients of basis functions need not be continuous accross elements, so the location cannot be on the boundary of
			//two different elements.
			int n_elem_contains = 0;
			for (size_t el_idx=0; el_idx<support.size(); el_idx++)
			{
				if (support[el_idx]->contains(location)) {n_elem_contains+=1;}
			}
			assert(n_elem_contains<2);

			//evaluate the gradient. could be faster, but calling Element_t::contains() should be fast and never a bottleneck.
			for (size_t el_idx=0; el_idx<support.size(); el_idx++)
			{
				if (support[el_idx]->contains(location))
				{
					return support[el_idx]->evaluate_grad(location, local_node_numbers[el_idx]);
				}
			}

			//if no element contains the specified point, then it is not in the support of this basis function and evaluates to 0.
			return Point_t(0.0);
		}


	///method to add elements to the support. adds some safety with the pointers and ensures each support element has a
	///corresponding local node number.
	void add_support_element(const Element_t* elem, const size_t node_number)
	{
		support.push_back(elem);
		local_node_numbers.push_back(node_number);
	}


	private:
		///track natural support elements and local node numbers. the depth of these elements and the current basis function
		///in the hierarchy are equal. This basis function corresponds to a local node number on each element, which must be tracked.
		std::vector<const Element_t*> support;
		std::vector<size_t> local_node_numbers;
	};


	///octree storage container for basis functions
	template <int dim, typename Basis_t>
	class CharmsBasisFunOctree : public gv::util::BasicOctree<Basis_t, dim, false, 32>
	{
	public:
		CharmsBasisFunOctree() : gv::util::BasicOctree<Basis_t, dim, false, 32>() {}
		CharmsBasisFunOctree(const gv::util::Box<dim>& bbox) : gv::util::BasicOctree<Basis_t, dim, false, 32>(bbox) {}
	private:
		bool is_data_valid(const gv::util::Box<dim>& box, const Basis_t& data) const override {return box.contains(data.coord);}
	};
}