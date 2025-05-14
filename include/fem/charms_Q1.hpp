#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"

#include "fem/charms_util.hpp"

namespace gv::fem
{
	template <int dim>
	class CharmsQ1Element : public CharmsElement<dim>
	{
	private:
		typedef gv::util::Point<dim,double> Point_t;
		typedef gv::util::Box<dim> Box_t;

		static constexpr const int nodes_per_element = std::pow(2,dim);
		static constexpr const gv::util::Box<dim> reference_box; //default is [-1,1]^dim

		///access list of global coordinates
		const gv::util::Box<dim> bbox;
	public:
		CharmsQ1Element(const Box_t& bbox) : bbox(bbox), CharmsElement<dim>()
		{
			if (dim==2) {vtk_id=8;}
			if (dim==3) {vtk_id=11;}
		}
		
		size_t _nodes_ref[nodes_per_element]; //used for writing the mesh to a file
		int vtk_id; //used for writing the mesh to a file.

		///evaluate the specified basis function
		double evaluate(const Point_t& point, const int local_node) const
		{
			Point_t local_coord     = (point-bbox.center())/bbox.sidelength();
			Point_t basis_fun_coefs = reference_box.voxelvertex(local_node);

			double value = 0.125;
			for (int k=0; k<dim; k++)  {value *= (1.0+basis_fun_coefs[k]*local_coord[k]);}
			return value;
		}


		///evaluate the gradient of the specified basis function
		Point_t gradient(const Point_t& point, const int local_node) const
		{
			Point_t local_coord     = (point-bbox.center())/bbox.sidelength();
			Point_t basis_fun_coefs = reference_box.voxelvertex(local_node);

			Point_t value = 0.125 * basis_fun_coefs / bbox.sidelength();
			for (int k=0; k<dim; k++)
			{
				if (k==local_node) {continue;}
				value *= (1.0+basis_fun_coefs[k]*local_coord[k]);
			}
			return value;
		}
	};








	template <int dim>
	class CharmsQ1
	{
	protected:
		typedef gv::util::Point<dim,double> Point_t;
		typedef gv::util::Box<dim> Box_t;
		typedef CharmsQ1Element<dim> Element_t;
		typedef CharmsBasisFun<gv::fem::CharmsElementOctree<Element_t, dim>> Basis_t; //each basis stores a reference into the element list
		
		///meshed region
		Box_t bbox(Point_t(-1.0),Point_t(1.0));
		
		///list of active elements
		gv::fem::CharmsElementOctree<Element_t, dim> _elements;
		
		///if implementing an un-refine method, this will need to change
		///list of active basis functions
		std::vector<Basis_t> _basis;

		///list of nodes used in the mesh
		gv::util::PointOctree<3,32> _nodes;

	public:
		CharmsQ1()
		{
			//construct initial element, basis functions, and mesh nodes
			Element_t elem(bbox);
			for (int k=0; k<8; k++)
			{
				//add current node
				_nodes.push_back(bbox.voxelvertex(k));

				//put node reference into element in correct order
				elem._nodes_ref[k] = k;

				//construct and add basis function for level 0 at the current node
				Basis_t basis_fun(_elements);
				basis_fun.support.push_back(0);
				basis_fun.depth = 0;
				basis_fun.local_node_number = k;
				_basis.push_back(basis_fun);

				//add current basis function to element
				elem.basis_s.push_back(k);
			}

			//add element to octree
			_elements.push_back(elem);
		}
		

	}




}