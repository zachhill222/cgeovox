#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "util/octree.hpp"
#include "util/octree_util.hpp"


namespace gv::mesh
{
	

	//each element is a node in the octree
	//when an element is split, no basis functions are removed
	//when an element is split, new (mesh) nodes/points define new basis functions (detail functions)
	class H_Element
	{
	protected:
		const gv::util::OctreePath<3> path;
		bool is_refined = false;

	public:
		H_Element() {};
		H_Element(const gv::util::OctreePath<3> &path) : path(path) {}

		///global indices for active basis functions
		std::vector<size_t> active_basis_functions;
	}


	struct BasisFunction
	{
		std::vector<size_t> element_idx; //similar to node2elem in a standard mesh
		
		//information to get coordinate of basis function "center" (node coordinate in the mesh)
		size_t depth=0;
		size_t ijk[3] {0}; //coordinate[n] = bbox.low()[n] + (0.5^depth) * ijk[n] * H[n] where H[n] is the width of the bounding box and bbox.low() is the origin

		void set_ijk()
	}


	class HierarchicalVoxelQ1
	{
	protected:
		typedef Point_t gv::util::Point<3,double>;
		typedef Box_t gv::util::Box<3>;

		///meshed region
		Box_t bbox(Point_t(-1,-1,-1),Point_t(1,1,1));
		///list of active elements
		std::vector<H_Element> _elements;
		///list of active basis functions
		std::vector<BasisFunction> _basis_functions;

	public:
		HierarchicalVoxelQ1()
		{
			//construct first element (root)
			H_Element first_element;
			_elements.push_back(first_element);

			//construct first basis functions
			BasisFunction first_basis_functions[8];
		}
		HierarchicalVoxelQ1(const Box_t &box) : bbox(box) {HierarchicalVoxelQ1();}

	}




}