//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder



#pragma once

#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/point_octree.hpp"

#include <vector>
#include <cassert>

namespace gv::fem
{
	//classes defined in this file
	template<gv::util::PointOctree<3>* vertices>
	class ElementQ1;

	template<gv::util::PointOctree<3>* vertices>
	class BasisFunctionQ1;

	template<gv::util::PointOctree<3>* vertices>
	class CharmsQ1Mesh;


	//////////////// ELEMENT IMPLEMENTATION ////////////////
	template<gv::util::PointOctree<3>* vertices>
	class ElementQ1
	{
	public:
		using Index_t = gv::util::Point<3,size_t>;
		using Point_t = gv::util::Point<3,double>;

		ElementQ1(const int depth, const Index_t& index, const ElementQ1<vertices>* root) : depth(depth), index(index), root(root)
		{
			//get bounding box for this element
			gv::util::Point<3,double> H = std::pow(0.5,depth) * root->sidelength();
			gv::util::Point<3,double> low = root->low() + H * gv::util::Point<3,double>(index);
			gv::util::Box<3> bbox(low, low+H);
			
			//convert index triplet to vertices and find their indices. add to vertices if necessary.
			for (int k=0; k<8; k++)
			{
				gv::util::Point<3,double> vertex = bbox.voxelvertex(k);
				size_t idx;
				int flag = *vertices.push_back(vertex, idx);
				assert(flag!=-1); //ensure node was added succesfully
				nodes[k] = idx;
			}
		}

		const int depth; //number of divisions of the root element required to reach this element
		const Index_t index; //index to easily refer to elements which may or may not have been instantiated yet
		const ElementQ1<vertices>* root; //element in the coarsest mesh that this element is a subdivision of

		size_t nodes[8]; //mesh information. indices of nodes that make this element (vtk voxel ordering)
		static const vtk_id = 11; //all elements are vtk voxels
		bool is_active = false;
		std::vector<size_t> basis_a; //indices of active ancestor basis functions whose support overlaps with this element
		std::vector<size_t> basis_s; //indices of active same-level basis functions whose support overlaps with this element

		gv::util::Point<3,double> centroid() const {return 0.5*( (*vertices)[0] + (*vertices)[7]);} //return centroid of this element, used for octree storage
	};



	//////////////// BASIS FUNCTION IMPLEMENTATION ////////////////
	template<gv::util::PointOctree<3>* vertices>
	class BasisFunctionQ1;


}