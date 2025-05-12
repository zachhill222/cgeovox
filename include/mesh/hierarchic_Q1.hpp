#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/homo_mesh.hpp"
#include "mesh/vtkVoxel.hpp"

#include <vector>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>

namespace gv::mesh
{
	///compact method to refer to an element of the octree without constructing the octree
	struct OctreeElementPath
	{
		static const int MAX_DEPTH = 16;
		int path[MAX_DEPTH] {-1};
		int depth() const
		{
			int result = -1;
			for (int i=0; i<MAX_DEPTH; i++)
			{
				if (path[i]>=0) {result++;}
				else {return result;}
			}
			return result;
		}
	}


	///convert a path to a linear index
	size_t path2lin(const OctreeElementPath &path)
	{
		//get depth
		int depth = path.depth();
		assert(depth>=0, "Uninitialized OctreeElementPath");
		
		size_t elements_per_side = std::pow(2,depth);
		size_t linear_index = (elements_per_side*elements_per_side*elements_per_side-1)/7; //count elements in all lower depths

		linear_index += 


	}


	class H_Element
	{
	public:
		H_Element(const size_t level, const size_t path[level])
	}


	class HierarchicalVoxelQ1
	{
	protected:
		typedef Point_t gv::util::Point<3,double>;
		typedef Box_t gv::util::Box<3>;

		///given the mesh level, ijk index (0 <= i,j,k < 2^level), and vertex number (vtk Voxel order) return the coordinate of the point
		static Point_t idx2pt(const size_t level, const size_t i, const size_t j, const size_t k, const size_t v) const;

		///given the mesh level, ijk index (0 <= i,j,k < 2^level), and vertex number (vtk Voxel order) return the standardized basis number
		static size_t idx2lin(const size_t level, const size_t i, const size_t j, const size_t k, const size_t v) const;

		///meshed region
		Box_t bbox;


	}




}