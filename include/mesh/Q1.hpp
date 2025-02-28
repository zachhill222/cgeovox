#pragma once

#include "util/octree.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/mesh.hpp"

#include <vector>
#include <algorithm>

#include <sstream>
#include <iostream>
#include <fstream>

namespace gv::mesh
{
	//voxels embedded in 3D with piecewise d-linear shape functions.
	class VoxelMeshQ1 : public Mesh<8> {
	private:
		gv::util::PointOctree<3,double,32> _nodes;
		std::vector<size_t> _elem2node;
		std::vector<size_t> _node2elem_start_idx; //each node belongs to an unknown number of elements. track where node2element begins for each node.
		std::vector<size_t> _node2elem;

	public:
		VoxelMeshQ1() : Mesh<8>() {this->vtkID=11;}
	};
}