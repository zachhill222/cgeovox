#pragma once

#include "gutil.hpp"
#include "mesh/voxel_mesh_keys.hpp"


namespace gv::mesh
{
	//This class is for a hierarchical voxel mesh.
	//This structure allows us to very efficiently store elements, vertices, faces, etc.
	//The base mesh is 1x1x1 so that every vertex is (in reference coordinates) a dyadic rational number
	//Elements, vertices, and faces all have special index/key structs for their storage and logical relations.
	//All information for every element, vertex, and face is compressed into a 64-bit unsigned integer
	
	class HierarchicalVoxelMesh
	{
	protected:
		std::vector<std::vector<VoxelElementKey>> elements_by_depth;
	};


}