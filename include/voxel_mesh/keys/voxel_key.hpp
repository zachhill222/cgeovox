#pragma once

#include <cstdint>

#include "voxel_mesh/keys/voxel_key_base.hpp"
#include "voxel_mesh/keys/voxel_key_element.hpp"
#include "voxel_mesh/keys/voxel_key_vertex.hpp"
#include "voxel_mesh/keys/voxel_key_face.hpp"

namespace gv::vmesh
{
	//implement the adacency operations
	template<int I_W, bool MORTON>
	inline constexpr VoxelVertexKey<I_W,MORTON> VoxelElementKey<I_W,MORTON>::vertex(const bool bi, const bool bj, const bool bk) const
	{
		return VoxelVertexKey<I_W,MORTON>(
			depth(),
			i() + static_cast<uint64_t>(bi),
			j() + static_cast<uint64_t>(bj),
			k() + static_cast<uint64_t>(bk)
		);
	}
	
	template<int I_W, bool MORTON>
	constexpr VoxelVertexKey<I_W,MORTON> VoxelElementKey<I_W,MORTON>::vertex(const int vn) const
	{
		switch (vn) {
		case 0: return vertex(0,0,0);
		case 1: return vertex(1,0,0);
		case 2: return vertex(0,1,0);
		case 3: return vertex(1,1,0);
		case 4: return vertex(0,0,1);
		case 5: return vertex(1,0,1);
		case 6: return vertex(0,1,1);
		case 7: return vertex(1,1,1);
		default: return VoxelVertexKey<I_W,MORTON>{DOES_NOT_EXIST};
		}
	}
	
	template<int I_W, bool MORTON>
	constexpr VoxelFaceKey<I_W,MORTON> VoxelElementKey<I_W,MORTON>::face(const int fn) const
	{
		switch (fn) {
		case 0: return VoxelFaceKey<I_W,MORTON>{0, depth(), i()  , j()  , k()  };
		case 1: return VoxelFaceKey<I_W,MORTON>{1, depth(), i()  , j()  , k()  };
		case 2: return VoxelFaceKey<I_W,MORTON>{2, depth(), i()  , j()  , k()  };
		case 3: return VoxelFaceKey<I_W,MORTON>{0, depth(), i()+1, j()  , k()  };
		case 4: return VoxelFaceKey<I_W,MORTON>{1, depth(), i()  , j()+1, k()  };
		case 5: return VoxelFaceKey<I_W,MORTON>{2, depth(), i()  , j()  , k()+1};
		default: return VoxelFaceKey<I_W,MORTON>{DOES_NOT_EXIST};
		}
	}




	template<int I_W, bool MORTON>
	constexpr VoxelElementKey<I_W,MORTON> VoxelFaceKey<I_W,MORTON>::element(const bool forward_flag) const
	{
		switch (axis()) {
		case 0: return VoxelElementKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(forward_flag), j(), k()};
		case 1: return VoxelElementKey<I_W,MORTON>{depth(), i(), j()+static_cast<uint64_t>(forward_flag), k()};
		case 2: return VoxelElementKey<I_W,MORTON>{depth(), i(), j(), k()+static_cast<uint64_t>(forward_flag)};
		}
		return VoxelElementKey<I_W,MORTON>{DOES_NOT_EXIST};
	}

	template<int I_W, bool MORTON>
	constexpr VoxelVertexKey<I_W,MORTON> VoxelFaceKey<I_W,MORTON>::vertex(const bool bi, const bool bj) const
	{
		switch (axis()) {
		case 0: return VoxelVertexKey<I_W,MORTON>{depth(), i(), j()+static_cast<uint64_t>(bi), k()+static_cast<uint64_t>(bj)};
		case 1: return VoxelVertexKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(bi), j(), k()+static_cast<uint64_t>(bj)};
		case 2: return VoxelVertexKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(bi), j()+static_cast<uint64_t>(bj), k()};
		}
		return VoxelVertexKey<I_W,MORTON>{DOES_NOT_EXIST};
	}

	template<int I_W, bool MORTON>
	constexpr VoxelVertexKey<I_W,MORTON> VoxelFaceKey<I_W,MORTON>::vertex(const int vn) const
	{
		switch (vn) {
		case 0: return vertex(0,0);
		case 1: return vertex(1,0);
		case 2: return vertex(0,1);
		case 3: return vertex(1,1);
		default: return VoxelVertexKey<I_W,MORTON>{DOES_NOT_EXIST};
		}
	}


}