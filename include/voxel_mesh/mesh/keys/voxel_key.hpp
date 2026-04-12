#pragma once

#include <cstdint>
#include <type_traits>

#include "voxel_mesh/mesh/keys/voxel_key_base.hpp"
#include "voxel_mesh/mesh/keys/voxel_key_element.hpp"
#include "voxel_mesh/mesh/keys/voxel_key_vertex.hpp"
#include "voxel_mesh/mesh/keys/voxel_key_face.hpp"

namespace gv::vmesh
{
	//concepts
	template<typename T>
	concept VoxelElementKeyType = std::same_as<T, VoxelElementKey<T::I_W, T::MORTON>>;
	
	static_assert(VoxelElementKeyType<VoxelElementKey<>>);
	static_assert(!VoxelElementKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelElementKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelVertexKeyType = std::same_as<T, VoxelVertexKey<T::I_W, T::MORTON>>;
	static_assert(!VoxelVertexKeyType<VoxelElementKey<>>);
	static_assert(VoxelVertexKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelVertexKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelFaceKeyType = std::same_as<T, VoxelFaceKey<T::I_W, T::MORTON>>;
	static_assert(!VoxelFaceKeyType<VoxelElementKey<>>);
	static_assert(!VoxelFaceKeyType<VoxelVertexKey<>>);
	static_assert(VoxelFaceKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelKeyType = VoxelElementKeyType<T> || VoxelVertexKeyType<T> || VoxelFaceKeyType<T>;
	static_assert(VoxelKeyType<VoxelElementKey<>>);
	static_assert(VoxelKeyType<VoxelVertexKey<>>);
	static_assert(VoxelKeyType<VoxelFaceKey<>>);


	//useful standalone functions
	template<VoxelKeyType Key_t>
	constexpr uint64_t total_possible(const uint64_t max_depth) {
		return Key_t::depth_linear_start(max_depth+1);
	}

	//implement the adacency operations
	template<uint64_t I_W, bool MORTON>
	inline constexpr VoxelVertexKey<I_W,MORTON> VoxelElementKey<I_W,MORTON>::vertex(const bool bi, const bool bj, const bool bk) const
	{
		return VoxelVertexKey<I_W,MORTON>(
			depth(),
			i() + static_cast<uint64_t>(bi),
			j() + static_cast<uint64_t>(bj),
			k() + static_cast<uint64_t>(bk)
		);
	}
	
	template<uint64_t I_W, bool MORTON>
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
	
	template<uint64_t I_W, bool MORTON>
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




	template<uint64_t I_W, bool MORTON>
	constexpr VoxelElementKey<I_W,MORTON> VoxelFaceKey<I_W,MORTON>::element(const bool forward_flag) const
	{
		switch (axis()) {
		case 0: return VoxelElementKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(forward_flag), j(), k()};
		case 1: return VoxelElementKey<I_W,MORTON>{depth(), i(), j()+static_cast<uint64_t>(forward_flag), k()};
		case 2: return VoxelElementKey<I_W,MORTON>{depth(), i(), j(), k()+static_cast<uint64_t>(forward_flag)};
		}
		return VoxelElementKey<I_W,MORTON>{DOES_NOT_EXIST};
	}

	template<uint64_t I_W, bool MORTON>
	constexpr VoxelVertexKey<I_W,MORTON> VoxelFaceKey<I_W,MORTON>::vertex(const bool bi, const bool bj) const
	{
		switch (axis()) {
		case 0: return VoxelVertexKey<I_W,MORTON>{depth(), i(), j()+static_cast<uint64_t>(bi), k()+static_cast<uint64_t>(bj)};
		case 1: return VoxelVertexKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(bi), j(), k()+static_cast<uint64_t>(bj)};
		case 2: return VoxelVertexKey<I_W,MORTON>{depth(), i()+static_cast<uint64_t>(bi), j()+static_cast<uint64_t>(bj), k()};
		}
		return VoxelVertexKey<I_W,MORTON>{DOES_NOT_EXIST};
	}

	template<uint64_t I_W, bool MORTON>
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

	template<uint64_t I_W, bool MORTON>
	constexpr VoxelElementKey<I_W, MORTON> VoxelVertexKey<I_W,MORTON>::element(const bool bi, const bool bj, const bool bk) const
	{
		//we must subtract vertex at (1,1,1) has element at (0,0,0) for its first element
		const uint64_t ii = i() - static_cast<uint64_t>(bi);
		const uint64_t jj = j() - static_cast<uint64_t>(bj);
		const uint64_t kk = k() - static_cast<uint64_t>(bk);
		const uint64_t mi = VoxelElementKey<I_W,MORTON>::MAX_ELEMENT_INDEX;
		if (ii>mi || jj>mi || kk>mi) {return VoxelElementKey<I_W,MORTON>{DOES_NOT_EXIST};}
		return VoxelElementKey<I_W,MORTON>{depth(),ii,jj,kk};
	}

	template<uint64_t I_W, bool MORTON>
	constexpr VoxelElementKey<I_W, MORTON> VoxelVertexKey<I_W,MORTON>::element(const int en) const {
		switch (en) {
		case 0: return element(0,0,0);
		case 1: return element(1,0,0);
		case 2: return element(0,1,0);
		case 3: return element(1,1,0);
		case 4: return element(0,0,1);
		case 5: return element(1,0,1);
		case 6: return element(0,1,1);
		case 7: return element(1,1,1);
		default: return VoxelElementKey<I_W, MORTON>{DOES_NOT_EXIST};
		}
	}


}