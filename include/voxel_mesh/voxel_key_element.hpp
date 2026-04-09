#pragma once

#include "voxel_mesh/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>


namespace gv::vmesh
{
	//define the element key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<int I_W>
	struct VoxelVertexKey;

	template<int I_W>
	struct VoxelFaceKey;

	template<int I_W=16>
	struct VoxelElementKey : public VoxelKey<0,1,I_W>
	{
		//inherit constructors
		using BASE = VoxelKey<0,1,I_W>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::depth;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::_data_;

		//define useful constants
		static constexpr uint64_t MAX_ELEMENT_INDEX = BASE::MAX_INDEX-2;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;

		//define element specific constructors
		constexpr VoxelElementKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(0,0,dd,ii,jj,kk) {assert(is_valid());}

		constexpr VoxelElementKey(const uint64_t dd, uint64_t li) {
			assert(dd<=MAX_DEPTH);
			assert(li < (uint64_t{1} << (3*dd)));

			const uint64_t mask = (uint64_t{1} << dd) - 1;
			const uint64_t ii   = li & mask; li >>= dd;
			const uint64_t jj   = li & mask; li >>= dd;
			const uint64_t kk   = li & mask;

			_data_ = (dd << BASE::D_S) | (ii << BASE::I_S) | (jj << BASE::J_S) | (kk << BASE::K_S);
		}

		//check if a voxel is valid
		constexpr bool is_valid() const {
			const uint64_t mei = (uint64_t{1} << depth()) - 2; //max element index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mei) 			 {return false;}
			if (j() > mei) 			 {return false;}
			if (k() > mei) 			 {return false;}
			return true;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			assert(is_valid());
			const uint64_t dd   = depth();
			const uint64_t mask = ((uint64_t{1} << dd) - 1); //maximum element index at this depth
			const uint64_t ii   = (_data_ >> BASE::I_S) & mask;
			const uint64_t jj   = (_data_ >> BASE::J_S) & mask;
			const uint64_t kk   = (_data_ >> BASE::K_S) & mask;
			return ii | (jj << dd) | (kk << (2*dd));
		}

		//hierarchy logic
		constexpr VoxelElementKey parent() const {
			assert(this->exists());
			if (depth()==0) {return VoxelElementKey{DOES_NOT_EXIST};}
			return VoxelElementKey{depth()-1, i()/2, j()/2, k()/2}; //integer division is intended
		}

		inline constexpr VoxelElementKey child(const bool bi, const bool bj, const bool bk) const {
			return VoxelElementKey{depth()+1, 2*i()+static_cast<uint64_t>(bi), 2*j()+static_cast<uint64_t>(bj), 2*k()+static_cast<uint64_t>(bk)};
		}

		constexpr VoxelElementKey child(const int cn) const {
			switch (cn) {
			case 0: return child(0,0,0);
			case 1: return child(1,0,0);
			case 2: return child(0,1,0);
			case 3: return child(1,1,0);
			case 4: return child(0,0,1);
			case 5: return child(1,0,1);
			case 6: return child(0,1,1);
			case 7: return child(1,1,1);
			default: return VoxelElementKey{DOES_NOT_EXIST};
			}
		}

		constexpr bool is_active() const {return static_cast<bool>(this->other()&1);}
		constexpr void set_active(const bool a) {
			if (a) {_data_|=BASE::O_S;}
			else {_data_&= ~BASE::O_S;}
		}


		//adjacency logic
		inline constexpr VoxelVertexKey<I_W> vertex(const bool bi, const bool bj, const bool bk) const;
		constexpr VoxelVertexKey<I_W> vertex(const int vn) const;
		constexpr VoxelFaceKey<I_W> face(const int fn) const;
	};
}