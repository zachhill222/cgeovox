#pragma once

#include "voxel_mesh/mesh/keys/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>
#include <array>

namespace gv::vmesh
{
	//define the element key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<uint64_t I_W, bool MORTON>
	struct VoxelVertexKey;

	template<uint64_t I_W, bool MORTON>
	struct VoxelFaceKey;

	template<uint64_t I_W=16, bool MORTON_=false>
	struct VoxelElementKey : public VoxelKey<0,0,0,I_W>
	{
		//inherit constructors
		using BASE = VoxelKey<0,0,0,I_W>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::depth;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::_data_;

		//define useful constants
		static constexpr uint64_t MAX_ELEMENT_INDEX = BASE::MAX_INDEX-2;
		static constexpr bool MORTON = MORTON_;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;


		//define element specific constructors
		constexpr VoxelElementKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(0,0,dd,ii,jj,kk) {
				if (!is_valid()) {_data_=DOES_NOT_EXIST;}
			}

		constexpr VoxelElementKey(const uint64_t dd, uint64_t li) requires (!MORTON) {
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
			const uint64_t mei = (uint64_t{1} << depth()) - 1; //max element index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mei) 			 {return false;}
			if (j() > mei) 			 {return false;}
			if (k() > mei) 			 {return false;}
			return true;
		}

		VoxelElementKey& operator++() requires (!MORTON) {
			uint64_t dd = depth();
			const uint64_t mi = (uint64_t{1} << dd)-1; //2^dd elements per axis (max index)
			uint64_t ii=i(), jj=j(), kk=k();
			
			//increment with carry, but we must respect the entire
			//field width of I_W, J_W, and K_W
			++ii;
			if (ii>mi) {ii=0; ++jj;}
			if (jj>mi) {jj=0; ++kk;}
			if (kk>mi) {kk=0; ++dd;}

			//reset d-i-j-k bits
			_data_ &= ~BASE::DEPTH_INDEX_MASK;
			_data_ |=  BASE::DEPTH_INDEX_MASK & ((ii<<BASE::I_S) | (jj<<BASE::J_S) | (kk<<BASE::K_S) | (dd<<BASE::D_S));
			return *this;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			if (!is_valid()) {
				std::cout << "INVALID ELEMENT\n" << *this << std::endl;
			}

			assert(is_valid());
			const uint64_t dd   = depth();
			const uint64_t mask = ((uint64_t{1} << dd) - 1); //maximum element index at this depth
			const uint64_t ii   = (_data_ >> BASE::I_S) & mask;
			const uint64_t jj   = (_data_ >> BASE::J_S) & mask;
			const uint64_t kk   = (_data_ >> BASE::K_S) & mask;
			
			if constexpr (MORTON) {
				//morton indexing (two features of the same color on the same depth are 8 indices from eachother)
				//better for looping over a single color
				const uint64_t clr = (ii&1) | ((jj&1)<<1) | ((kk&1)<<2); //color in the least significant bits
				const uint64_t rem = ((ii>>1)) | ((jj>>1) << (dd-1)) | ((kk>>1) << (2*dd-1));
				return (rem<<3) | clr;
			}
			else {
				//standard ordering (contiguous i, loop k->j->i (outer to inner))
				return ii | (jj << dd) | (kk << (2*dd));
			}
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			//sum from d=0 to dd-1 of 8^d
			return ((uint64_t{1} << (3*dd)) - 1)/7;
		}

		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
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

		constexpr std::array<VoxelElementKey,8> children() const {
			const uint64_t ii=2*i(), jj=2*j(), kk=2*k(), dd=depth()+1;
			if (dd>MAX_DEPTH) {return{};}
			return {
				VoxelElementKey{dd, ii  ,jj  ,kk  },
				VoxelElementKey{dd, ii+1,jj  ,kk  },
				VoxelElementKey{dd, ii  ,jj+1,kk  },
				VoxelElementKey{dd, ii+1,jj+1,kk  },
				VoxelElementKey{dd, ii  ,jj  ,kk+1},
				VoxelElementKey{dd, ii+1,jj  ,kk+1},
				VoxelElementKey{dd, ii  ,jj+1,kk+1},
				VoxelElementKey{dd, ii+1,jj+1,kk+1}
			};
		}

		//adjacency logic
		inline constexpr VoxelVertexKey<I_W,MORTON_> vertex(const bool bi, const bool bj, const bool bk) const;
		constexpr VoxelVertexKey<I_W,MORTON_> vertex(const int vn) const;
		constexpr VoxelFaceKey<I_W,MORTON_> face(const int fn) const;

		//get the color of the feature by index modding. There are 8 unique colors.
		//coloring means different things for differnet mesh elements, but for example,
		//two elements with the same color at the same depth share no vertices or faces
		inline constexpr uint64_t color() const {
			//even/even/even -> 0
			//odd/even/even  -> 1
			//even/odd/even  -> 2
			//etc.
			return (i()&1) | ((j()&1) << 1) | ((k()&1) << 2);
		}
	};
}