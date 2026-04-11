#pragma once

#include "gutil.hpp"
#include "voxel_mesh/keys/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>
#include <cmath>

namespace gv::vmesh
{
	//define the vertex key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<int I_W, bool MORTON>
	struct VoxelElementKey;

	template<int I_W, bool MORTON>
	struct VoxelFaceKey;

	template<int I_W=16, bool MORTON=false>
	struct VoxelVertexKey : public VoxelKey<0,0,0,I_W>
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
		static constexpr uint64_t MAX_VERTEX_INDEX = BASE::MAX_INDEX;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;

		//define vertex specific constructors
		constexpr VoxelVertexKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(0,0,dd,ii,jj,kk) {assert(is_valid());}

		constexpr VoxelVertexKey(const uint64_t dd, uint64_t li) requires (!MORTON) {
			assert(dd<=MAX_DEPTH);
			const uint64_t nv   = (uint64_t{1} << dd) + 1; //2^d + 1 vertices per axis
			const uint64_t ii   = li % nv; li /= nv;
			const uint64_t jj   = li % nv; li /= nv;
			const uint64_t kk   = li;

			//linear index is ii + nv*jj + nv^2*kk

			_data_ = (dd << BASE::D_S) | (ii << BASE::I_S) | (jj << BASE::J_S) | (kk << BASE::K_S);
			assert(is_valid());
		}

		//check if a voxel is valid
		constexpr bool is_valid() const {
			const uint64_t mvi = uint64_t{1} << depth(); //max vertex index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mvi) 			 {return false;}
			if (j() > mvi) 			 {return false;}
			if (k() > mvi) 			 {return false;}
			return true;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			const uint64_t nv   = depth()+1; //number of vertices per side
			
			// const uint64_t ii   = (_data_ >> BASE::I_S);
			// const uint64_t jj   = (_data_ >> BASE::J_S);
			// const uint64_t kk   = (_data_ >> BASE::K_S);
			if constexpr (MORTON) {
				//morton indexing (two features of the same color on the same depth are 8 indices from eachother)
				//better for looping over a single color
				// const uint64_t clr = (ii&1) | ((jj&1)<<1) | ((kk&1)<<2); //color in the least significant bits
				// const uint64_t rem = ((ii>>1)) | ((jj>>1) << (wd-1)) | ((kk>>1) << (2*wd-1));
				// return (rem<<3) | clr;
				return i() + nv*(j() + nv*k());
			}
			else {
				//standard ordering (contiguous i, loop k->j->i (outer to inner))
				return i() + nv*(j() + nv*k());
			}
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			//(2^d + 1)^3 vertices per depth. expand and sum from 0 to dd-1
			return ((uint64_t{1} << (3*dd)) - 1)/7 
				 + (uint64_t{1}<<(2*dd)) 
				 + 3*(uint64_t{1}<<dd)
				 + dd - 4;
		}

		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
		}

		//geometry logic
		constexpr bool on_bbox_boundary() const {
			const uint64_t mvi = uint64_t{1} << depth();
			const uint64_t ii=i(), jj=j(), kk=k();
			return ii==0 || ii==mvi || jj==0 || jj==mvi || kk==0 || kk==mvi;
		}

		inline constexpr double x() const {return std::ldexp(static_cast<double>(i()), -static_cast<int>(depth()));}
		inline constexpr double y() const {return std::ldexp(static_cast<double>(j()), -static_cast<int>(depth()));}
		inline constexpr double z() const {return std::ldexp(static_cast<double>(k()), -static_cast<int>(depth()));}

		constexpr VoxelVertexKey reduced_key() const {
			//return the vertex key at the lowest depth
			//that is at the same gemetric location as this vertex
			uint64_t dd = depth();
			uint64_t ii=i(), jj=j(), kk=k();
			while (dd>0 && !(ii&1) && !(jj&1) && !(kk&1)) {
				//shift right until the least significant bit of i,j,k is used or the depth is 0
				dd  -= 1;
				ii >>= 1;
				jj >>= 1;
				kk >>= 1;
			}
			return VoxelVertexKey{dd, ii, jj, kk};
		}

		constexpr bool is_same_coord(const VoxelVertexKey other) const {
			return reduced_key() == other.reduced_key();
		}

		constexpr gutil::Point<3,double> normalized_coordinate() const {
			const int exponent = -static_cast<int>(depth());
			return gutil::Point<3,double>{
				std::ldexp(static_cast<double>(i()), exponent),
				std::ldexp(static_cast<double>(j()), exponent),
				std::ldexp(static_cast<double>(k()), exponent)
			};
		}

		//hierarchy logic
		constexpr VoxelVertexKey parent() const {
			assert(this->exists());
			const uint64_t ii = i();
			const uint64_t jj = j();
			const uint64_t kk = k();
			const uint64_t dd = depth();
			if (ii&1 || jj&1 || kk&1 || dd==0) {return VoxelVertexKey{DOES_NOT_EXIST};} //vertices with odd indices don't have parents
			return VoxelVertexKey{dd-1, ii/2, jj/2, kk/2};
		}

		inline constexpr VoxelVertexKey child() const {
			assert(this->exists());
			assert(depth()+1 <= MAX_DEPTH);
			return VoxelVertexKey{depth()+1, 2*i(), 2*j(), 2*k()};
		}

		//adjacency logic
		//TODO: implement if needed

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