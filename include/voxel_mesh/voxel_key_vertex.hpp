#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>
#include <cmath>

namespace gv::vmesh
{
	//define the vertex key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<int I_W>
	struct VoxelElementKey;

	template<int I_W>
	struct VoxelFaceKey;

	template<int I_W=16>
	struct VoxelVertexKey : public VoxelKey<0,0,I_W>
	{
		//inherit constructors
		using BASE = VoxelKey<0,0,I_W>;
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

		constexpr VoxelVertexKey(const uint64_t dd, uint64_t li) {
			assert(dd<=MAX_DEPTH);
			assert(li < (uint64_t{1} << (3*dd+3)));

			const uint64_t mask = (uint64_t{1} << (dd+1)) - 1;
			const uint64_t ii   = li & mask; li >>= (dd+1);
			const uint64_t jj   = li & mask; li >>= (dd+1);
			const uint64_t kk   = li & mask;

			_data_ = (dd << BASE::D_S) | (ii << BASE::I_S) | (jj << BASE::J_S) | (kk << BASE::K_S);
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
			const uint64_t dd   = depth();
			const uint64_t mask = ((uint64_t{1} << (dd+1)) - 1); //2^d is the largets valid vertex index
			const uint64_t ii   = (_data_ >> BASE::I_S) & mask;
			const uint64_t jj   = (_data_ >> BASE::J_S) & mask;
			const uint64_t kk   = (_data_ >> BASE::K_S) & mask;
			return ii | (jj << dd) | (kk << (2*dd));
		}

		//geometry logic
		const bool on_bbox_boundary() const {
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
	};
}