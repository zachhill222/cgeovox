#pragma once

#include "voxel_mesh/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>


namespace gv::vmesh
{
	//define the face key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<int I_W>
	struct VoxelElementKey;

	template<int I_W>
	struct VoxelVertexKey;

	template<int I_W=16>
	struct VoxelFaceKey : public VoxelKey<0,2,I_W>
	{
		//inherit constructors
		using BASE = VoxelKey<0,2,I_W>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::depth;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::_data_;

		//re-name other() to axis() for readability
		//get the normal axis to the face
		inline constexpr uint64_t axis() const {return this->other();}

		//define useful constants
		static constexpr uint64_t MAX_FACE_INDEX_NAX = BASE::MAX_INDEX -1;
		static constexpr uint64_t MAX_FACE_INDEX_AX  = BASE::MAX_INDEX;
		static constexpr uint64_t A_S                = BASE::O_S;
		static constexpr uint64_t A_W				 = BASE::O_W;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;

		//define face specific constructors
		VoxelFaceKey(const uint64_t aa, const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(0,aa,dd,ii,jj,kk) {assert(is_valid());}

		constexpr VoxelFaceKey(const uint64_t dd, uint64_t li) {
			//define by linear index
			assert(dd <= MAX_DEPTH);
			assert(li < 3*(uint64_t{1} << (3*dd+1)));

			const uint64_t mna = (uint64_t{1} << dd) -1;     //maximum element index at this depth, non-axis indices
			const uint64_t ma  = (uint64_t{1} << (dd+1)) -1; //maximum vertex index at this depth, axis index

			//get axis to unpack indices correctly
			const uint64_t aa = (li >> (3*dd+1)) & 3; //get the two most significant bits of the linear index

			//unpack indices
			const uint64_t ii = li & (aa==0 ? ma : mna); li >>= (aa==0 ? dd+1 : dd);
			const uint64_t jj = li & (aa==1 ? ma : mna); li >>= (aa==1 ? dd+1 : dd);
			const uint64_t kk = li & (aa==2 ? ma : mna);

			//construct key
			_data_ = (dd << BASE::D_S) | (aa << A_S) | (ii << BASE::I_S) | (jj << BASE::J_S) | (kk << BASE::K_S);
		}

		//check if a face is valid
		constexpr bool is_valid() const {
			const uint64_t mfiax  = uint64_t{1} << depth(); //max face index in the normal axis direction
			const uint64_t mfinax = mfiax-1;
			const uint64_t aa     = axis();
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > ((aa==0) ? mfiax : mfinax)) {return false;}
			if (j() > ((aa==1) ? mfiax : mfinax)) {return false;}
			if (k() > ((aa==2) ? mfiax : mfinax)) {return false;}
			return true;
		}

		//conversion to the linear index each axis is contiguous
		//loop order should be axis - k - j - i (outer to inner)
		//for each axis, there are 2^(3d+1) faces, 2^(d+1) for the axis-index and 2^d for the other two
		//thus for each depth, there are 3*2^(3*d+1) unique faces and the linear index range is [0, 3*2^(3*d+1) ) (half-open)
		constexpr uint64_t depth_linear_index() const {
			const uint64_t dd = depth();
			const uint64_t aa = axis();
			const uint64_t mna    = (uint64_t{1} << dd) - 1;    //maximum element index at this depth, non-axis indices
			const uint64_t ma     = (uint64_t{1} << (dd+1)) -1; //maximum vertex index at this depth, axis index
			const uint64_t ii = (_data_ >> BASE::I_S) & (aa==0 ? ma : mna);
			const uint64_t jj = (_data_ >> BASE::J_S) & (aa==1 ? ma : mna);
			const uint64_t kk = (_data_ >> BASE::K_S) & (aa==2 ? ma : mna);
			
			uint64_t idx = ii;
			idx |= jj << (aa==0 ? dd+1 : dd); //move past ii
			idx |= kk << (aa==2 ? 2*dd : 2*dd+1); //move past i_data and jj
			idx |= aa << (3*dd+1);
			return idx;
		}

		//geometry logic
		const bool on_bbox_boundary() const {
			const uint64_t mfiax = uint64_t{1} << depth();
			const uint64_t idx   = this->index(axis());
			return idx==0 || idx==mfiax;
		}

		//hierarchy logic
		constexpr VoxelFaceKey child(const bool ii, const bool jj) const {
			const uint64_t a = axis();
			const uint64_t ci=2*i(), cj=2*j(), ck=2*k();
			
			switch (a) {
			case 0: return VoxelFaceKey{a,depth()+1, ci, cj+static_cast<uint64_t>(ii), ck+static_cast<uint64_t>(jj)};
			case 1: return VoxelFaceKey{a,depth()+1, ci+static_cast<uint64_t>(ii), cj, ck+static_cast<uint64_t>(jj)};
			case 2: return VoxelFaceKey{a,depth()+1, ci+static_cast<uint64_t>(ii), cj+static_cast<uint64_t>(jj), ck};
			default: return VoxelFaceKey{};
			}
		}

		constexpr VoxelFaceKey child(const int child_number) const {
			switch (child_number) {
			case 0: return child(0,0);
			case 1: return child(1,0);
			case 2: return child(0,1);
			case 3: return child(1,1);
			default: assert(false); return VoxelFaceKey{};
			}
		}

		constexpr VoxelFaceKey parent() const {
			//not all faces have parents
			const uint64_t a = axis();
			const uint64_t d = depth();
			const uint64_t ii = i();
			const uint64_t jj = j();
			const uint64_t kk = k();

			if (d==0) {return VoxelFaceKey{DOES_NOT_EXIST};}
			switch (a) {
			case 0: return (jj&1 || kk&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			case 1: return (kk&1 || ii&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			case 2: return (ii&1 || jj&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			default: return VoxelFaceKey{DOES_NOT_EXIST};
			}
		}

		//adjacency operations
		constexpr VoxelElementKey<I_W> element(const bool forward_flag) const;
		constexpr VoxelVertexKey<I_W>  vertex(const bool ii, const bool jj) const;
		constexpr VoxelVertexKey<I_W>  vertex(const int vertex_number) const;
	};
}