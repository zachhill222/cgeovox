#pragma once

#include "voxel_mesh/keys/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>


namespace gv::vmesh
{
	//define the face key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<int I_W, bool MORTON>
	struct VoxelElementKey;

	template<int I_W, bool MORTON>
	struct VoxelVertexKey;

	template<int I_W=16, bool MORTON=false>
	struct VoxelFaceKey : public VoxelKey<0,0,2,I_W>
	{
		//inherit constructors
		using BASE = VoxelKey<0,0,2,I_W>;
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

		constexpr VoxelFaceKey(const uint64_t dd, uint64_t li) requires (!MORTON) {
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

		//partition the indices into: axis0 normal | axis1 normal | axis2 normal
		//the numbering within and across each partion is continuous as there are 2^d * 2^d * 2^(d+1) = 2^(3d+1) faces with a given normal axis
		//thus axis0 uses indices [0,N), axis1 uses [N,2N), and axis2 uses [2N,3N).
		//the numbering within a partion is either standard or morton. use morton if looping by color and standard if looping by index
		//for better memory strides

		//the standard indexing within an axis group is
		// a2 | a1 | a0 where a2 is the axis-index, a1 is the larger non-axis index and a0 is the smaller non-axis index
		//this organizes faces in the same plane contiguously
		//note a1 and a0 are (depth) bits wide while a2 is (depth+1) bits wide

		//for morton indexing, the last two bits of a1 and a0 are moved to the LSB
		// a2 | a1-remainder | a0-remainder | a1&1 | a0&1
		//note that a1-rem and a0-rem are (depth-1) bits wide and a2 is still (depth+1) bits wide
		//the color a1&1 | a0&1 is of course 2 bits wide.

		constexpr uint64_t depth_linear_index() const {
			const uint64_t dd = depth();
			const uint64_t aa = axis();

			//we assume that the index a,i,j,k is valid and do not mask it
			const uint64_t a0 = (aa == 0) ? j() : i(); //first index
			const uint64_t a1 = (aa == 2) ? j() : k(); //second index
			const uint64_t a2 = (aa == 0) ? i() : (aa==1) ? j() : k(); //axis index

			const uint64_t ps = depth_axis_start(dd,aa); //start of this partision
			
			if constexpr (MORTON) {
				const uint64_t clr = (a0&1) | ((a1&1)<<1); //color bits
				const uint64_t r0  = a0>>1; //mask and compact the low-index bits
				const uint64_t r1  = a1>>1; //mask and compact the high-index bits
				const uint64_t pn  = clr | (r0<<2) | (r1<<(dd+1)) | (a2<<(2*dd));
				return ps + pn;
			}
			else {
				
				const uint64_t pn = (a0) | (a1<<dd) | (a2<<(2*dd)); //index within this axis partition
				return ps + pn;
			}
		}

		static const uint64_t depth_axis_start(const uint64_t dd, const uint64_t aa) {
			const uint64_t N = uint64_t{1} << (3*dd+1);
			return aa*N;
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			// 8^d + 4^d faces per axis at depth d, summed from 0 to dd-1
			return 3 * (((uint64_t{1} << (3*dd)) - 1)/7) + ((uint64_t{1}<<(2*dd)) - 1);
		}

		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
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
		constexpr VoxelElementKey<I_W,MORTON> element(const bool forward_flag) const;
		constexpr VoxelVertexKey<I_W,MORTON>  vertex(const bool ii, const bool jj) const;
		constexpr VoxelVertexKey<I_W,MORTON>  vertex(const int vertex_number) const;

		//color by the even/odd parity of the non-axis indices
		inline constexpr uint64_t color() const {
			const uint64_t aa = axis();
			const uint64_t a0 = (aa == 0) ? j() : i(); //first index
			const uint64_t a1 = (aa == 2) ? j() : k(); //second index
			return (a0&1) | ((a1&1)<<1);
		}
	};
}