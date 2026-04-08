#pragma once

#include "gutil.hpp"

#include <cstdint>
#include <cassert>
#include <cmath>

#include <stdexcept>
#include <string>



namespace gv::vmesh
{
	//These are helper data types for a hierarchical voxel mesh
	//This structure allows us to very efficiently store elements, vertices, faces, etc.
	//The base mesh is 1x1x1 so that every vertex is (in reference coordinates) a dyadic rational number
	//Elements, vertices, and faces all have special index/key structs for their storage and logical relations.
	//All information for every element, vertex, and face is compressed into a 64-bit unsigned integer
	struct VoxelElementKey;
	struct VoxelVertexKey;
	struct VoxelFaceKey;

	//shared constants
	static constexpr uint64_t DOES_NOT_EXIST = uint64_t(-1);
	static constexpr uint64_t MAX_DEPTH = 16;

	//shared bit offsets
	static constexpr int DEPTH_START =   12;
	static constexpr int I_START     =   16;
	static constexpr int J_START     = 2*16;
	static constexpr int K_START     = 3*16;

	//shared masks
	static constexpr uint64_t IJK_MASK    = (uint64_t{1} << 16) - 1;
	static constexpr uint64_t DEPTH_MASK  = (uint64_t{1} << 4) - 1;

	//////////////////////////////////////////////////////////////////////////
	/// Element Key
	///
	/// bit  00-10 : unused (usually 0 if initialized successfully)
	/// bits 11    : active flag
	/// bits 12-15 : depth
	/// bits 16-31 : i
	/// bits 32-47 : j
	/// bits 48-63 : k
	///
	///
	/// Relation to child elements:
	/// If element E has tuple (d,i,j,k), then its children have tuples (d+1, ci, cj, ck)
	/// where ci - 2*i = 0 or 1, cj - 2*j = 0 or 1, ck - 2*k = 0 or 1
	//////////////////////////////////////////////////////////////////////////
	struct VoxelElementKey
	{
		//key
		uint64_t data = DOES_NOT_EXIST;

		//active mask is unique to VoxelElementKey
		static constexpr uint64_t ACTIVE_MASK = uint64_t{1} << 11;

		//accessors
		inline constexpr uint64_t i()     const {return (data>>I_START)     & IJK_MASK;}
		inline constexpr uint64_t j()     const {return (data>>J_START)     & IJK_MASK;}
		inline constexpr uint64_t k()     const {return (data>>K_START)     & IJK_MASK;}
		inline constexpr uint64_t depth() const {return (data>>DEPTH_START) & DEPTH_MASK;}
		inline constexpr uint64_t index(const int axis) const {
			switch (axis) {
				case 0: return i();
				case 1: return j();
				case 2: return k();
				default: assert(false); return 0;
			}
		}

		//constructors
		constexpr VoxelElementKey() : data(DOES_NOT_EXIST) {};
		explicit constexpr VoxelElementKey(uint64_t data) : data(data) {}

		constexpr VoxelElementKey(const uint64_t depth, uint64_t linear_index) {
			assert(depth<MAX_DEPTH);
			assert(linear_index < (uint64_t{1} << (3*depth)));

			const uint64_t mask   = (uint64_t{1} << depth) - 1;
			const uint64_t i_data = linear_index & mask; linear_index >>= depth;
			const uint64_t j_data = linear_index & mask; linear_index >>= depth;
			const uint64_t k_data = linear_index & mask;

			data = (depth << DEPTH_START) | (i_data << I_START) | (j_data << J_START) | (k_data << K_START);
		}

		constexpr VoxelElementKey(const uint64_t depth, const uint64_t i, const uint64_t j, const uint64_t k) :
			data(
				(DEPTH_MASK & depth) << DEPTH_START | 
				(IJK_MASK   & i)     << I_START     |
				(IJK_MASK   & j)     << J_START     |
				(IJK_MASK   & k)     << K_START
				)
			{
				//note that the first two bits are always 0 here, so elements are inavtive by default
				//debug asserts to warn if truncating occured
				assert(depth <= DEPTH_MASK);
				assert(    i < IJK_MASK);
				assert(    j < IJK_MASK);
				assert(    k < IJK_MASK);
			}

		//convert to a linear index comparing the linear index of two valid elements at the same depth
		//should be equivalent to using operator<
		//note that loops should be over k -> j -> i (innermost) to iterate over a contiguous linear index.
		constexpr uint64_t depth_linear_index() const {
			const uint64_t d = depth();
			const uint64_t mask = ((uint64_t{1} << d) - 1); //maximum element index at this depth
			const uint64_t i_data = (data >> I_START) & mask;
			const uint64_t j_data = (data >> J_START) & mask;
			const uint64_t k_data = (data >> K_START) & mask;
			return i_data | (j_data << d) | (k_data << (2*d));
		}

		//comparisons and logical operations
		constexpr bool exists() const {
			return data!=DOES_NOT_EXIST;
		}
		
		constexpr bool is_valid() const	{
			//depth is always valid
			//indices can't be their maximum value, otherwise its faces and vertices
			//won't have valid indices
			const uint64_t max_elem_index = uint64_t{1} << depth(); //2^depth elements in each coordinate direction
			if (i() >= max_elem_index) {return false;}
			if (j() >= max_elem_index) {return false;}
			if (k() >= max_elem_index) {return false;}
			return true;
		}

		constexpr bool operator==(const VoxelElementKey other) const {
			//the comparison should not care if the element is active or not
			return (data & ~ACTIVE_MASK) == (other.data & ~ACTIVE_MASK);
		}

		constexpr bool operator<(const VoxelElementKey other) const	{
			//the comparison should not care if the element is active or not
			return (data & ~ACTIVE_MASK) < (other.data & ~ACTIVE_MASK);
		}

		//hierarcy logic
		constexpr VoxelElementKey child(const bool ii, const bool jj, const bool kk) const {
			return VoxelElementKey{depth()+1, 2*i()+static_cast<uint64_t>(ii), 2*j()+static_cast<uint64_t>(jj), 2*k()+static_cast<uint64_t>(kk)};
		}
		
		constexpr VoxelElementKey child(const int child_number) const {
			switch (child_number) {
			case 0: return child(0,0,0);
			case 1: return child(1,0,0);
			case 2: return child(0,1,0);
			case 3: return child(1,1,0);
			case 4: return child(0,0,1);
			case 5: return child(1,0,1);
			case 6: return child(0,1,1);
			case 7: return child(1,1,1);
			default:
				throw std::range_error("VoxelElementKey::child(int) - invalid child_number (" + std::to_string(child_number) + ")");
				return VoxelElementKey{};
			}
		}

		constexpr VoxelElementKey parent() const {
			assert(exists());
			if (this->depth()==0) {return VoxelElementKey{DOES_NOT_EXIST};}
			return VoxelElementKey{depth()-1, i()>>1, j()>>1, k()>>1}; //integer part of i,j,k/2
		}

		constexpr bool is_active() const {return ACTIVE_MASK & data;}

		void set_active(const bool flag) {
			if (flag) {data |= ACTIVE_MASK;}
			else {data &= ~ACTIVE_MASK;}
		}

		constexpr bool is_parent_of(const VoxelElementKey other) const {
			assert(exists());
			assert(other.exists());
			return other.parent() == *this;
		}

		constexpr bool is_child_of(const VoxelElementKey other) const {
			assert(exists());
			assert(other.exists());
			return parent() == other;
		}

		constexpr bool is_ancestor_of(const VoxelElementKey other) const {
			assert(exists());
			assert(other.exists());

			const uint64_t this_depth = this->depth();
			const uint64_t other_depth = other.depth();
			if (this_depth >= other_depth) {return false;}

			const uint64_t dd    = other_depth - this_depth;
			const uint64_t range = uint64_t{1} << dd;

			if (other.i() - (this->i() << dd) >= range) {return false;}
			if (other.j() - (this->j() << dd) >= range) {return false;}
			if (other.k() - (this->k() << dd) >= range) {return false;}
			return true;
		}
		
		constexpr bool is_descendant_of(const VoxelElementKey other) const {
			assert(exists());
			assert(other.exists());
			return other.is_ancestor_of(*this);
		}

		//adjacency operations
		constexpr VoxelVertexKey vertex(const bool ii, const bool jj, const bool kk) const;
		constexpr VoxelVertexKey vertex(const int vertex_number) const;
		constexpr VoxelFaceKey face(const int face_number) const;		
	};



	//////////////////////////////////////////////////////////////////////////
	/// Vertex Key
	///
	/// bit  00-11 : unused (usually 0 if initialized successfully)
	/// bits 12-15 : depth
	/// bits 16-31 : i
	/// bits 32-47 : j
	/// bits 48-63 : k
	///
	/// The normalized coordinate of a vertex is at:
	/// 		x = 2^-depth * i
	///			y = 2^-depth * j
	/// 		z = 2^-depth * k
	///
	/// Note that different memory layouts have may correspond to the same point in space.
	/// For example if i,j,k all have a factor of 2^d, then then the same point in space can be reached at a lower depth.
	//////////////////////////////////////////////////////////////////////////
	struct VoxelVertexKey
	{
		//key
		uint64_t data = DOES_NOT_EXIST;

		//accessors
		inline constexpr uint64_t i()     const {return (data>>I_START)     & IJK_MASK;}
		inline constexpr uint64_t j()     const {return (data>>J_START)     & IJK_MASK;}
		inline constexpr uint64_t k()     const {return (data>>K_START)     & IJK_MASK;}
		inline constexpr uint64_t depth() const {return (data>>DEPTH_START) & DEPTH_MASK;}
		inline constexpr uint64_t index(const int axis) const {
			switch (axis) {
				case 0: return i();
				case 1: return j();
				case 2: return k();
				default: assert(false); return 0;
			}
		}

		//constructors
		constexpr VoxelVertexKey() : data{DOES_NOT_EXIST} {}
		explicit constexpr VoxelVertexKey(const uint64_t data) : data(data) {}
		
		constexpr VoxelVertexKey(const uint64_t depth, uint64_t linear_index) {
			assert(depth<MAX_DEPTH);
			assert(linear_index < (uint64_t{1} << (3*depth+3)));

			const uint64_t mask   = (uint64_t{1} << (depth+1)) - 1;
			const uint64_t i_data = linear_index & mask; linear_index >>= depth;
			const uint64_t j_data = linear_index & mask; linear_index >>= depth;
			const uint64_t k_data = linear_index & mask;

			data = (depth << DEPTH_START) | (i_data << I_START) | (j_data << J_START) | (k_data << K_START);
		}

		constexpr VoxelVertexKey(const uint64_t depth, const uint64_t i, const uint64_t j, const uint64_t k) :
			data(
				(DEPTH_MASK & depth) << DEPTH_START | 
				(IJK_MASK   & i)     << I_START     |
				(IJK_MASK   & j)     << J_START     |
				(IJK_MASK   & k)     << K_START
				)
			{
				//debug asserts to warn if truncating occured
				assert(depth <= DEPTH_MASK);
				assert(    i <= IJK_MASK);
				assert(    j <= IJK_MASK);
				assert(    k <= IJK_MASK);
			}

		//convert to linear indexing
		constexpr uint64_t depth_linear_index() const {
			const uint64_t d = depth();
			const uint64_t mask = ((uint64_t{1} << (d+1)) - 1); //2^d is the largets valid vertex index
			const uint64_t i_data = (data >> I_START) & mask;
			const uint64_t j_data = (data >> J_START) & mask;
			const uint64_t k_data = (data >> K_START) & mask;
			return i_data | (j_data << d) | (k_data << (2*d));
		}

		//comparisons and logical operations
		constexpr bool exists() const {return data!=DOES_NOT_EXIST;}

		constexpr bool is_valid() const {
			const uint64_t max_elem_index = (uint64_t{1} << depth()); //2^depth elements in each coordinate direction
			if (i() >= max_elem_index+1) {return false;}
			if (j() >= max_elem_index+1) {return false;}
			if (k() >= max_elem_index+1) {return false;}
			return true;
		}

		constexpr bool on_bbox_boundary() const	{
			const uint64_t max = uint64_t{1} << depth();
			const uint64_t vi=i(), vj=j(), vk=k();
			return vi==0 || vi==max || vj==0 || vj==max || vk==0 || vk==max;
		}

		constexpr bool operator==(const VoxelVertexKey other) const {return data == other.data;}

		constexpr bool operator<(const VoxelVertexKey other) const {return data < other.data;}

		//convert to normalized coordinates
		constexpr double x() const {return std::ldexp(static_cast<double>(i()), -static_cast<int>(depth()));}

		constexpr double y() const {return std::ldexp(static_cast<double>(j()), -static_cast<int>(depth()));}

		constexpr double z() const {return std::ldexp(static_cast<double>(k()), -static_cast<int>(depth()));}

		constexpr gutil::Point<3,double> normalized_coordinate() const {
			const int exponent = -static_cast<int>(depth());
			return gutil::Point<3,double>{
				std::ldexp(static_cast<double>(i()), exponent),
				std::ldexp(static_cast<double>(j()), exponent),
				std::ldexp(static_cast<double>(k()), exponent)
			};
		}


		//hieararchy logic
		constexpr VoxelVertexKey reduced_key() const {
			//return the vertex key to this normalized coordinate at the lowest possible depth
			uint64_t dd = depth();
			uint64_t ii = i();
			uint64_t jj = j();
			uint64_t kk = k();

			while (dd>0 && !(ii&1) && !(jj&1) && !(kk&1)) {
				//shift right until the least significant bit of i,j,k is used or the depth is 0
				dd  -= 1;
				ii >>= 1;
				jj >>= 1;
				kk >>= 1;
			}

			return VoxelVertexKey{dd, ii, jj, kk};
		}

		constexpr VoxelVertexKey child() const {
			return VoxelVertexKey{depth()+1, 2*i(), 2*j(), 2*k()};
		}

		constexpr VoxelVertexKey parent() const {
			const uint64_t ii = i();
			const uint64_t jj = j();
			const uint64_t jj = k();
			const uint64_t dd = depth();
			if (ii&1 || jj&1 || kk&1 || dd==0) {return VoxelVertexKey{DOES_NOT_EXIST};}
			return VoxelVertexKey{dd-1, ii>>1, jj>>1, kk>>1}; //decrease depth, divide index by 2
		}

		constexpr bool is_same_coord(const VoxelVertexKey other) const {
			return this->reduced_key() == other.reduced_key();
		}

		//adjacency operations
		constexpr VoxelElementKey element(const bool ii, const bool jj, const bool kk) const;
		constexpr VoxelElementKey element(const int element_number) const;
		constexpr VoxelFaceKey face(const int face_number) const;
	};




	//////////////////////////////////////////////////////////////////////////
	/// Face Key
	///
	/// bit  00-08 : unused (usually 0 if initialized successfully)
	/// bits 09-11 : axis
	/// bits 12-15 : depth
	/// bits 16-31 : i
	/// bits 32-47 : j
	/// bits 48-63 : k
	//////////////////////////////////////////////////////////////////////////
	struct VoxelFaceKey
	{
		//key
		uint64_t data = DOES_NOT_EXIST;

		//axis data is unique to VoxelFaceKey
		static constexpr int      AXIS_START = 9;
		static constexpr uint64_t AXIS_MASK  = (uint64_t{1} << 2 ) - 1;

		//accessors
		inline constexpr uint64_t i()     const {return (data>>I_START)     & IJK_MASK;}
		inline constexpr uint64_t j()     const {return (data>>J_START)     & IJK_MASK;}
		inline constexpr uint64_t k()     const {return (data>>K_START)     & IJK_MASK;}
		inline constexpr uint64_t depth() const {return (data>>DEPTH_START) & DEPTH_MASK;}
		inline constexpr uint64_t axis()  const {return (data>>AXIS_START)  & AXIS_MASK;}
		inline constexpr uint64_t index(const int axis) const {
			switch (axis) {
				case 0: return i();
				case 1: return j();
				case 2: return k();
				default: assert(false); return 0;
			}
		}
		

		//constructors
		constexpr VoxelFaceKey() : data{DOES_NOT_EXIST} {}
		explicit constexpr VoxelFaceKey(const uint64_t data) : data(data) {}

		constexpr VoxelFaceKey(const uint64_t depth, uint64_t linear_index) {
			assert(depth < MAX_DEPTH);
			assert(linear_index < 3*(uint64_t{1} << (3*depth+1)));

			const uint64_t mask      = (uint64_t{1} << depth) -1; //maximum element index at this depth, non-axis indices
			const uint64_t mask_axis = (uint64_t{1} << (depth+1)) -1; //maximum vertex index at this depth, axis index

			//get axis to unpack indices correctly
			const uint64_t axis = (linear_index >> (3*depth+1)) & 3; //get the two most significant bits of the linear index

			//unpack indices
			const uint64_t i_data = linear_index & (axis==0 ? mask_axis : mask); linear_index >>= (axis==0 ? depth+1 : depth);
			const uint64_t j_data = linear_index & (axis==1 ? mask_axis : mask); linear_index >>= (axis==1 ? depth+1 : depth);
			const uint64_t k_data = linear_index & (axis==2 ? mask_axis : mask);

			//construct key
			data = (depth << DEPTH_START) | (axis << AXIS_START) | (i_data << I_START) | (j_data << J_START) | (k_data << K_START);
		}

		constexpr VoxelFaceKey(const uint64_t axis, const uint64_t depth, const uint64_t i, const uint64_t j, const uint64_t k) :
			data(
				(AXIS_MASK  & axis)  << AXIS_START  |
				(DEPTH_MASK & depth) << DEPTH_START | 
				(IJK_MASK   & i)     << I_START     |
				(IJK_MASK   & j)     << J_START     |
				(IJK_MASK   & k)     << K_START
				)
			{
				//debug asserts to warn if truncating occured
				assert(axis  <= 2);
				assert(depth <= DEPTH_MASK);
				assert(    i <= IJK_MASK);
				assert(    j <= IJK_MASK);
				assert(    k <= IJK_MASK);
			}

		//conversion to the linear index each axis is contiguous
		//loop order should be axis - k - j - i (outer to inner)
		//for each axis, there are 2^(3d+1) faces, 2^(d+1) for the axis-index and 2^d for the other two
		//thus for each depth, there are 3*2^(3*d+1) unique faces and the linear index range is [0, 3*2^(3*d+1) ) (half-open)
		constexpr uint64_t depth_linear_index() const {
			const uint64_t d = depth();
			const uint64_t a = axis();
			const uint64_t mask      = (uint64_t{1} << d) - 1; //maximum element index at this depth, non-axis indices
			const uint64_t mask_axis = (uint64_t{1} << (d+1)) -1; //maximum vertex index at this depth, axis index
			const uint64_t i_data = (data >> I_START) & (a==0 ? mask_axis : mask);
			const uint64_t j_data = (data >> J_START) & (a==1 ? mask_axis : mask);
			const uint64_t k_data = (data >> K_START) & (a==2 ? mask_axis : mask);
			
			uint64_t idx = i_data;
			idx |= j_data << (a==0 ? d+1 : d); //move past i_data
			idx |= k_data << (a==2 ? 2*d : 2*d+1); //move past i_data and j_data
			idx |= a << (3*d+1);
			return idx;
		}



		//comparisons
		constexpr bool exists() const {
			return data!=DOES_NOT_EXIST;
		}

		constexpr bool is_valid() const	{
			const uint64_t max_elem_index = (uint64_t{1} << depth()); //2^depth elements in each coordinate direction
			const uint64_t a = axis();
			for (uint64_t b=0; b<3; ++b) {
				bool invalid = index(b) >= max_elem_index + (a==b ? 1 : 0);
				if (invalid) {return false;}
			}
			return true;
		}

		constexpr bool on_bbox_boundary() const {
			const uint64_t max = uint64_t{1} << depth();
			const uint64_t idx = index(axis());
			return idx==0 || idx==max;
		}

		constexpr bool operator==(const VoxelFaceKey other) const {return data == other.data;}
		constexpr bool operator<(const VoxelFaceKey other) const {return data < other.data;}
		
		//hierarchy operations
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

			if (d==0) {return VoxelElementKey{DOES_NOT_EXIST};}
			switch (a) {
			case 0: return (jj&1 || kk&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			case 1: return (kk&1 || ii&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			case 2: return (ii&1 || jj&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{a,d-1, ii>>1, jj>>1, kk>>1};
			default: return VoxelFaceKey{DOES_NOT_EXIST};
			}
		}

		//adjacency operations
		constexpr VoxelElementKey element(const bool forward_flag) const;
		constexpr VoxelVertexKey vertex(const bool ii, const bool jj) const;
		constexpr VoxelVertexKey vertex(const int vertex_number) const;
	};


	/////////////////////////////////////////////////////////////////////////
	/// Adjacency operation definitions (Element)
	/////////////////////////////////////////////////////////////////////////
	constexpr VoxelVertexKey VoxelElementKey::vertex(const bool ii, const bool jj, const bool kk) const
	{
		return VoxelVertexKey(
			depth(),
			i() + static_cast<uint64_t>(ii),
			j() + static_cast<uint64_t>(jj),
			k() + static_cast<uint64_t>(kk)
		);
	}

	constexpr VoxelVertexKey VoxelElementKey::vertex(const int vertex_number) const
	{
		switch (vertex_number) {
		case 0: return vertex(0,0,0);
		case 1: return vertex(1,0,0);
		case 2: return vertex(0,1,0);
		case 3: return vertex(1,1,0);
		case 4: return vertex(0,0,1);
		case 5: return vertex(1,0,1);
		case 6: return vertex(0,1,1);
		case 7: return vertex(1,1,1);
		default:
			throw std::range_error("VoxelElementKey::vertex(int) - invalid vertex_number (" + std::to_string(vertex_number) + ")");
			return VoxelVertexKey{};
		}
	}

	constexpr VoxelFaceKey VoxelElementKey::face(const int face_number) const
	{
		switch (face_number) {
		case 0: return VoxelFaceKey{0, depth(), i()  , j()  , k()  };
		case 1: return VoxelFaceKey{1, depth(), i()  , j()  , k()  };
		case 2: return VoxelFaceKey{2, depth(), i()  , j()  , k()  };
		case 3: return VoxelFaceKey{0, depth(), i()+1, j()  , k()  };
		case 4: return VoxelFaceKey{1, depth(), i()  , j()+1, k()  };
		case 5: return VoxelFaceKey{2, depth(), i()  , j()  , k()+1};
		default:
			throw std::range_error("VoxelElementKey::face(int) - invalid face_number (" + std::to_string(face_number) + ")");
			return VoxelFaceKey{};
		}
	}

	/////////////////////////////////////////////////////////////////////////
	/// Adjacency operation definitions (Vertex)
	/////////////////////////////////////////////////////////////////////////
	constexpr VoxelElementKey VoxelVertexKey::element(const bool ii, const bool jj, const bool kk) const
	{
		return VoxelElementKey{
			depth(),
			i() + static_cast<uint64_t>(ii),
			j() + static_cast<uint64_t>(jj),
			k() + static_cast<uint64_t>(kk)
		};
	}

	constexpr VoxelElementKey VoxelVertexKey::element(const int element_number) const 
	{
		switch (element_number) {
		case 0: return element(0,0,0);
		case 1: return element(1,0,0);
		case 2: return element(0,1,0);
		case 3: return element(1,1,0);
		case 4: return element(0,0,1);
		case 5: return element(1,0,1);
		case 6: return element(0,1,1);
		case 7: return element(1,1,1);
		default:
			throw std::range_error("VoxelVertexKey::element(int) - invalid element_number (" + std::to_string(element_number) + ")");
			return VoxelElementKey{};
		}
	}

	constexpr VoxelFaceKey VoxelVertexKey::face(const int face_number) const
	{
		switch (face_number) {
		case 0 : return VoxelFaceKey{0, depth(), i(),   j(),   k()  };
		case 1 : return VoxelFaceKey{0, depth(), i(),   j()+1, k()  };
		case 2 : return VoxelFaceKey{0, depth(), i(),   j(),   k()+1};
		case 3 : return VoxelFaceKey{0, depth(), i(),   j()+1, k()+1};
		case 4 : return VoxelFaceKey{1, depth(), i(),   j(),   k()  };
		case 5 : return VoxelFaceKey{1, depth(), i()+1, j(),   k()  };
		case 6 : return VoxelFaceKey{1, depth(), i(),   j(),   k()+1};
		case 7 : return VoxelFaceKey{1, depth(), i()+1, j(),   k()+1};
		case 8 : return VoxelFaceKey{2, depth(), i(),   j(),   k()  };
		case 9 : return VoxelFaceKey{2, depth(), i()+1, j(),   k()  };
		case 10: return VoxelFaceKey{2, depth(), i(),   j()+1, k()  };
		case 11: return VoxelFaceKey{2, depth(), i()+1, j()+1, k()  };
		default:
			throw std::range_error("VoxelVertexKey::face(int) - invalid face_number (" + std::to_string(face_number) + ")");
			return VoxelFaceKey{};
		}
	}

	/////////////////////////////////////////////////////////////////////////
	/// Adjacency operation definitions (Face)
	/////////////////////////////////////////////////////////////////////////
	constexpr VoxelElementKey VoxelFaceKey::element(const bool forward_flag) const
	{
		switch (axis()) {
		case 0: return VoxelElementKey{depth(), i()+static_cast<uint64_t>(forward_flag), j(), k()};
		case 1: return VoxelElementKey{depth(), i(), j()+static_cast<uint64_t>(forward_flag), k()};
		case 2: return VoxelElementKey{depth(), i(), j(), k()+static_cast<uint64_t>(forward_flag)};
		}
		return VoxelElementKey{DOES_NOT_EXIST};
	}

	constexpr VoxelVertexKey VoxelFaceKey::vertex(const bool ii, const bool jj) const
	{
		switch (axis()) {
		case 0: return VoxelVertexKey{depth(), i(), j()+static_cast<uint64_t>(ii), k()+static_cast<uint64_t>(jj)};
		case 1: return VoxelVertexKey{depth(), i()+static_cast<uint64_t>(ii), j(), k()+static_cast<uint64_t>(jj)};
		case 2: return VoxelVertexKey{depth(), i()+static_cast<uint64_t>(ii), j()+static_cast<uint64_t>(jj), k()};
		}
		return VoxelVertexKey{DOES_NOT_EXIST};
	}

	constexpr VoxelVertexKey VoxelFaceKey::vertex(const int vertex_number) const
	{
		switch (vertex_number) {
		case 0: return vertex(0,0);
		case 1: return vertex(1,0);
		case 2: return vertex(0,1);
		case 3: return vertex(1,1);
		default:
			throw std::range_error("VoxelFaceKey::vertex(int) - invalid vertex_number (" + std::to_string(vertex_number) + ")");
			return VoxelVertexKey{};
		}
	}
}