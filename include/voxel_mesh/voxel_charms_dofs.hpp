#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh_keys.hpp"

#include <cstdint>

namespace gv::vmesh
{
	//base DOF class to provide some utility methods.
	//evaulation, storage of support elments and such must be done in the derived classes.
	template<typename Key_t>
	requires std::is_same_v<Key_t,VoxelElementKey> || std::is_same_v<Key_t,VoxelFaceKey> || std::is_same_v<Key_t,VoxelVertexKey>
	struct DOFBase
	{
		//type aliases to distinguish reference coordinates and geometric coordinate
		//purely for logical aid.
		using RefPoint_t = gutil::Point<3,double>;
		using GeoPoint_t = gutil::Point<3,double>;

		//store the logical key where this element lives
		//note that this key does not need to correspond to an active feature of the mesh
		const Key_t key;

		constexpr DOFBase(Key_t k) : key(k) {}

		//xi are normalized coordinates to quad_elem
		//each component of xi is in [-1,1]
		//quad_elem is a descendant of some support element of this basis function
		//the support element and the reference point of xi in the support element will be found here.
		//the updated reference point in the support element and the initial reference point in the quadrature element correspond
		//to the same geometric point in the domain.
		//the quad_elem will be changed into the correct support element
		static void quad_elem2support_elem(VoxelElementKey& quad_elem, const uint64_t support_depth, RefPoint_t& xi) {
			assert(quad_elem.depth() >= support_depth);

			while (quad_elem.depth() > support_depth) {
				//how to map upwards depends on which child this is
				//this can be recovered from the least significant bit of its i,j,k indices
				//if bi=0, then we are in the [-1,0] half. if bi=1, we are in the [0,1] half.
				//the same applies to bj and bk.
				const double bi = static_cast<double>(quad_elem.i() & 1);
				const double bj = static_cast<double>(quad_elem.j() & 1);
				const double bk = static_cast<double>(quad_elem.k() & 1);

				//note the map is affine in each coordinate.
				//if xi[0]=-1, then it should be -1 or 0 in the parent element (depending on bi).
				//the same applies to xi[1] and xi[2]. also consider xi[*]=0.
				xi[0] = 0.5*xi[0] + bi - 0.5;
				xi[1] = 0.5*xi[1] + bj - 0.5;
				xi[2] = 0.5*xi[2] + bk - 0.5;

				quad_elem = quad_elem.parent();
			}
		}

		//xi are normalized coordinates to quad_elem
		//each component of xi is in [-1,1]
		//these are evaluated in the reference element.
		//in particular, the gradient will need to be scaled by the jacobian.
		//some DOF types might return a vector or matrix, so use auto here.
		//the return type must be specified in the Derived class.
		
		//the interface for any needed operations must follow the form
		// eval(VoxelElementKey support, const RefPoint_t& quad_point)
		//where the support element is a natural support element at the same depth as the key
		//and quad_point is a reference coordinate in [-1,1]^3
		//these values will be the result of calling quad_elem2support_elem.
	};
}