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
		const Key_t key{};

		//constructors
		//keep a default so that std::vector<DOFBase>::resize() can be called.
		constexpr DOFBase() : key{} {}
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
		//it is the caller's responsibility to guarantee that support is actually a support element of the DOF
	};


	//P0 dofs (constant on each element)
	struct VoxelP0 : public DOFBase<VoxelElementKey>
	{
		//get types from the Base class
		using Base = DOFBase<VoxelElementKey>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate always to 1
		constexpr double eval(VoxelElementKey support_elem, const RefPoint_t& xi) {return 1.0;}

		//gradient is always 0
		constexpr RefPoint_t grad(VoxelElementKey support_elem, const RefPoint_t& xi) {return {0.0, 0.0, 0.0};}
	};

	//Q1 DOFs
	struct VoxelQ1 : public DOFBase<VoxelVertexKey>
	{
		//get types from the Base class
		using Base = DOFBase<VoxelVertexKey>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate the basis function
		constexpr double eval(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			//if the key vertex for the DOF is a vertex of the support_elem,
			//then key.i()>=support_elem.i(). Specifically, key.i()-suppoert_elem.i() is 0 or 1.
			//similar applies to j() and k(). The combination of those three give the local vertx.

			//check that the index logic is correct
			assert(support_elem.is_valid());
			assert(key.i() - support_elem.i() <= 1);
			assert(key.j() - support_elem.j() <= 1);
			assert(key.k() - support_elem.k() <= 1);
			
			const bool bi = static_cast<bool>(key.i() - support_elem.i());
			const bool bj = static_cast<bool>(key.j() - support_elem.j());
			const bool bk = static_cast<bool>(key.k() - support_elem.k());

			//high corner (evaluates to 2 at xi[0]=1 and 0 at xi[0]-1), low corner is opposite
			//similar logic applies to axis 1 and 2
			const double fi = bi ? (1.0+xi[0]) : (1.0-xi[0]);
			const double fj = bj ? (1.0+xi[1]) : (1.0-xi[1]);
			const double fk = bk ? (1.0+xi[2]) : (1.0-xi[2]);
			return (0.125*fi)*(fj*fk);
		}

		//compute the gradient of the basis function on the reference coordinate
		//the caller must scale the gradient to the mesh domain by
		//multiplying by the inverse transpose jacobian (i.e., divide by half the diagonal of the geometric support element)
		constexpr RefPoint_t grad(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			//if the key vertex for the DOF is a vertex of the support_elem,
			//then key.i()>=support_elem.i(). Specifically, key.i()-suppoert_elem.i() is 0 or 1.
			//similar applies to j() and k(). The combination of those three give the local vertx.

			//check that the index logic is correct
			assert(support_elem.is_valid());
			assert(key.i() - support_elem.i() <= 1);
			assert(key.j() - support_elem.j() <= 1);
			assert(key.k() - support_elem.k() <= 1);
			
			const bool bi = static_cast<bool>(key.i() - support_elem.i());
			const bool bj = static_cast<bool>(key.j() - support_elem.j());
			const bool bk = static_cast<bool>(key.k() - support_elem.k());

			//non-derivitive portions
			const double fi = bi ? (1.0+xi[0]) : (1.0-xi[0]);
			const double fj = bj ? (1.0+xi[1]) : (1.0-xi[1]);
			const double fk = bk ? (1.0+xi[2]) : (1.0-xi[2]);

			//derivative portions, include the (1/2)^3 factor
			const double di = bi ? 0.125 : -0.125;
			const double dj = bj ? 0.125 : -0.125;
			const double dk = bk ? 0.125 : -0.125;

			return RefPoint_t {
				di * fj * fk,
				fi * dj * fk,
				fi * fj * dk
			};
		}
	};


	struct VoxelRT0 : public DOFBase<VoxelFaceKey>
	{
		//get types from the Base class
		using Base = DOFBase<VoxelFaceKey>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate normal component. this is the same as the flux through the face.
		constexpr double flux(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			//similar to Q1, but only evaluate the coordinate in the normal direction
			//if the dof is normal to the x-axis and on the low side,
			//then we should return 1 when xi[0]=-1 and 0 when xi[0]=1.
			//similar reasoning applies to the other faces.
			const uint64_t a = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == support_elem.i()) :
									(a==1) ? (key.j() == support_elem.j()) :
											 (key.k() == support_elem.k());
			
			return low_face ? 0.5*(1.0 - xi[a]) : 0.5*(1.0 + xi[a]);
		}

		constexpr RefPoint_t eval(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			RefPoint_t result{0.0, 0.0, 0.0};
			result[a] = flux(support_elem, xi); //evaluate normal component
			return result;
		}

		constexpr double div(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == support_elem.i()) :
									(a==1) ? (key.j() == support_elem.j()) :
											 (key.k() == support_elem.k());
			return low_face ? -0.5 : 0.5;
		}
	};
}