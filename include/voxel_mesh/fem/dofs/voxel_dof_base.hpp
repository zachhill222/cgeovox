#pragma once

#include "gutil.hpp"
#include "voxel_mesh/mesh/keys/voxel_key.hpp"

#include <array>
#include <cstdint>
#include <omp.h>

namespace gv::vmesh
{
	//base DOF class to provide some utility methods.
	//evaulation, storage of support elments and such must be done in the derived classes.
	template<VoxelKeyType Key_type, typename Derrived = void>
	struct VoxelDOFBase
	{
		static constexpr bool IS_REFINABLE = !std::is_same_v<Derrived,void>;

		//type aliases to distinguish reference coordinates and geometric coordinate
		//purely for logical aid.
		using RefPoint_t = gutil::Point<3,double>;
		using GeoPoint_t = gutil::Point<3,double>;
		using Key_t      = Key_type;
		using QuadElem_t = VoxelElementKey<Key_t::I_W, Key_t::MORTON>;

		//store the logical key where this element lives
		//note that this key does not need to correspond to an active feature of the mesh
		Key_t key{};

		inline constexpr bool exists() const {return key.exists();}
		inline constexpr bool is_valid() const {return key.is_valid();}
		inline constexpr bool operator==(const Derrived other) const {return key==other.key;}
		inline constexpr auto operator<=>(const Derrived other) const {return key<=>other.key;}
		inline constexpr uint64_t linear_index() const {return key.linear_index();}

		//constructors
		//keep a default so that std::vector<VoxelDOFBase>::resize() can be called.
		constexpr VoxelDOFBase() : key{} {}
		constexpr VoxelDOFBase(Key_t k) : key(k) {}

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

		//we will often need to check the basis function's support elements
		template<bool PERIODIC=false>
		inline constexpr auto support() const {
			return static_cast<const Derrived*>(this) -> template support_impl<PERIODIC>();
		}

		constexpr int n_dof_per_elem() const {
			return static_cast<const Derrived*>(this) -> N_DOF_PER_ELEM;
		}

		constexpr auto dofs_on_elem(const QuadElem_t el) const {
			return static_cast<const Derrived*>(this) -> dofs_on_elem_impl(el);
		}

		//charms specific interface. only here to enforce conformity
		inline constexpr auto children() const requires(IS_REFINABLE) {
			return static_cast<const Derrived*>(this) -> children_impl();
		}

		constexpr auto children_coef() const requires(IS_REFINABLE) {
			return static_cast<const Derrived*>(this) -> children_coef_impl();
		}

		constexpr int n_children() const requires(IS_REFINABLE) {
			return static_cast<const Derrived*>(this) -> N_CHILDREN; //n_children_impl() should be static
		}

		constexpr auto parents() const requires(IS_REFINABLE) {
			return static_cast<const Derrived*>(this) -> parents_impl();
		}

		constexpr auto parent_coefs() const requires(IS_REFINABLE) {
			return static_cast<const Derrived*>(this) -> parent_coefs_impl();
		}

		//project a reference point on a deeper quadrature element to a support element
		//use a vectorized version for quadrature (in the kernel class)
		constexpr void proj_to_support(QuadElem_t& spt, RefPoint_t& pt) const {
			const uint64_t md = spt.depth(); //max starting depth, work up
			for (uint64_t d = md; d>key.depth(); --d) {
				//determine if the element is an even/odd child in each coordinate
				//and compute the increment to shift each coordinate
				const double dx = static_cast<double>(spt.i() & 1) - 0.5;
				const double dy = static_cast<double>(spt.j() & 1) - 0.5;
				const double dz = static_cast<double>(spt.k() & 1) - 0.5;

				pt[0] = 0.5*pt[0] + dx;
				pt[1] = 0.5*pt[1] + dy;
				pt[2] = 0.5*pt[2] + dz;

				spt = spt.parent();
			}
			assert(spt.depth()==key.depth());
		}
	};
}