#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh_keys.hpp"
#include "voxel_mesh/voxel_dof_base.hpp"
#include <array>
#include <cstdint>
#include <omp.h>

namespace gv::vmesh
{
	//P0 dofs (constant on each element)
	struct CharmsVoxelP0 : public VoxelDOFBase<VoxelElementKey, CharmsVoxelP0>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<VoxelElementKey, CharmsVoxelP0>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate always to 1
		static constexpr double eval(VoxelElementKey support_elem, const RefPoint_t& xi) {return 1.0;}

		//gradient is always 0
		static constexpr RefPoint_t grad(VoxelElementKey support_elem, const RefPoint_t& xi) {return {0.0, 0.0, 0.0};}

		//vectorized evaluation (for a consistent interface across dof types)
		template<int N> requires (N>0)
		void eval(	std::array<double,N>&       vl, //values 
					VoxelElementKey             el, //support element
					const std::array<double,N>& qx, //reference/quadrature points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {
			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			//constant evaluation
			vl.fill(1.0);
		}

		//vectorized grad (for a consistent interface across dof types)
		template<int N> requires (N>0)
		void grad(	std::array<double,N>&		gx, //gradient result
					std::array<double,N>& 		gy, 
					std::array<double,N>& 		gz, 
					VoxelElementKey  	 		el, //support element
					const std::array<double,N>& qx, //reference/quadratrue points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {

			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			//constant evaluation
			gx.fill(0.0);
			gy.fill(0.0);
			gz.fill(0.0);
		}

		//refinement operations
		constexpr CharmsVoxelP0 child_impl(const int i) const {return CharmsVoxelP0{key.child(i)};}
		constexpr double child_coef_impl(const int i) const {return 1.0;}
		static constexpr int n_children_impl() {return 8;}
	};
}