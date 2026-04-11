#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh_keys.hpp"
#include "voxel_mesh/voxel_dof_base.hpp"
#include <array>
#include <cstdint>
#include <omp.h>

namespace gv::vmesh
{
	struct CharmsVoxelRT0 : public VoxelDOFBase<VoxelFaceKey,CharmsVoxelRT0>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<VoxelFaceKey,CharmsVoxelRT0>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate normal component. this is the same as the flux through the face.
		//use the vectorized version. this is only for debugging.
		constexpr double flux(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			//similar to Q1, but only evaluate the coordinate in the normal direction
			//if the dof is normal to the x-axis and on the low side,
			//then we should return 1 when xi[0]=-1 and 0 when xi[0]=1.
			//similar reasoning applies to the other faces.
			const uint64_t a = key.axis();
			std::array<double,1> vl;
			std::array<double,1> nx{xi[a]};
			flux(vl, support_elem, nx);
			return vl[0];
		}

		//use the vectorized flux. this is only for debugging.
		constexpr RefPoint_t eval(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			RefPoint_t result{0.0, 0.0, 0.0};

			result[a] = flux(support_elem, xi); //evaluate normal component
			return result;
		}

		//use the vectorized version. this is only for debugging.
		constexpr double div(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == support_elem.i()) :
									(a==1) ? (key.j() == support_elem.j()) :
											 (key.k() == support_elem.k());
			return low_face ? -0.5 : 0.5;
		}

		//vectorized flux
		template<int N> requires (N>0)
		void flux(	std::array<double,N>&       vl, //values 
					VoxelElementKey             el, //support element
					const std::array<double,N>& nx //reference/quadrature points (normal component)
					) const {
			
			const uint64_t a    = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == el.i()) :
									(a==1) ? (key.j() == el.j()) :
											 (key.k() == el.k());
			const bool s = low_face ? -0.5 : 0.5;
			#pragma omp simd
			for (int q=0; q<N; ++q) {
				vl[q] = 0.5 + s*nx[q];
			}
		}

		//vectorized div
		template<int N> requires (N>0)
		void div(	std::array<double,N>&       vl, //values 
					VoxelElementKey             el, //support element
					) const {
			
			const uint64_t a    = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == el.i()) :
									(a==1) ? (key.j() == el.j()) :
											 (key.k() == el.k());
			
			if (low_face) {vl.fill(-0.5);}
			else {vl.fill(0.5);}
		}
		

		constexpr CharmsVoxelRT0 child_impl(const int i) const {
			assert(i>=0 && i<12);
			const uint64_t a = key.axis();
			const VoxelFaceKey tk = key.child(i%4);
			const uint64_t da = i<4 ? 2 : i>=8 ? 1 : 0; 

			const uint64_t ci = (a!=0) ? tk.i() : (da==2) ? tk.i()-1 : tk.i() + da;
			const uint64_t cj = (a!=1) ? tk.j() : (da==2) ? tk.j()-1 : tk.j() + da;
			const uint64_t ck = (a!=2) ? tk.k() : (da==2) ? tk.k()-1 : tk.k() + da;

			const VoxelFaceKey childkey(a, tk.depth(), ci, cj, ck);
			return childkey.is_valid() ? CharmsVoxelRT0{childkey} : CharmsVoxelRT0{};
		}

		constexpr double child_coef_impl(const int i) const {
			assert(i>=0 && i<12);
			//re-ordering to call i<8 ? 0.5 : 1.0 is nice here, but make an awkward child dof ordering
			return (i<4 || i>=8) ? 0.5 : 1.0;
		}

		static constexpr int n_children_impl() {return 12;}
	};
}