#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh_keys.hpp"
#include "voxel_mesh/voxel_dof_base.hpp"
#include <array>
#include <cstdint>
#include <omp.h>

namespace gv::vmesh
{
	//Q1 DOFs
	struct CharmsVoxelQ1 : public VoxelDOFBase<VoxelVertexKey,CharmsVoxelQ1>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<VoxelVertexKey,CharmsVoxelQ1>;
		using Base::RefPoint_t;
		using Base::GeoPoint_t;

		//use the base constructors
		using Base::Base;

		//evaluate the basis function (convenience method)
		double eval(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			std::array<double,1> val;
			std::array<double,1> x{xi[0]};
			std::array<double,1> y{xi[1]};
			std::array<double,1> z{xi[2]};

			eval(val, support_elem, x, y, z);
			return val[0];
		}


		//compute the gradient of the basis function on the reference coordinate
		//the caller must scale the gradient to the mesh domain by
		//multiplying by the inverse transpose jacobian (i.e., divide by half the diagonal of the geometric support element)
		RefPoint_t grad(VoxelElementKey support_elem, const RefPoint_t& xi) const {
			std::array<double,1> gx, gy, gz;
			std::array<double,1> x(xi[0]);
			std::array<double,1> y(xi[1]);
			std::array<double,1> z(xi[2]);

			grad(gx, gy, gz, support_elem, x, y, z);
			return RefPoint_t{gx[0], gy[0], gz[0]};
		}

		//vectorized evaluation
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
			
			const bool bx = static_cast<bool>(key.i() - el.i());
			const bool by = static_cast<bool>(key.j() - el.j());
			const bool bz = static_cast<bool>(key.k() - el.k());

			const double sx = bx ? 1.0 : -1.0;
			const double sy = by ? 1.0 : -1.0;
			const double sz = bz ? 1.0 : -1.0;

			#pragma omp simd
			for (size_t q=0; q<N; ++q) {
				vl[q] = 0.125 * (1.0+sx*qx[q]) * (1.0+sy*qy[q]) * (1.0+sz*qz[q]);
			}
		}

		//vectorized grad
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
			
			const bool bx = static_cast<bool>(key.i() - el.i());
			const bool by = static_cast<bool>(key.j() - el.j());
			const bool bz = static_cast<bool>(key.k() - el.k());

			const double sx = bx ? 1.0 : -1.0;
			const double sy = by ? 1.0 : -1.0;
			const double sz = bz ? 1.0 : -1.0;

			#pragma omp simd
			for (size_t q=0; q<N; ++q) {
				const double fx = 1.0+sx*qx[q];
				const double fy = 1.0+sy*qy[q];
				const double fz = 1.0+sz*qz[q];

				gx[q] = 0.125 * sx * fy * fz;
				gy[q] = 0.125 * fx * sy * fz;
				gz[q] = 0.125 * fx * fy * sz;
			}
		}

		//charms refinement
		constexpr CharmsVoxelQ1 child_impl(const int i) const {
			assert(0<=i && i<27);
			const int z = i/9 - 1; //z=-1 plane, then z=0
			const int y = (i%9)/3 - 1;
			const int x = i%3 - 1;

			const uint64_t ci = (x<0) ? 2*key.i()-1 : (x==0) ? 2*key.i() : 2*key.i()+1;
			const uint64_t cj = (y<0) ? 2*key.j()-1 : (y==0) ? 2*key.j() : 2*key.j()+1;
			const uint64_t ck = (z<0) ? 2*key.k()-1 : (z==0) ? 2*key.k() : 2*key.k()+1;

			const VoxelVertexKey childkey(key.depth()+1, ci, cj, ck);
			return childkey.is_valid() ? CharmsVoxelQ1{childkey} : CharmsVoxelQ1{};
		}

		constexpr double child_coef_impl(const int i) const {
			assert(0<=i && i<27);
			//need the distance (0 or 1) to the child vertex in each coordinate direction
			//this gives us the reference coordinate of the child vertex
			const bool dz = static_cast<bool>(i/9 - 1); //z=-1 plane, then z=0
			const bool dy = static_cast<bool>((i%9)/3 - 1);
			const bool dx = static_cast<bool>(i%3 - 1);
			
			const double wx = dx? 0.5 : 1.0;
			const double wy = dy? 0.5 : 1.0;
			const double wz = dz? 0.5 : 1.0;

			return wx*wy*wz;
		}

		static constexpr int n_children_impl() {return 27;}
	};
}