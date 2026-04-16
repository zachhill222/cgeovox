#pragma once

#include "voxel_mesh/fem/bilinear_form.hpp"

#include <array>
#include <cstdint>
#include <omp.h>

namespace gv::vmesh
{
	//A symmetric mass kernel for bilinear forms a(psi,phi) = integral_D phi*psi
	template<typename DOF_type>
	struct SymmetricMassForm : public BilinearForm<DOF_type,DOF_type,true>
	{
		using BASE = BilinearForm<DOF_type,DOF_type,true>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using DOF_t = DOF_type;
		
		using BASE::BASE;

		//only provides the evaluation
		//should be vectorized with simd
		template<uint64_t N>
		static constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const DOF_t psi_i,
			const QuadElem_t spt_i,
			const std::array<double,N>& X_i,
			const std::array<double,N>& Y_i,
			const std::array<double,N>& Z_i, 
			const DOF_t phi_j,
			const QuadElem_t spt_j,
			const std::array<double,N>& X_j,
			const std::array<double,N>& Y_j,
			const std::array<double,N>& Z_j)
		{
			std::array<double,N> psi_i_vals;
			psi_i.eval(psi_i_vals, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_vals;
			phi_j.eval(phi_j_vals, spt_j, X_j, Y_j, Z_j);

			const double jac_det = Jxx*Jyy*Jzz;

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = psi_i_vals[i]*phi_j_vals[i]*jac_det;
			}
		}
	};

}
