#pragma once

#include "voxel_mesh/voxel_mesh_keys.hpp"
#include "util/quadrature_rules.hpp"

#include<span>
#include<omp.h>

#ifndef KERNEL_PROJECTION_PARALLEL_BASIS_THRESHOLD
#define KERNEL_PROJECTION_PARALLEL_BASIS_THRESHOLD 16
#endif


namespace gv::vmesh
{
	//A generic kernel type for assembline FEM matrices.
	//The matrix is assembled as M(i,j) = sum( kernel(el, phi[i], psi[j]) ) by default
	//but it can be specified to be symmetric and then uses M(i,j) = sum( kernel(el, phi[i], phi[j] ))
	//For the standard case, phi[i] is the i-th trial (solution space) basis function and psi[j] is the j-th
	//test space function. For the symmetric case, the kernel only needs to be evaluated when i>=j, but this
	//optimization is done outside this class.

	template<typename TrialDOF_t,
			 typename TestDOF_t,
			 bool IS_SYMMETRIC,
			 int  N_QUAD_POINTS_AXIS,
			 bool NEEDS_JACOBIAN,
			 typename DERRIVED>
	struct BaseKernel
	{
		explicit BaseKernel(double dx, double dy, double dz) : domain_diagonal{dx, dy, dz} {}

		std::span<const TrialDOF_t> current_trial_dofs;
		std::span<const TestDOF_t>  current_test_dofs; //unused when IS_SYMMETRIC=true
		
		VoxelElementKey current_element;

		//many kernels need this information.
		//updating this information can be disabled by setting NEEDS_JACOBIAN=false
		const double domain_diagonal[3]; //needed for computing jacobians
		double jac_diag[3]; //the jacobian matrix is diagonal on voxels
		double jac_det;

		//the template parameter N_QUAD_POINTS_AXIS is per-axis, so we need that many points cubed
		static constexpr int N_QUAD_POINT = N_QUAD_POINTS_AXIS*N_QUAD_POINTS_AXIS*N_QUAD_POINTS_AXIS;
		static constexpr std::array<double, N_QUAD_POINT> quad_x = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 0, double>(); //xcoordinates
		static constexpr std::array<double, N_QUAD_POINT> quad_y = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 1, double>(); //ycoordinates
		static constexpr std::array<double, N_QUAD_POINT> quad_z = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 2, double>(); //zcoordinates
		static constexpr std::array<double, N_QUAD_POINT> quad_w = gauss_legendre_cartesian_weight<N_QUAD_POINTS_AXIS, 3, double>(); //weights

		//store local matrix values and sizes and conveniently read them
		//the derived kernel must implement the storage
		int n_trial, m_test;
		std::vector<double> local_mat_values;
		inline int _linear_index(const int i, const int j) const {
			if constexpr (IS_SYMMETRIC) {
				return i <= j ? i + n_trial*j : j + n_trial*i;
			}
			else {
				return i + n_trial*j;
			}
		}

		inline double local_mat(const int i, const int j) const {return local_mat_values[_linear_index(i,j)];}
		inline double& local_mat(const int i, const int j) {return local_mat_values[_linear_index(i,j)];}

		//the derived class should call Base::setup during its own setup method.
		//the derived class setup method should ensure that the local matrix is ready to be computed
		void setup(VoxelElementKey el, std::span<const TrialDOF_t> trial_dofs, std::span<const TestDOF_t> test_dofs) requires (!IS_SYMMETRIC) {
			current_element = el;
			n_trial = trial_dofs.size();
			m_test  = test_dofs.size();
			local_mat_values.resize(n_trial*m_test);

			current_trial_dofs = trial_dofs;
			current_test_dofs = test_dofs;

			if constexpr (NEEDS_JACOBIAN) {
				//0.5*2^{-depth} includes the extra 0.5 for the jacobian
				const double scale = 0.5/static_cast<double>( uint64_t{1} << el.depth() );
				jac_diag[0] = domain_diagonal[0] * scale;
				jac_diag[1] = domain_diagonal[1] * scale;
				jac_diag[2] = domain_diagonal[2] * scale;
				jac_det     = jac_diag[0] * jac_diag[1] * jac_diag[2];
			}
		}

		void setup(VoxelElementKey el, std::span<const TrialDOF_t> trial_dofs) requires (IS_SYMMETRIC) {
			current_element = el;
			n_trial = trial_dofs.size();
			current_trial_dofs = trial_dofs;
			local_mat_values.resize(((n_trial+1)*n_trial)/2);

			if constexpr (NEEDS_JACOBIAN) {
				//0.5*2^{-depth} includes the extra 0.5 for the jacobian
				const double scale = 0.5/static_cast<double>( uint64_t{1} << el.depth() );
				jac_diag[0] = domain_diagonal[0] * scale;
				jac_diag[1] = domain_diagonal[1] * scale;
				jac_diag[2] = domain_diagonal[2] * scale;
				jac_det     = jac_diag[0] * jac_diag[1] * jac_diag[2];
			}
		}

		//project quadrature points into the support elements and get the support element for each basis function
		template<typename DOF_t> requires (std::is_same_v<DOF_t,TrialDOF_t> || std::is_same_v<DOF_t,TestDOF_t>)
		void project_to_support(
				std::span<const DOF_t> basis_funs,
				std::vector<VoxelElementKey>& supports,
				std::vector<std::array<double,N_QUAD_POINT>>& x,
				std::vector<std::array<double,N_QUAD_POINT>>& y,
				std::vector<std::array<double,N_QUAD_POINT>>& z)
		{
			supports.resize(basis_funs.size(), current_element);
			x.resize(basis_funs.size(), quad_x);
			y.resize(basis_funs.size(), quad_y);
			z.resize(basis_funs.size(), quad_z);

			#pragma omp parallel for if (basis_funs.size() > KERNEL_PROJECTION_PARALLEL_BASIS_THRESHOLD)
			for (size_t b=0; b<basis_funs.size(); ++b) {
				basis_funs[b].quad_elem2support_elem(supports[b], x[b], y[b], z[b]);
			}
		}
	};
}