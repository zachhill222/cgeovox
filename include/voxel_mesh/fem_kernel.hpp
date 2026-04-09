#pragma once

#include "voxel_mesh/voxel_mesh_keys.hpp"
#include "util/quadrature_rules.hpp"

#include<span>
#include<omp.h>

#ifndef KERNEL_PROJECTION_PARALLEL_BASIS_THRESHOLD
#define KERNEL_PROJECTION_PARALLEL_BASIS_THRESHOLD 64
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
		
		VoxelElementKey q_elem; //current quadrature element

		//many kernels need this information.
		//updating this information can be disabled by setting NEEDS_JACOBIAN=false
		const double domain_diagonal[3]; //needed for computing jacobians
		double jac_diag[3]; //the jacobian matrix is diagonal on voxels
		double jac_det;

		//the template parameter N_QUAD_POINTS_AXIS is per-axis, so we need that many points cubed
		static constexpr int N_QUAD_POINTS = N_QUAD_POINTS_AXIS*N_QUAD_POINTS_AXIS*N_QUAD_POINTS_AXIS;
		static constexpr std::array<double, N_QUAD_POINTS> quad_x = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 0, double>(); //xcoordinates
		static constexpr std::array<double, N_QUAD_POINTS> quad_y = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 1, double>(); //ycoordinates
		static constexpr std::array<double, N_QUAD_POINTS> quad_z = gauss_legendre_cartesian_coord_component<N_QUAD_POINTS_AXIS, 3, 2, double>(); //zcoordinates
		static constexpr std::array<double, N_QUAD_POINTS> quad_w = gauss_legendre_cartesian_weight<N_QUAD_POINTS_AXIS, 3, double>(); //weights

		//during a kernel call, we will need to convert the quadrature points from the quadrature element
		//to the reference coordinates in the support element for each basis function. when doing this
		//we can recover the support element for the basis function. this process only depends on the 
		//depth of the basis function support elments and not on the basis function itself. we can "sweep"
		//the computation from the depth of the quadrature element up to depth 0 one time rather than for each element.
		//note that the quadrature points are a cartesian grid on each support element, so we only need to compute the unique x,y,z values once
		std::array<std::array<double, N_QUAD_POINTS_AXIS>, MAX_DEPTH> p_qx, p_qy, p_qz; //projected quadrature points
		std::array<VoxelElementKey, MAX_DEPTH> s_el; //support element for each depth
		void project_to_support();

		//assemble the quadrature points at the specified depth to pass
		//to DOFs for vectorized evaluation
		void collect_quad_points(	const int d,
									std::array<double, N_QUAD_POINTS>& pd_qx,
									std::array<double, N_QUAD_POINTS>& pd_qy,
									std::array<double, N_QUAD_POINTS>& pd_qz) const;

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
			q_elem  = el;
			n_trial = trial_dofs.size();
			m_test  = test_dofs.size();
			local_mat_values.resize(n_trial*m_test);

			current_trial_dofs = trial_dofs;
			current_test_dofs = test_dofs;

			//compute the jacobian
			if constexpr (NEEDS_JACOBIAN) {
				//0.5*2^{-depth} includes the extra 0.5 for the jacobian
				const double scale = 0.5/static_cast<double>( uint64_t{1} << el.depth() );
				jac_diag[0] = domain_diagonal[0] * scale;
				jac_diag[1] = domain_diagonal[1] * scale;
				jac_diag[2] = domain_diagonal[2] * scale;
				jac_det     = jac_diag[0] * jac_diag[1] * jac_diag[2];
			}

			//project the quadrature points to the correct support elements
			project_to_support();
		}

		void setup(VoxelElementKey el, std::span<const TrialDOF_t> trial_dofs) requires (IS_SYMMETRIC) {
			q_elem  = el;
			n_trial = trial_dofs.size();
			current_trial_dofs = trial_dofs;
			local_mat_values.resize(((n_trial+1)*n_trial)/2);

			//compute the jacobian
			if constexpr (NEEDS_JACOBIAN) {
				//0.5*2^{-depth} includes the extra 0.5 for the jacobian
				const double scale = 0.5/static_cast<double>( uint64_t{1} << el.depth() );
				jac_diag[0] = domain_diagonal[0] * scale;
				jac_diag[1] = domain_diagonal[1] * scale;
				jac_diag[2] = domain_diagonal[2] * scale;
				jac_det     = jac_diag[0] * jac_diag[1] * jac_diag[2];
			}

			//project the quadrature points to the correct support elements
			project_to_support();
		}
	};


	template<typename TrialDOF_t,
			 typename TestDOF_t,
			 bool IS_SYMMETRIC,
			 int  N_QUAD_POINTS_AXIS,
			 bool NEEDS_JACOBIAN,
			 typename DERRIVED>
	void BaseKernel::project_to_support()
	{
		const uint64_t md = q_elem.depth(); //max starting depth, work up
		p_qx[md] = quad_x;
		p_qy[md] = quad_y;
		p_qz[md] = quad_z;
		s_el[md] = q_elem;

		for (uint64_t d = md; d>0; --d) {
			//determine if the element is an even/odd child in each coordinate
			//and compute the increment to shift each coordinate
			const double dx = static_cast<double>(s_el[d].i() & 1) - 0.5;
			const double dy = static_cast<double>(s_el[d].j() & 1) - 0.5;
			const double dz = static_cast<double>(s_el[d].k() & 1) - 0.5;

			#pragma omp simd
			for (int q=0; q<N_QUAD_POINTS_AXIS; ++q) {
				p_qx[d-1][q] = 0.5*p_qx[d][q] + dx;
				p_qy[d-1][q] = 0.5*p_qy[d][q] + dy;
				p_qz[d-1][q] = 0.5*p_qz[d][q] + dz;
			}

			s_el[d-1] = s_el[d].parent();
		}
	}

	template<typename TrialDOF_t,
			 typename TestDOF_t,
			 bool IS_SYMMETRIC,
			 int  N_QUAD_POINTS_AXIS,
			 bool NEEDS_JACOBIAN,
			 typename DERRIVED>
	void BaseKernel::collect_quad_points(const int d,
									std::array<double, N_QUAD_POINTS>& pd_qx,
									std::array<double, N_QUAD_POINTS>& pd_qy,
									std::array<double, N_QUAD_POINTS>& pd_qz) const
	{
		for (int k=0; k<N_QUAD_POINTS_AXIS; ++k) {
			for (int j=0; j<N_QUAD_POINTS_AXIS; ++j) {
				for (int i=0; i<N_QUAD_POINTS_AXIS; ++i) {
					const int l = i + N_QUAD_POINTS_AXIS*(j + N_QUAD_POINTS_AXIS*k);
					pd_qx[l] = p_qx[d][i];
					pd_qy[l] = p_qy[d][j];
					pd_qz[l] = p_qz[d][k];
				}
			}
		}
	}
}