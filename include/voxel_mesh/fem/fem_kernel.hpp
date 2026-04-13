#pragma once

#include "voxel_mesh/mesh/keys/voxel_key.hpp"
#include "util/quadrature_rules.hpp"

#include<type_traits>
#include<cstdlib>
#include<tuple>
#include<span>
#include<omp.h>

#ifndef KERNEL_OMP__BASIS_THRESHOLD
#define KERNEL_OMP__BASIS_THRESHOLD 64
#endif


namespace gv::vmesh
{
	//A generic kernel type for assembline FEM matrices.
	//This class will construct one or more local matricies (e.g., mass/stiffness)
	//over a single quadrature element of m trial basis functions (solution dofs) against
	//n test basis functions (also dofs, but can be of a different type than the trial).
	//the result is an n-by-m local matrix (stored as a vector).
	//This is designed for use in the CHARMS method, so m and n are unknown at compile time.
	//The H in CHARMS stands for Hierarchical. For each quadrature point in the quadrature element,
	//we must `project' it to the reference coordinate in the support element for each basis function.
	//This only needs to be done once for each unique depth of the basis functions (in the mesh/function space hierarchy)
	//To avoid doing this computation (and to avoid unnecessary looping through the mesh elements) more than necessary,
	//multiple evaluation methods can be passed to the kernel. One local matrix is produced per evaluation method.
	//The evaluation method must take the signature
	//
	//		static constexpr void eval(val,quad, psi_i,spt_i,X_i,Y_i,Z_i, phi_j,spt_j,X_j,Y_j,Z_j);
	//
	//Where val is a reference to an (un-initialized) array<double,NQ> to store the values, quad is the quadrature element
	//psi_i is the test dof, phi_j is the trial dof, X_*, Y_*, Z_* are const references to an array<double,NQ>
	//that record the reference coordinate of basis psi_i or phi_j on its support element spt_*.
	//The DOF types should provide vectorized methods for eval or grad as needed with a similar signature.
	//
	//For a bilinear form integral_D a(psi, phi) dx, the corresponding entry in the global matrix is 
	//		A_IJ = integral_D a(psi_I,phi_J) = sum_E integral_E a(psi_I,phi_J)
	//where D is the problem domain, a(*,*) is the bilinear form, and E is an element of the mesh of D.
	//The local matrix for element E is then
	//      a_ij = integral_E a(psi_I, phi_J) ~ sum_q a(psi_i, phi_j, q) * w_q
	//where i and j are local dof numbers corresponding to the global numbers I and J, q is a quadrature point
	//with w(q) its corresponding quadrature weight. For hierarchical methods, it is more feasible to convert
	//the quadrature points to the correct reference points for each function psi_i and phi_j ahead of time.
	//the supplied eval(*) method described above is responsible for evaluating a(psi_i, phi_j, q) at each of the
	//supplied quadrature points. The values are then reduced with multiplication with the wieights into the local a_ij entry.
	//
	//Note that it is the responsibility of the evaluator method to correctly handle the jacobian.
	//The jacobian depends only on the mapping from the reference element to the quadrature element.
	//For an octree voxel mesh, this depends only on the depth of the quadrature element and the dimensions of the domain.

	//information required for each evaluation method.
	//bilinear forms will be constructed and passed to the kernel. the kernel will handle dispatching evaluations
	//to accumulate the local matrix for each bilinear form. The storage of the local matrix and logic for accessing values
	//is stored in the bilinear form, as this changes if the form is symmetric or not.
	//additionally, based on boundary conditions, the bilinear form may be responsible for applying boundary condions to the
	//local matrix after it is assembled by the kernel (with 'natural' BC).
	//for better convenience when applying BC as a post processing step, the full local matrix is stored, even in the symmetric case.
	template<typename TrialDOF_type,
			 typename TestDOF_type,
			 bool IS_SYMMETRIC_=false>
	struct BilinearForm {
		using TestDOF_t  = TestDOF_type;
		using TrialDOF_t = TrialDOF_type;
		static constexpr bool IS_SYMMETRIC = IS_SYMMETRIC_;
		static_assert(!IS_SYMMETRIC || (IS_SYMMETRIC && std::same_as<TrialDOF_type, TestDOF_type>),
			"BilinearForm - The test and trial spaces/dofs must be the same for a symmetric bilinear form.");

		using QuadElem_t = typename TrialDOF_t::QuadElem_t::NonPeriodicType;
		static_assert(std::same_as<QuadElem_t, typename TestDOF_t::QuadElem_t::NonPeriodicType>,
			"BilinearForm - The test and trial spaces/dofs must have compatible quadrature elements.");

		std::vector<double>      	loc_m_v; 	//local matrix values (n_test by m_trial)
		std::span<const TestDOF_t>  test_dofs;	//local test basis functions (row dofs)
		std::span<const TrialDOF_t> trial_dofs; 	//local trial basis functions (column dofs)

		uint64_t n_test, m_trial;

		void set_basis(const std::vector<TestDOF_t>& test_dofs_, const std::vector<TrialDOF_t>& trial_dofs_) requires (!IS_SYMMETRIC) {
			trial_dofs = trial_dofs_;
			test_dofs  = test_dofs_;
			n_test  = test_dofs.size();
			m_trial = trial_dofs.size();
			loc_m_v.resize(n_test*m_trial, 0.0);
		}

		void set_basis(const std::vector<TestDOF_t>& test_dofs_, const std::vector<TrialDOF_t>& trial_dofs_) requires (IS_SYMMETRIC) {
			trial_dofs = trial_dofs_;
			test_dofs = trial_dofs_;
			m_trial = trial_dofs.size();
			n_test  = m_trial;
			loc_m_v.resize(n_test*m_trial, 0.0);
		}

		inline double& local_mat(uint64_t i, uint64_t j) {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_m_v[j + i*m_trial]; //row-major is better for BC setting
		}

		inline double local_mat(uint64_t i, uint64_t j) const {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_m_v[j + i*m_trial]; //row-major is better for BC setting
		}
	};



	//A class to handle mapping of the quadrature points to each element
	template<uint64_t N_QUAD_POINTS, VoxelElementKeyType QuadElem_t>
	struct QuadPointMap
	{
		//standard axis values
		static constexpr uint64_t NQ_A = N_QUAD_POINTS;  //number of quadrature points along each axis
		static constexpr uint64_t NQ   = NQ_A*NQ_A*NQ_A; //total number of quadrature points
		static constexpr uint64_t MAX_DEPTH = QuadElem_t::MAX_DEPTH; //maximum depth that we may need to project points to
		static constexpr std::array<double, NQ_A> quad_x = gauss_legendre_cartesian_coord_component<NQ_A, 3, 0, double>(); //xcoordinates
		static constexpr std::array<double, NQ_A> quad_y = gauss_legendre_cartesian_coord_component<NQ_A, 3, 1, double>(); //ycoordinates
		static constexpr std::array<double, NQ_A> quad_z = gauss_legendre_cartesian_coord_component<NQ_A, 3, 2, double>(); //zcoordinates

		//during a kernel call, we will need to convert the quadrature points from the quadrature element
		//to the reference coordinates in the support element for each basis function. when doing this
		//we can recover the support element for the basis function. this process only depends on the 
		//depth of the basis function support elments and not on the basis function itself. we can "sweep"
		//the computation from the depth of the quadrature element up to depth 0 one time rather than for each element.
		//note that the quadrature points are a cartesian grid on each support element, so we only need to compute the unique x,y,z values once
		
		static constexpr std::array<double, NQ> p_qw = gauss_legendre_cartesian_weight<NQ_A, 3, double>(); //quadrature weight at each point.
		std::array<std::array<double, NQ_A>, MAX_DEPTH> p_qx, p_qy, p_qz; //projected quadrature points
		std::array<QuadElem_t, MAX_DEPTH> s_el; //support element for each depth

		constexpr void project_to_support(
			const QuadElem_t q_elem,	//starting element (the quadrature element)
			const uint64_t dd)			//desired depth (the depth of the dof)
		{

			const uint64_t md = q_elem.depth(); //max starting depth, work to depth 0
			p_qx[md] = quad_x;
			p_qy[md] = quad_y;
			p_qz[md] = quad_z;
			s_el[md] = q_elem;

			for (uint64_t dd = md; dd>0; --dd) {
				//determine if the element is an even/odd child in each coordinate
				//and compute the increment to shift each coordinate
				const double dx = static_cast<double>(s_el[d].i() & 1) - 0.5;
				const double dy = static_cast<double>(s_el[d].j() & 1) - 0.5;
				const double dz = static_cast<double>(s_el[d].k() & 1) - 0.5;

				#pragma omp simd
				for (int q=0; q<NQ_A; ++q) {
					p_qx[dd-1][q] = 0.5*p_qx[dd][q] + dx;
					p_qy[dd-1][q] = 0.5*p_qy[dd][q] + dy;
					p_qz[dd-1][q] = 0.5*p_qz[dd][q] + dz;
				}

				s_el[dd-1] = s_el[dd].parent();
			}
		}

		//before passing to the dofs, we need to assemble the x/y/z coordinates together
		//the quadrature points are always a NQ_A x NQ_A x NQ_A cartesian grid with the grid locations
		//at each depth stored in p_q* for each x,y,z axis. This routine takes their axial values
		//and produces all NQ_A^3 points, split into x,y,z components.
		//the linear index of the (i,j,k) point is
		//
		//		l = i + NQ_A*(j + NQ_A*k) = i + j*NQ_A + k*NQ_A^2
		//
		constexpr void collect_quad_points(
			const uint64_t dd,
			std::array<double, NQ>& pd_qx, 			//result x-component
			std::array<double, NQ>& pd_qy,			//result y-component
			std::array<double, NQ>& pd_qz) const 	//result z-component
		{
			for (uint64_t k=0; k<NQ_A; ++k) {
				for (uint64_t j=0; j<NQ_A; ++j) {
					for (uint64_t i=0; i<NQ_A; ++i) {
						const uint64_t l = i + NQ_A*(j + NQ_A*k);
						pd_qx[l] = p_qx[dd][i];
						pd_qy[l] = p_qy[dd][j];
						pd_qz[l] = p_qz[dd][k];
					}
				}
			}
		}
	};


	//Kernel class allows multiple interactions (bilinear forms) to be integrated simultaneously
	//The bilinear forms in the kernel are allowed to have different dof types
	//but they must all have compatable quadrature element types (i.e., voxel elements with the same maximum depth)
	//the kernel accepts vectors of dofs (essentially uint64_t with additional logic), a quadrature element
	//and then organizes the projection of the quadrature points from the quadrature element into the support elements
	//of the dofs. Then it dispatches the dofs to appropriate bilinear forms and accumulates the results against the
	//quadrature weights into the local matrix.
	template<uint64_t N_QUAD_POINTS, typename... BilinearForm_ts>
	struct Kernel
	{
		//organize forms and collect types
		static constexpr uint64_t N_FORMS = sizeof...(BilinearForm_ts);
		static_assert(N_FORMS>0, "Kernel - no bilenar form was provided");

		template<uint64_t I>
		using Form = std::tuple_element_t<I, std::tuple<BilinearForm_ts...>>;
		
		template<uint64_t I>
		using TrialDOF_t = typename Form<I>::TrialDOF_t;

		template<uint64_t I>
		using TestDOF_t = typename Form<I>::TestDOF_t;

		using QuadElem_t = typename Form<0>::QuadElem_t;

		//validity checks
		static_assert((std::same_as<typename BilinearForm_ts::QuadElem_t, QuadElem_t> && ...),
			"Kernel - all BilinearForms must share the same type of quadrature element (QuadElem_t).");

		static_assert(can_eval_all(), "Kernel - all BilinearForms must have a static constexpr eval() method with the signature.");
			

		//access individual bilinear forms
		template<int I> requires (I<N_FORMS)
		auto& form() {return std::get<I>(forms);}

		template<int I> requires (I<N_FORMS)
		const auto& form() const {return std::get<I>(forms);}

		//interface to use in the element loop
		template<uint64_t I> requires (I<N_FORMS)
		inline void set_basis(const std::vector<TestDOF_t<I>>& test_dofs, const std::vector<TrialDOF_t<I>>& trial_dofs) {
			form<I>().set_basis(test_dofs,trial_dofs);
		}

		inline set_element(QuadElem_t el) {
			q_map.project_to_support(el);
			q_elem = el;
		}

		template<uint64_t I> requires (I<N_FORMS)
		void compute();

		private:
		//total number of quadrature points
		static constexpr NQ = QuadPointMap<N_QUAD_POINTS,QuadElem_t>::NQ;

		//instantiate copies of each bilinear form
		//these hold the local matrix values and methods to compute the contributions at each quadrature point
		std::tuple<BilinearForm_ts...> forms;

		//container to handle projecting from the quadrature element to the support elements
		QuadPointMap<N_QUAD_POINTS,QuadElem_t> q_map;
		QuadElem_t q_elem; //current quadrature element (pass to bilinear forms for jacobian)

		//template logic to make sure that the kernels can be evaluated
		template<uint64_t I> //what needs to be checked
		static constexpr bool can_evaluate() {
			return std::is_invocable_v<
				decltype(&Form<I>::eval),
				std::array<double, NQ>&,
				const QuadElem_t&,
				const typename Form<I>::TestDOF_t&,
				const QuadElem_t&,
				const std::array<double, NQ>&,
				const std::array<double, NQ>&,
				const std::array<double, NQ>&,
				const typename Form<I>::TrialDOF_t&,
				const QuadElem_t&,
				const std::array<double, NQ>&,
				const std::array<double, NQ>&,
				const std::array<double, NQ>&
			>;
		}

		template<uint64_t... Is> //how to check
		static constexpr bool can_eval_check_(std::index_sequence<Is...>) {
			return (can_evaluate<Is>() && ...);
		}

		static constexpr bool can_eval_all() {
			return can_eval_check_(std::make_index_sequence<N_FORMS>{});
		}

	};
	

	template<uint64_t N_QUAD_POINTS, typename... BilinearForm_ts>
	template<uint64_t I> requires (I<Kernel<N_QUAD_POINTS,BilinearForm_ts...>::N_FORMS)
	void Kernel<N_QUAD_POINTS,BilinearForm_ts...>::compute()
	{
		const uint64_t m_trial=form<I>().m_trial;
		const uint64_t n_test=form<I>().n_test;

		#pragma omp parallel if(n_test*m_trial > KERNEL_OMP__BASIS_THRESHOLD)
		{
			std::array<double,NQ> vals;
			#pragma omp for
			for (uint64_t j=0; j<m_trial; ++j) {
				const auto phi_j = form<I>().trial_dofs[j];
				const uint64_t depth_j = phi_j.depth();
				const QuadElem_t spt_j = q_map.s_el[depth_j];
				const auto& X_j = q_map.p_qx[depth_j];
				const auto& Y_j = q_map.p_qy[depth_j];
				const auto& Z_j = q_map.p_qz[depth_j];

				for (uint64_t i=0; i<n_test; ++i) {
					if constexpr (Form<I>::IS_SYMMETRIC) {if (i<j) {continue;}}

					const auto psi_i = form<I>().test_dofs[i];
					const uint64_t depth_i = psi_i.depth();
					const QuadElem_t spt_i = q_map.s_el[depth_i];
					const auto& X_i = q_map.p_qx[depth_i];
					const auto& Y_i = q_map.p_qy[depth_i];
					const auto& Z_i = q_map.p_qz[depth_i];

					Form<I>::eval(vals, q_elem,
							psi_i, spt_i, X_i, Y_i, Z_i,
							phi_j, spt_j, X_j, Y_j, Z_j);

					double v_ij = 0.0;
					#pragma omp simd reduction(+:v_ij)
					for (uint64_t l=0; l<NQ; ++l) {
						v_ij += vals[l] * q_map.p_qw[l];
					}

					form<I>().local_mat(i,j) = v_ij;
					if constexpr (Form<I>::IS_SYMMETRIC) {form<I>().local_mat(j,i) = v_ij;}
				}
			}
		}
	}
}