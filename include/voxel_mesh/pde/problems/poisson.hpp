#pragma once

#include "gutil.hpp"

#include "voxel_mesh/fem/dofhandler.hpp"
#include "voxel_mesh/fem/dofs/voxel_dof_Q1.hpp"
#include "voxel_mesh/fem/kernel.hpp"
#include "voxel_mesh/mesh/voxel_mesh.hpp"
#include "voxel_mesh/pde/forms/L2_inner.hpp"
#include "voxel_mesh/pde/forms/H1_semi.hpp"
#include "util/log_time.hpp"
#include "util/concepts.hpp"

#include <Eigen/SparseCore>
// #include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include <array>
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{

	template<int BC=0, int MAX_DEPTH=8>
	struct PoissonQ1
	{
		using Mesh_t    = HierarchicalVoxelMesh<MAX_DEPTH>;
		using Elem_t    = typename Mesh_t::VoxelElement;
		using Vert_t    = typename Mesh_t::VoxelVertex;
		using DofKey_t  = typename Vert_t::OtherPeriodicType<BC>;
		using DOF_t     = VoxelQ1<DofKey_t>;
		using Handler_t = DofHandler<Mesh_t,DOF_t>;

		using BiMass_t  = SymmetricL2<DOF_t>;
		using BiStiff_t = SymmetricH1<DOF_t>;
		using Kernel_t  = Kernel<4,BiMass_t,BiStiff_t>;
		using SpMat_t   = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
		using Vec_t     = Eigen::VectorXd;

		using Point_t   = gutil::Point<3,double>;

		Mesh_t		mesh;
		Handler_t   dofhandler;
		BiMass_t    mass_form;
		BiStiff_t   stiff_form;
		SpMat_t     M, A;
		Vec_t		solution;
		Vec_t		rhs;

		PoissonQ1(const Point_t low, const Point_t high) :
			mesh{low, high},
			dofhandler{mesh} {}

		//initialize/reset problem to the specified depth of the mesh
		//the mesh will be in a conformal state after this
		void set_depth(int dd) {
			LogTime timer{"PoissonQ1::set_depth"};
			mesh.set_depth(dd);
			dofhandler.set_depth(dd);
			dofhandler.save_dof_list();
			assert(dofhandler.last_compressed_dofs().size() == dofhandler.n_dofs() );
		}

		void integrate() {
			LogTime timer{"PoissonQ1::integrate"};
			const auto diag = mesh.high - mesh.low;
			Kernel_t kernel(diag[0], diag[1], diag[2], mass_form, stiff_form);
			auto action = [this, &kernel](Elem_t el) {
				const auto el_basis = dofhandler.basis_active(el);
				kernel.set_element(el);
				kernel.template form<0>().set_basis(el_basis, el_basis);
				kernel.template form<1>().set_basis(el_basis, el_basis);
				kernel.template compute_scatter<0>();
				kernel.template compute_scatter<1>();
			};

			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};
			mesh.template for_each<Elem_t>(action, false, predicate);

			//TODO: replace rhs with a linear form
			rhs = 0.0*Eigen::VectorXd::Ones(dofhandler.n_dofs());
		}

		void build_matrices() {
			LogTime timer{"PoissonQ1::build_matrices"};
			const auto& dofs = dofhandler.last_compressed_dofs();

			#ifdef _OPENMP
			omp_set_max_active_levels(2);
			omp_set_nested(1);
			#pragma omp parallel
			#pragma omp single
			#endif
			{
				#ifdef _OPENMP
				#pragma omp task
				#endif
				{
					M = mass_form.to_eigen_csr(dofs,dofs);
				}

				#ifdef _OPENMP
				#pragma omp task
				#endif
				{
					A = stiff_form.to_eigen_csr(dofs,dofs);
				}
			}
		}

		//apply BC to A and the rhs
		//Each active DOF that satisfies the Predicate will have its corresponding row
		//set to its identity. note that this does not change its krylov spaces so that
		//CG should still converge
		template<typename Function, typename Predicate>
		void apply_dirichlet(Function&& fun, Predicate&& pred) {
			LogTime timer{"PoissonQ1::apply_dirichlet"};

			assert(A.isCompressed());
			const auto& dofs = dofhandler.last_compressed_dofs();
			for (size_t r=0; r<dofs.size(); ++r) {
				const auto dof = dofs[r];
				if (pred(dof)) {
					//zero the row of A to the identity
					int r_start = A.outerIndexPtr()[r];
					int r_end   = A.outerIndexPtr()[r+1];
					for (int idx=r_start; idx<r_end; ++idx) {
						const int c = A.innerIndexPtr()[idx];
						A.valuePtr()[idx] = (c==static_cast<int>(r)) ? 1.0 : 0.0;
					}

					//set the rhs
					rhs[r] = fun(dof);
				}
			}
		}

		void solve() {
			LogTime timer{"PoissonQ1::solve"};

			Eigen::ConjugateGradient<SpMat_t, Eigen::Lower|Eigen::Upper> cg;
			cg.compute(A);
			solution = cg.solve(rhs);
			// Eigen::SparseLU<SpMat_t> lu;
			// lu.compute(A);
			// solution = lu.solve(rhs);
		}

		void save_as(const std::string filename) const {
			LogTime timer{"PoissonQ1::save_as"};

			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("PoissonQ1::save_as - could not open file: " + filename);
			}

			//write the mesh and get the number of vertices
			const auto n_verts = mesh.write_unstructured_vtk(file);

			//interpolate the solution to the vertex values
			auto vert_vals = dofhandler.interpolate_to_vertices(solution, n_verts);

			//append solution header
			file << "POINT_DATA " << n_verts << "\n";
			mesh.append_unstructured_point_data_vtk(
				file,
				"SCALARS solution float 1\nLOOKUP_TABLE default",
				n_verts,
				[&vert_vals](Vert_t vtx) {return vert_vals[vtx.linear_index()];});
		}


		void check_mats() const {
			Eigen::VectorXd vec = Eigen::VectorXd::Ones(M.rows());
			std::cout << "mass: " << (M * vec).transpose() * vec << std::endl;


			//populate the vec with the x coordinates of each dof to test the stiffness matrix
			for (size_t i=0; i<dofhandler.last_compressed_dofs().size(); ++i) {
				double x = dofhandler.last_compressed_dofs()[i].key.x();
				vec[i] = (1.0-x)*mesh.low[0] + x*mesh.high[0];
			}

			std::cout << "stiff: " << (A * vec).transpose() * vec << std::endl;
		}
	};




	template<int BC=0, int MAX_DEPTH=8>
	struct PoissonP0RT0
	{
		using Mesh_t    = HierarchicalVoxelMesh<MAX_DEPTH>;
		using Elem_t    = typename Mesh_t::VoxelElement;
		using Face_t    = typename Mesh_t::VoxelFace;
		using Vert_t    = typename Mesh_t::VoxelVertex;
		using DofTKey_t = typename Elem_t::OtherPeriodicType<BC>;
		using DofQKey_t = typename Face_t::OtherPeriodicType<BC>;
		using DOF_t     = VoxelQ1<DofKey_t>;
		using Handler_t = DofHandler<Mesh_t,DOF_t>;

		using BiMass_t  = SymmetricL2<DOF_t>;
		using BiStiff_t = SymmetricH1<DOF_t>;
		using Kernel_t  = Kernel<4,BiMass_t,BiStiff_t>;
		using SpMat_t   = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
		using Vec_t     = Eigen::VectorXd;

		using Point_t   = gutil::Point<3,double>;

		Mesh_t		mesh;
		Handler_t   dofhandler;
		BiMass_t    mass_form;
		BiStiff_t   stiff_form;
		SpMat_t     M, A;
		Vec_t		solution;
		Vec_t		rhs;
	}

}

