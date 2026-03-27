#pragma once
#include "gutil.hpp"
#include "fem/charms_integrator.hpp"
#include "fem/spmatrix_util.hpp"
#include "mesh/vtk_elements.hpp"
#include <Eigen/SparseCore>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>
namespace gv::pde
{
	//Solve the helmholtz equation with natural (homogeneous neumann) BC
	//strong form: -laplace(u) = lamda u
	//weak form: a(u,v) = lambda*m(u,v) where a(u,v) = int(grad u * grad v) and m(u,v) = int(u*v)
	//discrete form: Find the leading eigenvalue of A
	
	template<typename Mesh_t, typename DOFHandler_t>
	struct HelmholtzSolver
	{

		Mesh_t& mesh;
		DOFHandler_t& dofhandler;

		Eigen::SparseMatrix<double,Eigen::RowMajor> A;
		Eigen::SparseMatrix<double,Eigen::RowMajor> M;

		Eigen::VectorXd eigenvalues;
		Eigen::MatrixXd eigenvectors;

		HelmholtzSolver(Mesh_t& mesh_, DOFHandler_t& dofhandler_) : mesh(mesh_), dofhandler(dofhandler_) {}

		double mass_kernel(const size_t e_idx, const size_t i, const size_t j) const
		{
			const auto& DOF_i = dofhandler.dof(i);
			const auto& DOF_j = dofhandler.dof(j);
			assert(mesh.getElement(e_idx).vtkID == 11); //voxels
			gv::mesh::VTK_VOXEL<Mesh_t> VOXEL;
			using RefPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::RefPoint_t;
			VOXEL.set_element(mesh, e_idx);

			constexpr double gauss_x[4] {-0.861136, -0.339981, 0.339981, 0.861136};
			constexpr double gauss_w[4] { 0.347855,  0.652145, 0.652145, 0.347855};
			double val=0;
			for (int ii=0; ii<4; ++ii) {
				for (int jj=0; jj<4; ++jj) {
					for (int kk=0; kk<4; ++kk) {
						const auto coordinate = VOXEL.ref2geo(RefPoint_t{gauss_x[ii], gauss_x[jj], gauss_x[kk]});
						val += DOF_i.eval_at(coordinate, mesh)*DOF_j.eval_at(coordinate, mesh)*gauss_w[ii]*gauss_w[jj]*gauss_w[kk];
					}
				}
			}

			const auto& jac = VOXEL.jacobian(RefPoint_t{0,0,0}); //constant
			const double jacdet = jac(0,0)*jac(1,1)*jac(2,2);
			return val*jacdet;
		}

		double stiff_kernel(const size_t e_idx, const size_t i, const size_t j) const
		{
			const auto& DOF_i = dofhandler.dof(i);
			const auto& DOF_j = dofhandler.dof(j);
			assert(mesh.getElement(e_idx).vtkID == 11); //voxels
			gv::mesh::VTK_VOXEL<Mesh_t> VOXEL;
			using RefPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::RefPoint_t;
			VOXEL.set_element(mesh, e_idx);

			constexpr double gauss_x[4] {-0.861136, -0.339981, 0.339981, 0.861136};
			constexpr double gauss_w[4] { 0.347855,  0.652145, 0.652145, 0.347855};
			
			
			double val=0;
			for (int ii=0; ii<4; ++ii) {
				for (int jj=0; jj<4; ++jj) {
					for (int kk=0; kk<4; ++kk) {
						const auto coordinate = VOXEL.ref2geo(RefPoint_t{gauss_x[ii], gauss_x[jj], gauss_x[kk]});
						const double dot = gutil::dot(DOF_i.eval_grad_at(coordinate, mesh), DOF_j.eval_grad_at(coordinate, mesh));
						val += dot*gauss_w[ii]*gauss_w[jj]*gauss_w[kk];
					}
				}
			}

			const auto& jac = VOXEL.jacobian(RefPoint_t{0,0,0}); //constant
			const double jacdet = jac(0,0)*jac(1,1)*jac(2,2);
			return val*jacdet;
		}

		void assemble_mats()
		{
			//A and M have the same sparsity pattern
			A = gv::fem::init_matrix(dofhandler);
			M = A;

			//save the sparsity pattern
			gv::fem::SparseMatImage(A).save_as_bw("helmholtz_sparsity.bmp");

			//construct the matrices
			gv::fem::integrate_kernel(dofhandler, A, [this](size_t e, size_t i, size_t j) {return stiff_kernel(e,i,j);});
			gv::fem::integrate_kernel(dofhandler, M, [this](size_t e, size_t i, size_t j) {return mass_kernel(e,i,j);});
		}

		void compute_eigenvals(int n=1)
		{
			Spectra::SparseSymMatProd<double, Eigen::Upper, Eigen::RowMajor> op(A);
			Spectra::SparseCholesky<double, Eigen::Upper, Eigen::RowMajor>   Bop(M);

			const int ncv = std::max(3*n+1, 20);
			Spectra::SymGEigsSolver< 
				Spectra::SparseSymMatProd<double, Eigen::Upper, Eigen::RowMajor>,
				Spectra::SparseCholesky<double, Eigen::Upper, Eigen::RowMajor>,
				Spectra::GEigsMode::Cholesky>
			 	solver(op, Bop, n, ncv);
			solver.init();
			solver.compute(Spectra::SortRule::SmallestAlge);

			if (solver.info() != Spectra::CompInfo::Successful) {
				throw std::runtime_error("Helmholtz: compute_eigenvals failed.");
			}

			eigenvalues  = solver.eigenvalues();
			eigenvectors = solver.eigenvectors();
		}

		void save_to_vtk()
		{
			const auto& map = dofhandler.get_dof_map();

			for (size_t i=0; i<eigenvalues.size(); ++i) {
				//populate relevent coefficients in the dofhandler
				for (size_t c_idx=0; c_idx < map.ndof(); ++c_idx) {
					dofhandler.coef(map.compressed2global[c_idx]) = eigenvectors(c_idx, i);
				}

				//save as a vtk
				dofhandler.save_as("helmholtz_" + std::to_string(i) + ".vtk", 1, true);
			}
		}
	};
}
