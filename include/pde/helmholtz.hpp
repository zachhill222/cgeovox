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

		void mass_kernel(const size_t e_idx, const std::vector<size_t>& dofs, std::vector<double>& local_mass_vals) const
		{
			
			//construct quadrature points in reference coordinates on the specified element
			using RefPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::RefPoint_t;
			
			constexpr std::array<RefPoint_t,64> ref_quad_points = [](){
				constexpr double gauss_x[4] {-0.861136, -0.339981, 0.339981, 0.861136};
				std::array<RefPoint_t,64> result;
				for (int k=0; k<4; ++k) {
					for (int j=0; j<4; ++j) {
						for (int i=0; i<4; i++) {
							result[i+4*j+16*k] = RefPoint_t{gauss_x[i],gauss_x[j],gauss_x[k]};
						}
					}
				}
				return result;
			}();

			constexpr std::array<double,64> quad_weights = [](){
				constexpr double gauss_w[4] { 0.347855,  0.652145, 0.652145, 0.347855};
				std::array<double,64> result;
				for (int k=0; k<4; ++k) {
					for (int j=0; j<4; ++j) {
						for (int i=0; i<4; i++) {
							result[i+4*j+16*k] = gauss_w[i]*gauss_w[j]*gauss_w[k];
						}
					}
				}
				return result;
			}();

			//convert quadrature points to global coordinates
			using GeoPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::GeoPoint_t;
			assert(mesh.getElement(e_idx).vtkID == 11); //voxels
			gv::mesh::VTK_VOXEL<Mesh_t> VOXEL;
			VOXEL.set_element(mesh, e_idx);
			std::array<GeoPoint_t,64> mesh_quad_points;
			for (int p=0; p<64; ++p) {
				mesh_quad_points[p] = VOXEL.ref2geo(ref_quad_points[p]);
			}

			//tabulate basis function values at quadrature points
			int n = static_cast<int>(dofs.size());
			std::vector<double> dof_vals(64*n, 0.0);
			for (int i=0; i<n; ++i) {
				const auto& DOF = dofhandler.dof(dofs[i]);
				for (int p=0; p<64; ++p) {
					//TODO: eval_at is slow, can I remove this in CHARMS?
					dof_vals[p+64*i] = DOF.eval_at(mesh_quad_points[p], mesh);
				}
			}


			//populate local mass matrix
			//TODO: vectorize?
			const auto& jac = VOXEL.jacobian(RefPoint_t{0,0,0});
			const double jacdet = jac(0,0)*jac(1,1)*jac(2,2);
			for (int i=0; i<n; ++i) {
				for (int j=i; j<n; ++j) {
					double acc = 0.0;
					for (int p=0; p<64; ++p) {
						acc += dof_vals[p+64*i]*dof_vals[p+64*j] * quad_weights[p];
					}
					acc *= jacdet;
					local_mass_vals[i+n*j] = acc;
					if (i!=j) {
						local_mass_vals[j+n*i] = acc;
					}
				}
			}
		}

		void stiff_kernel(const size_t e_idx, const std::vector<size_t>& dofs, std::vector<double>& local_stiff_vals) const
		{
			
			//construct quadrature points in reference coordinates on the specified element
			using RefPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::RefPoint_t;
			
			constexpr std::array<RefPoint_t,64> ref_quad_points = [](){
				constexpr double gauss_x[4] {-0.861136, -0.339981, 0.339981, 0.861136};
				std::array<RefPoint_t,64> result;
				for (int k=0; k<4; ++k) {
					for (int j=0; j<4; ++j) {
						for (int i=0; i<4; i++) {
							result[i+4*j+16*k] = RefPoint_t{gauss_x[i],gauss_x[j],gauss_x[k]};
						}
					}
				}
				return result;
			}();

			constexpr std::array<double,64> quad_weights = [](){
				constexpr double gauss_w[4] { 0.347855,  0.652145, 0.652145, 0.347855};
				std::array<double,64> result;
				for (int k=0; k<4; ++k) {
					for (int j=0; j<4; ++j) {
						for (int i=0; i<4; i++) {
							result[i+4*j+16*k] = gauss_w[i]*gauss_w[j]*gauss_w[k];
						}
					}
				}
				return result;
			}();

			//convert quadrature points to global coordinates
			using GeoPoint_t = typename gv::mesh::VTK_VOXEL<Mesh_t>::GeoPoint_t;
			assert(mesh.getElement(e_idx).vtkID == 11); //voxels
			gv::mesh::VTK_VOXEL<Mesh_t> VOXEL;
			VOXEL.set_element(mesh, e_idx);
			std::array<GeoPoint_t,64> mesh_quad_points;
			for (int p=0; p<64; ++p) {
				mesh_quad_points[p] = VOXEL.ref2geo(ref_quad_points[p]);
			}

			//tabulate basis function gradients at quadrature points
			int n = static_cast<int>(dofs.size());
			std::vector<GeoPoint_t> dof_grads(64*n);
			for (int i=0; i<n; ++i) {
				const auto& DOF = dofhandler.dof(dofs[i]);
				for (int p=0; p<64; ++p) {
					//TODO: eval_grad_at is slow, can I remove this in CHARMS?
					dof_grads[p+64*i] = DOF.eval_grad_at(mesh_quad_points[p], mesh);
				}
			}


			//populate local stiffness matrix
			//TODO: vectorize?
			const auto& jac = VOXEL.jacobian(RefPoint_t{0,0,0});
			const double jacdet = jac(0,0)*jac(1,1)*jac(2,2);
			for (int i=0; i<n; ++i) {
				for (int j=i; j<n; ++j) {
					double acc = 0.0;
					for (int p=0; p<64; ++p) {
						const auto& gi = dof_grads[p+64*i];
						const auto& gj = dof_grads[p+64*j];
						acc += (gi[0]*gj[0] + gi[1]*gj[1] + gi[2]*gj[2]) * quad_weights[p];
					}
					acc *= jacdet;
					local_stiff_vals[i+n*j] = acc;
					if (i!=j) {
						local_stiff_vals[j+n*i] = acc;
					}
				}
			}
		}

		void assemble_mats()
		{
			//A and M have the same sparsity pattern
			A = gv::fem::init_matrix(dofhandler);
			M = A;

			//save the sparsity pattern
			gv::fem::SparseMatImage(A).save_as_bw("helmholtz_sparsity.bmp");

			//construct the matrices
			gv::fem::integrate_kernel(dofhandler, A, [this](size_t e, const std::vector<size_t>& dofs, std::vector<double>& local) {return stiff_kernel(e,dofs,local);});
			gv::fem::integrate_kernel(dofhandler, M, [this](size_t e, const std::vector<size_t>& dofs, std::vector<double>& local) {return mass_kernel(e,dofs,local);});
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

		void save_to_vtk(const std::string& prefix = "")
		{
			const auto& map = dofhandler.get_dof_map();

			for (auto i=0; i<eigenvalues.size(); ++i) {
				//populate relevent coefficients in the dofhandler
				for (size_t c_idx=0; c_idx < map.ndof(); ++c_idx) {
					dofhandler.coef(map.compressed2global[c_idx]) = eigenvectors(c_idx, i);
				}

				//save as a vtk
				dofhandler.save_as(prefix + "helmholtz_" + std::to_string(i) + ".vtk", 1, true);
			}
		}
	};
}
