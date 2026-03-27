#pragma once

#include "fem/charms_dofs.hpp"
#include "fem/charms_dofhandler.hpp"

#include <Eigen/SparseCore>
#include <omp.h>

namespace gv::fem
{
	//create CSR/CSC matrix with the correct sparsity structure
	//This is only for symmetric matricies (i.e., mass/stiffness) defined with the same dofs for the function space and trial space
	//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
	//RowMajor format is better for settng boundary conditions, but ColMajor might be better for Eigen routines
	template<typename Mesh_t, typename DOF_t, typename Coef_t, int Format=Eigen::RowMajor>
	Eigen::SparseMatrix<double,Format> init_matrix(const CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>& dofhandler)
	{
		using T = double;
		using Triplet = Eigen::Triplet<T>;
		std::vector<std::vector<Triplet>> color_coo_idx;

		//convenient references
		const auto& mesh            = dofhandler.mesh;
		const auto& element_basis_a = dofhandler.get_element_basis_a();
		const auto& element_basis_s = dofhandler.get_element_basis_s();
		const auto& dofs            = dofhandler.get_dofs();
		const auto& dof_map         = dofhandler.get_dof_map();		

		size_t ncolors = mesh.nColors();
		color_coo_idx.resize(ncolors);

		#pragma omp parallel for
		for (size_t c=0; c<ncolors; ++c) {
			auto& coo_idx = color_coo_idx[c];
			coo_idx.reserve(64*mesh.colorCount(c)); //approximate. in CHARMS, it is difficult to know which basis functions interact

			for (size_t e_idx=0; e_idx<mesh.nElements(false); ++e_idx) {
				const auto& ELEM = mesh.getElement(e_idx);
				if (ELEM.color != c) {continue;}

				//basis_s-basis_s and basis_s-basis_a interactions
				for (size_t global_dof_1 : element_basis_s[ELEM.index]) {
					if (!dofs[global_dof_1].active) {continue;}

					size_t compressed_dof_1 = dof_map.global2compressed[global_dof_1];
					coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_1, T{0}));

					//basis_s-basis_s
					for (size_t global_dof_2 : element_basis_s[ELEM.index]) {
						if (global_dof_2 <= global_dof_1) {continue;}
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}

					//basis_s-basis_a
					for (size_t global_dof_2 : element_basis_a[ELEM.index]) {
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}
				}

				//basis_a-basis_a interactions
				for (size_t global_dof_1 : element_basis_a[ELEM.index]) {
					if (!dofs[global_dof_1].active) {continue;}

					size_t compressed_dof_1 = dof_map.global2compressed[global_dof_1];
					coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_1, T{0}));

					for (size_t global_dof_2 : element_basis_a[ELEM.index]) {
						if (global_dof_2 <= global_dof_1) {continue;}
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}
				}
			}
		}

		//join coo_idx from each color
		size_t n_coo_idx = 0;
		for (size_t c=0; c<mesh.nColors(); ++c) {n_coo_idx += color_coo_idx[c].size();}
		std::vector<Triplet> all_coo_idx;
		all_coo_idx.reserve(n_coo_idx);
		for (size_t c=0; c<mesh.nColors(); ++c) {
			all_coo_idx.insert(
				all_coo_idx.end(), 
				std::make_move_iterator(color_coo_idx[c].begin()), 
				std::make_move_iterator(color_coo_idx[c].end())
			);
		}

		//create matrix
		Eigen::SparseMatrix<T,Format> mat;
		mat.resize(dof_map.ndof(), dof_map.ndof());
		mat.setFromTriplets(all_coo_idx.begin(), all_coo_idx.end());
		return mat;
	}



	//create CSR/CSC matrix with the correct sparsity structure
	//This does not assume symmetry
	//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
	//RowMajor format is better for settng boundary conditions, but ColMajor might be better for Eigen routines
	template<typename Mesh_t,
			typename DOF_ROW_t, typename Coef_row_t,
			typename DOF_COL_t, typename Coef_col_t,
			int Format=Eigen::RowMajor>
	Eigen::SparseMatrix<double,Format> init_matrix(const CharmsDOFhandler<Mesh_t,DOF_ROW_t,Coef_row_t>& dofhandler_row,
		const CharmsDOFhandler<Mesh_t,DOF_COL_t,Coef_col_t>& dofhandler_col)
	{
		//make sure that both dofhandlers are defined on the same mesh
		if (&dofhandler_row.mesh != &dofhandler_col.mesh) {
			throw std::runtime_error("init_matrix: both dofhandlers must be defined on the same mesh.");
		}

		using T = double;
		using Triplet = Eigen::Triplet<T>;
		std::vector<std::vector<Triplet>> color_coo_idx;

		//convenient references
		const auto& mesh        = dofhandler_row.mesh;

		const auto& basis_a_row = dofhandler_row.get_element_basis_a();
		const auto& basis_s_row = dofhandler_row.get_element_basis_s();
		const auto& dofs_row    = dofhandler_row.get_dofs();
		const auto& dof_map_row = dofhandler_row.get_dof_map();		

		const auto& basis_a_col = dofhandler_col.get_element_basis_a();
		const auto& basis_s_col = dofhandler_col.get_element_basis_s();
		const auto& dofs_col    = dofhandler_col.get_dofs();
		const auto& dof_map_col = dofhandler_col.get_dof_map();		


		size_t ncolors = mesh.nColors();
		color_coo_idx.resize(ncolors);

		#pragma omp parallel for
		for (size_t c=0; c<ncolors; ++c) {
			auto& coo_idx = color_coo_idx[c];
			coo_idx.reserve(64*mesh.colorCount(c)); //approximate. in CHARMS, it is difficult to know which basis functions interact
			
			//helper function to insert the interactions between the row and column basis sets
			auto insert_block = [&](
				const std::unordered_set<size_t>& row_set,
				const std::unordered_set<size_t>& col_set)
			{
				for (size_t global_r_dof : row_set) {
					if (!dofs_row[global_r_dof].active) {continue;}
					size_t compressed_r_dof = dof_map_row.global2compressed[global_r_dof];
					for (size_t global_c_dof : col_set) {
						if (!dofs_col[global_c_dof].active) {continue;}
						size_t compressed_c_dof = dof_map_col.global2compressed[global_c_dof];
						coo_idx.push_back(Triplet(compressed_r_dof, compressed_c_dof, T{0}));
					}
				}
			};

			for (size_t e_idx=0; e_idx<mesh.nElements(false); ++e_idx) {
				const auto& ELEM = mesh.getElement(e_idx);
				if (ELEM.color != c) {continue;}

				//All four interactions between row_a,row_s and col_a,col_s
				insert_block(basis_s_row[e_idx], basis_s_col[e_idx]);
				insert_block(basis_s_row[e_idx], basis_a_col[e_idx]);
				insert_block(basis_a_row[e_idx], basis_s_col[e_idx]);
				insert_block(basis_a_row[e_idx], basis_a_col[e_idx]);
			}
		}

		//join coo_idx from each color
		size_t n_coo_idx = 0;
		for (size_t c=0; c<mesh.nColors(); ++c) {n_coo_idx += color_coo_idx[c].size();}
		std::vector<Triplet> all_coo_idx;
		all_coo_idx.reserve(n_coo_idx);
		for (size_t c=0; c<mesh.nColors(); ++c) {
			all_coo_idx.insert(
				all_coo_idx.end(), 
				std::make_move_iterator(color_coo_idx[c].begin()), 
				std::make_move_iterator(color_coo_idx[c].end())
			);
		}

		//create matrix
		Eigen::SparseMatrix<T,Format> mat;
		mat.resize(dof_map_row.ndof(), dof_map_col.ndof());
		mat.setFromTriplets(all_coo_idx.begin(), all_coo_idx.end());
		return mat;
	}


	////////////////////////////////////
	/// Integration routines. Pass the Kernel as a lambda function
	////////////////////////////////////
	
	//integrate a symmetric matrix kernel.
	//the kernel function must handle any integration (exact or otherwise) and must be of the form
	// kernel(element_index, dof_list, vector<double>& local_mat_vals) where dof_list are the global dof indices and local_mat_vals
	// is a (correctly sized) vector to store the local matrix values.
	// local_mat_vals[i+n*j] should record the interaction between dof_list[i] and dof_list[j] over the specified element where dof_list.size()==n
	//the sparsity matrix of mat must have already been set.
	template<typename Mesh_t, typename DOF_t, typename Coef_t, typename SpMat_t, typename Kernel_t>
	void integrate_kernel(const CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>& dofhandler, SpMat_t& mat, Kernel_t&& kernel)
	{
		static_assert(gv::mesh::ColorableMeshType<Mesh_t>);

		//convenient references
		const auto& mesh            = dofhandler.mesh;
		const auto& element_basis_a = dofhandler.get_element_basis_a();
		const auto& element_basis_s = dofhandler.get_element_basis_s();
		const auto& dofs            = dofhandler.get_dofs();
		const auto& dof_map         = dofhandler.get_dof_map();		

		for (size_t c=0; c<mesh.nColors(); ++c) {
			#pragma omp parallel
			{
				std::vector<size_t> global_dofs;
				std::vector<size_t> compressed_dofs;
				std::vector<double> K_local_vals;

				#pragma omp for
				for (size_t e_idx=0; e_idx<mesh.nElements(false); ++e_idx) {
					const auto& ELEM = mesh.getElement(e_idx);
					if (ELEM.color != c) {continue;}

					//collect all active dofs
					global_dofs.clear();
					compressed_dofs.clear();
					for (size_t g : element_basis_s[e_idx]) {
						if (!dofs[g].active) {continue;}
						global_dofs.push_back(g);
						compressed_dofs.push_back(dof_map.global2compressed[g]);
					}
					for (size_t g : element_basis_a[e_idx]) {
						if (!dofs[g].active) {continue;}
						global_dofs.push_back(g);
						compressed_dofs.push_back(dof_map.global2compressed[g]);
					}

					//compute local matrix
					const int n = global_dofs.size();
					if (n==0) {continue;}
					K_local_vals.resize(n*n);
					std::fill(K_local_vals.begin(), K_local_vals.end(), 0.0);
					kernel(e_idx, global_dofs, K_local_vals);

					//scatter local matrix
					for (int j=0; j<n; ++j) {
						const size_t c_j = compressed_dofs[j];
						for (int i=0; i<n; ++i) {
							const size_t c_i = compressed_dofs[i];
							mat.coeffRef(c_i, c_j) += K_local_vals[i+n*j];
						}
					}
				}
			}
		}

		mat.makeCompressed();
	}



	//Integrate a generic bilinear form
	template<typename Mesh_t,
			typename DOF_ROW_t, typename Coef_row_t,
			typename DOF_COL_t, typename Coef_col_t,
			typename SpMat_t, typename Kernel_t>
	void integrate_kernel(const CharmsDOFhandler<Mesh_t,DOF_ROW_t,Coef_row_t>& dofhandler_row,
		const CharmsDOFhandler<Mesh_t,DOF_COL_t,Coef_col_t>& dofhandler_col, SpMat_t& mat, Kernel_t&& kernel)
	{
		static_assert(gv::mesh::ColorableMeshType<Mesh_t>);

		//make sure that both dofhandlers are defined on the same mesh
		if (&dofhandler_row.mesh != &dofhandler_col.mesh) {
			throw std::runtime_error("init_matrix: both dofhandlers must be defined on the same mesh.");
		}

		//convenient references
		const auto& mesh        = dofhandler_row.mesh;

		const auto& basis_a_row = dofhandler_row.get_element_basis_a();
		const auto& basis_s_row = dofhandler_row.get_element_basis_s();
		const auto& dofs_row    = dofhandler_row.get_dofs();
		const auto& dof_map_row = dofhandler_row.get_dof_map();		

		const auto& basis_a_col = dofhandler_col.get_element_basis_a();
		const auto& basis_s_col = dofhandler_col.get_element_basis_s();
		const auto& dofs_col    = dofhandler_col.get_dofs();
		const auto& dof_map_col = dofhandler_col.get_dof_map();

		

		//serial in color, paralel within a color
		//the mesh is colored so that no two elements of the same color share a vertex
		for (size_t c=0; c<mesh.nColors(); ++c) {
			#pragma omp parallel
			{
				std::vector<size_t> global_row_dofs;
				std::vector<size_t> compressed_row_dofs;

				std::vector<size_t> global_col_dofs;
				std::vector<size_t> compressed_col_dofs;
				
				Eigen::MatrixXd K_local(1,1);
				#pragma omp for
				for (size_t e_idx=0; e_idx<mesh.nElements(false); ++e_idx) {
					const auto& ELEM = mesh.getElement(e_idx);
					if (ELEM.color != c) {continue;}

					//collect all active row dofs
					global_row_dofs.clear();
					compressed_row_dofs.clear();
					for (size_t g : basis_s_row[e_idx]) {
						if (!dofs_row[g].active) {continue;}
						global_row_dofs.push_back(g);
						compressed_row_dofs.push_back(dof_map_row.global2compressed[g]);
					}
					for (size_t g : basis_a_row[e_idx]) {
						if (!dofs_row[g].active) {continue;}
						global_row_dofs.push_back(g);
						compressed_row_dofs.push_back(dof_map_row.global2compressed[g]);
					}

					//collect all active col dofs
					global_col_dofs.clear();
					compressed_col_dofs.clear();
					for (size_t g : basis_s_col[e_idx]) {
						if (!dofs_col[g].active) {continue;}
						global_col_dofs.push_back(g);
						compressed_col_dofs.push_back(dof_map_col.global2compressed[g]);
					}
					for (size_t g : basis_a_col[e_idx]) {
						if (!dofs_col[g].active) {continue;}
						global_col_dofs.push_back(g);
						compressed_col_dofs.push_back(dof_map_col.global2compressed[g]);
					}

					//compute local matrix
					const int n = global_row_dofs.size();
					const int m = global_col_dofs.size();
					if (n==0 or m==0) {continue;}

					K_local.resize(n,m);
					for (int i=0; i<n; ++i) {
						for (int j=0; j<m; ++j) {
							K_local(i,j) = kernel(e_idx, global_row_dofs[i], global_col_dofs[j]);
						}
					}

					//scatter local matrix
					for (int i=0; i<n; ++i) {
						for (int j=0; j<m; ++j) {
							const size_t c_i = compressed_row_dofs[i];
							const size_t c_j = compressed_col_dofs[j];
							mat.coeffRef(c_i, c_j) += K_local(i,j);
						}
					}
				}
			}
		}

		mat.makeCompressed();
	}


}

