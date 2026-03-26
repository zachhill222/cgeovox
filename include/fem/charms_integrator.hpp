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
		const auto& mesh = dofhandler.mesh;
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
				insert_block(basis_s_row[ELEM.index], basis_s_col[ELEM.index]);
				insert_block(basis_s_row[ELEM.index], basis_a_col[ELEM.index]);
				insert_block(basis_a_row[ELEM.index], basis_s_col[ELEM.index]);
				insert_block(basis_a_row[ELEM.index], basis_a_col[ELEM.index]);
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



}

