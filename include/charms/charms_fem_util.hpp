#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <vector>
#include <array>

namespace gv::charms
{
	template <class Mesh_t>
	class CharmsGalerkinMatrixConstructor
	{
	public:
		using Element_t     = typename Mesh_t::Element_t;
		using BasisFun_t    = typename Mesh_t::BasisFun_t;
		using BasisList_t   = typename Mesh_t::BasisList_t;
		using ElementList_t = typename Mesh_t::ElementList_t;
		using VertexList_t  = typename Mesh_t::VertexList_t;
		using Point_t       = typename Mesh_t::Point_t;
		using Box_t         = typename Mesh_t::Box_t;
		using Index_t       = typename Mesh_t::Index_t;

		CharmsGalerkinMatrixConstructor() {}
		
		
		//construct Galerkin mass matrix
		template <int Format_t>
		void make_mass_matrix(Eigen::SparseMatrix<double,Format_t> &mat, const BasisList_t &basis, const ElementList_t &elements)
		{
			//get list of active basis functions and elements (and their inverses)
			std::vector<size_t> basis_active2all, basis_all2active, elem_active2all, elem_all2active;
			get_active(basis_active2all, basis_all2active, basis);
			get_active(elem_active2all, elem_all2active, elements);

			//pre-process to determine matrix structure so that the integration loop can be done in parallel
			using Triplet = Eigen::Triplet<double>;
			std::vector<Triplet> coo_structure;
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++) //loop over active elements
			{
				const Element_t& ELEM = elements[elem_active2all[e_idx]];
				assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function
				
				for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
				{
					size_t basis_i = ELEM.basis_s[i];
					coo_structure.push_back(Triplet(basis_i, basis_i, 0)); //diagonal entry

					//check for interaction between basis functions on the same level
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						size_t basis_j = ELEM.basis_s[j];
						coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
						coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
					}

					//check for interaction with coarser basis functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						size_t basis_j = ELEM.basis_a[j];
						coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
						coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
					}
				}
			}

			//create matrix
			mat.setZero();
			mat.resize(basis_active2all.size(), basis_active2all.size());
			mat.setFromTriplets(coo_structure.begin(), coo_structure.end()); //all zeros, but structure is known

			//populate matrix
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++) //loop over active elements
			{
				const Element_t& ELEM = elements[elem_active2all[e_idx]];
				assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function

				for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
				{
					size_t basis_i = ELEM.basis_s[i];
					size_t active_basis_i = basis_all2active[basis_i];
					const BasisFun_t& FUN_i = basis[basis_i];

					//temporary: testing using mass lumped, TODO: MAKE A JACOBIAN METHOD
					Point_t H = ELEM.H();
					double volume = H[0]*H[1]*H[2];
					mat.coeffRef(active_basis_i,active_basis_i) += volume*FUN_i.eval(ELEM.center())*FUN_i.eval(ELEM.center());

					//check for interaction between basis functions on the same level
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						size_t basis_j = ELEM.basis_s[j];
						size_t active_basis_j = basis_all2active[basis_j];
						const BasisFun_t& FUN_j = basis[basis_j];


						double val = volume*FUN_i.eval(ELEM.center())*FUN_j.eval(ELEM.center());
						mat.coeffRef(active_basis_i,active_basis_j) += val;
						mat.coeffRef(active_basis_j,active_basis_i) += val;
					}

					//check for interaction with coarser basis functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						size_t basis_j = ELEM.basis_a[j];
						size_t active_basis_j = basis_all2active[basis_j];
						const BasisFun_t& FUN_j = basis[basis_j];

						double val = volume*FUN_i.eval(ELEM.center())*FUN_j.eval(ELEM.center());
						mat.coeffRef(active_basis_i,active_basis_j) += val;
						mat.coeffRef(active_basis_j,active_basis_i) += val;
					}
				}
			}
		}
	};
}