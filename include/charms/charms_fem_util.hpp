#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <vector>
#include <array>
#include <functional>

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
		using ScalarFun_t   = double (*)(Point_t); //function from Point_t to double
		using VectorFun_t   = Point_t (*)(Point_t); //function from Point_t to Point_t

		std::vector<double> gauss_locations {-0.7745966692414834, 0, 0.7745966692414834};
		std::vector<double> gauss_weights {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

		// std::vector<double> gauss_locations {0};
		// std::vector<double> gauss_weights {2};
		
		double gauss_quad(const std::function<double(Point_t)>& FUN_i, const std::function<double(Point_t)>& FUN_j, const Box_t& BOX) const
		{
			const Point_t H = BOX.sidelength();
			const Point_t& center = BOX.center();

			double integral  = 0;

			Point_t ref_point;
			for (size_t i=0; i<gauss_locations.size(); i++) {
				ref_point[0] = gauss_locations[i];
				for (size_t j=0; j<gauss_locations.size(); j++) {
					ref_point[1] = gauss_locations[j];
					for (size_t k=0; k<gauss_locations.size(); k++) {
						ref_point[2] = gauss_locations[k];
						Point_t point = center + 0.5*H*ref_point;
						integral += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * FUN_i(point) * FUN_j(point);
					}
				}
			}

			return 0.125*H[0]*H[1]*H[2]*integral;
		}


		double gauss_quad(const std::function<Point_t(Point_t)>& FUN_i, const std::function<Point_t(Point_t)>& FUN_j, const Box_t& BOX) const
		{
			const Point_t H = BOX.sidelength();
			const Point_t& center = BOX.center();

			double integral  = 0;

			Point_t ref_point;
			for (size_t i=0; i<gauss_locations.size(); i++) {
				ref_point[0] = gauss_locations[i];
				for (size_t j=0; j<gauss_locations.size(); j++) {
					ref_point[1] = gauss_locations[j];
					for (size_t k=0; k<gauss_locations.size(); k++) {
						ref_point[2] = gauss_locations[k];
						Point_t point = center + 0.5*H*ref_point;
						Point_t grad_i = FUN_i(point);
						Point_t grad_j = FUN_j(point);

						integral += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * (grad_i[0]*grad_j[0] + grad_i[1]*grad_j[1] + grad_i[2]*grad_j[2]);
					}
				}
			}

			//the jacobians for change of variables must be taken care of in the evaluation of FUN_i and FUN_j
			return 0.125*H[0]*H[1]*H[2]*integral;
		}


		CharmsGalerkinMatrixConstructor() {}
		
		//construct Galerkin mass matrix
		template <int Format_t>
		void make_mass_matrix(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active)
		{
			
			//compute sparse matrix in coo format
			using Triplet = Eigen::Triplet<double>;
			std::vector<Triplet> coo_structure;

			//loop over active elements and integrate interacting basis functions
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				//reference to current active element
				const Element_t& ELEM = elements[elem_active2all[e_idx]];

				//interaction between basis_s functions with other basis_s and basis_a functions
				for (int i=0; i<ELEM.cursor_basis_s; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_s[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//integrate FUN_i * FUN_i
					std::function<double(Point_t)> fun_i = [&FUN_i](Point_t point) -> double {return FUN_i.eval(point);};
					// ScalarFun_t fun_i = [&FUN_i](Point_t point) -> double {return FUN_i.eval(point);};
					double integral = gauss_quad(fun_i, fun_i, ELEM.bbox());
					coo_structure.push_back(Triplet(global_i, global_i, integral));

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						// auto fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}

					//loop through any ancestor basis_a functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						// auto fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}
				}

				//interaction between basis_a functions and other basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_a[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//integrate FUN_i * FUN_i
					std::function<double(Point_t)> fun_i = [&FUN_i](Point_t point) -> double {return FUN_i.eval(point);};
					// auto fun_i = [&FUN_i](Point_t point) -> double {return FUN_i.eval(point);};
					double integral = gauss_quad(fun_i, fun_i, ELEM.bbox());
					coo_structure.push_back(Triplet(global_i, global_i, integral));

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						// auto fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}
				}
			}

			//construct matrix
			mat.setZero();
			mat.resize(basis_active2all.size(), basis_active2all.size());
			mat.setFromTriplets(coo_structure.begin(), coo_structure.end());
		}


		//construct Galerkin stiffness matrix
		template <int Format_t>
		void make_stiff_matrix(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active)
		{
			
			//compute sparse matrix in coo format
			using Triplet = Eigen::Triplet<double>;
			std::vector<Triplet> coo_structure;

			//loop over active elements and integrate interacting basis functions
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				//reference to current active element
				const Element_t& ELEM = elements[elem_active2all[e_idx]];

				//interaction between basis_s functions with other basis_s and basis_a functions
				for (int i=0; i<ELEM.cursor_basis_s; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_s[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//integrate grad(FUN_i) * grad(FUN_i)
					std::function<Point_t(Point_t)> fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
					double integral = gauss_quad(fun_i, fun_i, ELEM.bbox());
					coo_structure.push_back(Triplet(global_i, global_i, integral));

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						// auto fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}

					//loop through any ancestor basis_a functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}
				}

				//interaction between basis_a functions and other basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_a[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//integrate FUN_i * FUN_i
					std::function<Point_t(Point_t)> fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
					double integral = gauss_quad(fun_i, fun_i, ELEM.bbox());
					coo_structure.push_back(Triplet(global_i, global_i, integral));

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral = gauss_quad(fun_i, fun_j, ELEM.bbox());
						coo_structure.push_back(Triplet(global_i, global_j, integral));
						coo_structure.push_back(Triplet(global_j, global_i, integral));
					}
				}
			}

			//construct matrix
			mat.setZero();
			mat.resize(basis_active2all.size(), basis_active2all.size());
			mat.setFromTriplets(coo_structure.begin(), coo_structure.end());
		}
	};
}