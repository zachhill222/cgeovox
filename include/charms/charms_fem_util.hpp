#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <omp.h>

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
		using ScalarFun_t   = typename Mesh_t::ScalarFun_t; //std::function<double(Point_t)>; //function from Point_t to double
		using VectorFun_t   = typename Mesh_t::VectorFun_t; //std::function<Point_t(Point_t)>; //function from Point_t to Point_t


		CharmsGalerkinMatrixConstructor() {}
		CharmsGalerkinMatrixConstructor(const Mesh_t& mesh) {}

		std::vector<double> gauss_locations {-0.7745966692414834, 0, 0.7745966692414834};
		std::vector<double> gauss_weights {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

		double penalty=-1; //parameter for weakly enforcing Dirichlet BC

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


		double gauss_quad_interface(const ScalarFun_t& FUN_i, const ScalarFun_t& FUN_j, const ScalarFun_t &W, const Box_t& BOX) const
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
						integral += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * FUN_i(point) * FUN_j(point) * W(point);
					}
				}
			}

			return 0.125*H[0]*H[1]*H[2]*integral;
		}


		double gauss_quad_interface(const VectorFun_t& FUN_i, const VectorFun_t& FUN_j, const ScalarFun_t &W,  const Box_t& BOX) const
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

						integral += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * (grad_i[0]*grad_j[0] + grad_i[1]*grad_j[1] + grad_i[2]*grad_j[2]) * W(point);
					}
				}
			}

			//the jacobians for change of variables must be taken care of in the evaluation of FUN_i and FUN_j
			return 0.125*H[0]*H[1]*H[2]*integral;
		}

		
		//set the sparsity structure for the mass or stiffness matrix
		template <int Format_t>
		void set_matrix_structure(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active)
		{
			//compute the structure of both matrices
			using Triplet = Eigen::Triplet<double>;
			std::vector<Triplet> coo_structure;


			//loop over active elements and record the interaction of active basis functions
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

					
					//FUN_i * FUN_i interaction
					coo_structure.push_back(Triplet(global_i, global_i, 0));

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						
						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}

					//loop through any ancestor basis_a functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						
						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}
				}

				//interaction between basis_a functions and other basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_a[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//FUN_i * FUN_i interaction
					coo_structure.push_back(Triplet(global_i, global_i, 0));

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}
				}
			}


			//initialize matrices with correct structure
			mat.setZero(); mat.makeCompressed();
			mat.resize(basis_active2all.size(), basis_active2all.size());
			mat.setFromTriplets(coo_structure.begin(), coo_structure.end());
		}


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

		//construct Galerkin stiffness matrix in a domain provided by a signed distance function
		template <int Format_t>
		void make_stiff_matrix(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active,
			const ScalarFun_t& HEAVISIDE)
		{
			
			//get sparsity structure
			set_matrix_structure(mat,
				basis,
				basis_active2all,
				basis_all2active,
				elements,
				elem_active2all,
				elem_all2active);

			// const ScalarFun_t HEAVISIDE = [this, sgndist](Point_t point) -> double {return heaviside(sgndist(point), this->epsilon);};

			//loop over active elements and integrate interacting basis functions
			#pragma omp parallel for
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
					double integral = gauss_quad_interface(fun_i, fun_i, HEAVISIDE, ELEM.bbox());
					
					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrategrad(FUN_i) * grad(FUN_i)
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral = gauss_quad_interface(fun_i, fun_j, HEAVISIDE, ELEM.bbox());

						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
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
						integral = gauss_quad_interface(fun_i, fun_j, HEAVISIDE, ELEM.bbox());

						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
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
					double integral = gauss_quad_interface(fun_i, fun_i, HEAVISIDE, ELEM.bbox());
					
					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral = gauss_quad_interface(fun_i, fun_j, HEAVISIDE, ELEM.bbox());
						
						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
					}
				}
			}
		}

		//construct Galerkin stiffness matrix in a domain provided by a signed distance function with weakly imposed Dirichlet BC
		template <int Format_t>
		void make_stiff_matrix_with_BC(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active,
			const ScalarFun_t& HEAVISIDE,
			const ScalarFun_t& DIRAC_DELTA)
		{
			assert(penalty>0);

			//get sparsity structure
			set_matrix_structure(mat,
				basis,
				basis_active2all,
				basis_all2active,
				elements,
				elem_active2all,
				elem_all2active);

			// const ScalarFun_t HEAVISIDE = [this, sgndist](Point_t point) -> double {return heaviside(sgndist(point), this->epsilon);};

			//loop over active elements and integrate interacting basis functions
			#pragma omp parallel for
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
					std::function<Point_t(Point_t)> grad_fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
					std::function<Point_t(Point_t)> fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.eval(point);};
					double integral = gauss_quad_interface(grad_fun_i, grad_fun_i, HEAVISIDE, ELEM.bbox());
					integral += penalty * gauss_quad_interface(fun_i, fun_i, DIRAC_DELTA, ELEM.bbox());

					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate grad(FUN_i) * grad(FUN_i)
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.eval(point);};
						integral = gauss_quad_interface(grad_fun_i, grad_fun_j, HEAVISIDE, ELEM.bbox());
						integral += penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());

						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
					}

					//loop through any ancestor basis_a functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.eval(point);};
						integral = gauss_quad_interface(grad_fun_i, grad_fun_j, HEAVISIDE, ELEM.bbox());
						integral += penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());

						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
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
					std::function<Point_t(Point_t)> grad_fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
						std::function<Point_t(Point_t)> fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.eval(point);};
					double integral = gauss_quad_interface(grad_fun_i, grad_fun_i, HEAVISIDE, ELEM.bbox());
					integral += penalty * gauss_quad_interface(fun_i, fun_i, DIRAC_DELTA, ELEM.bbox());
					
					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						std::function<Point_t(Point_t)> fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.eval(point);};
						integral = gauss_quad_interface(grad_fun_i, grad_fun_j, HEAVISIDE, ELEM.bbox());
						integral += penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());
						
						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
					}
				}
			}
		}


		//construct both Galerkin mass and stiffness matrices
		template <int Format_m, int Format_s>
		void make_integrating_matrices(Eigen::SparseMatrix<double,Format_m> &mat_m, Eigen::SparseMatrix<double,Format_s> &mat_s,
			const BasisList_t& basis,
			const std::vector<size_t> basis_active2all,
			const std::vector<size_t> basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t> elem_active2all,
			const std::vector<size_t> elem_all2active)
		{
			//compute the structure of both matrices
			using Triplet = Eigen::Triplet<double>;
			std::vector<Triplet> coo_structure;


			//loop over active elements and record the interaction of active basis functions
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

					
					//FUN_i * FUN_i interaction
					coo_structure.push_back(Triplet(global_i, global_i, 0));

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						
						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}

					//loop through any ancestor basis_a functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						
						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}
				}

				//interaction between basis_a functions and other basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN_i = basis[ELEM.basis_a[i]];
					if (!FUN_i.is_active) {continue;}
					const size_t global_i = basis_all2active[FUN_i.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					//FUN_i * FUN_i interaction
					coo_structure.push_back(Triplet(global_i, global_i, 0));

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//FUN_i * FUN_j interaction
						coo_structure.push_back(Triplet(global_i, global_j, 0));
						coo_structure.push_back(Triplet(global_j, global_i, 0));
					}
				}
			}


			//initialize matrices with correct structure
			mat_m.setZero();
			mat_m.resize(basis_active2all.size(), basis_active2all.size());
			mat_m.setFromTriplets(coo_structure.begin(), coo_structure.end());

			mat_s.setZero();
			mat_s.resize(basis_active2all.size(), basis_active2all.size());
			mat_s.setFromTriplets(coo_structure.begin(), coo_structure.end());


			//set the values in the matrices
			//loop over active elements and integrate interacting basis functions
			#pragma omp parallel for
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
					double integral_m = gauss_quad(fun_i, fun_i, ELEM.bbox());

					#pragma omp atomic
					mat_m.coeffRef(global_i,global_i) += integral_m;

					//integrate grad(FUN_i) * grad(FUN_i)
					std::function<Point_t(Point_t)> grad_fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
					double integral_s = gauss_quad(grad_fun_i, grad_fun_i, ELEM.bbox());
					
					#pragma omp atomic
					mat_s.coeffRef(global_i,global_i) += integral_s;

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral_m = gauss_quad(fun_i, fun_j, ELEM.bbox());

						#pragma omp atomic
						mat_m.coeffRef(global_i,global_j) += integral_m;

						#pragma omp atomic
						mat_m.coeffRef(global_j,global_i) += integral_m;

						//integrate grad(FUN_i) * grad(FUN_j)
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral_s = gauss_quad(grad_fun_i, grad_fun_j, ELEM.bbox());
						
						#pragma omp atomic
						mat_s.coeffRef(global_i,global_j) += integral_s;

						#pragma omp atomic
						mat_s.coeffRef(global_j,global_i) += integral_s;
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
						integral_m = gauss_quad(fun_i, fun_j, ELEM.bbox());
						
						#pragma omp atomic
						mat_m.coeffRef(global_i,global_j) += integral_m;

						#pragma omp atomic
						mat_m.coeffRef(global_j,global_i) += integral_m;

						//integrate grad(FUN_i) * grad(FUN_j)
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral_s = gauss_quad(grad_fun_i, grad_fun_j, ELEM.bbox());
						
						#pragma omp atomic
						mat_s.coeffRef(global_i,global_j) += integral_s;

						#pragma omp atomic
						mat_s.coeffRef(global_j,global_i) += integral_s;
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
					double integral_m = gauss_quad(fun_i, fun_i, ELEM.bbox());
					mat_m.coeffRef(global_i,global_i) += integral_m;
					// coo_structure.push_back(Triplet(global_i, global_i, integral));

					//integrate grad(FUN_i) * grad(FUN_i)
					std::function<Point_t(Point_t)> grad_fun_i = [&FUN_i](Point_t point) -> Point_t {return FUN_i.grad(point);};
					double integral_s = gauss_quad(grad_fun_i, grad_fun_i, ELEM.bbox());
					mat_s.coeffRef(global_i,global_i) += integral_s;
					// coo_structure.push_back(Triplet(global_i, global_i, integral));

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral_m = gauss_quad(fun_i, fun_j, ELEM.bbox());
						
						#pragma omp atomic
						mat_m.coeffRef(global_i,global_j) += integral_m;

						#pragma omp atomic
						mat_m.coeffRef(global_j,global_i) += integral_m;

						//integrate grad(FUN_i) * grad(FUN_j)
						std::function<Point_t(Point_t)> grad_fun_j = [&FUN_j](Point_t point) -> Point_t {return FUN_j.grad(point);};
						integral_s = gauss_quad(grad_fun_i, grad_fun_j, ELEM.bbox());
						
						#pragma omp atomic
						mat_s.coeffRef(global_i,global_j) += integral_s;

						#pragma omp atomic
						mat_s.coeffRef(global_j,global_i) += integral_s;
					}
				}
			}
		}


		//integrate a function against each basis vector
		template <int n>
		void integrate_fun(Eigen::Vector<double,n> &rhs, const ScalarFun_t &rhs_fun,
			const BasisList_t& basis,
			const std::vector<size_t>& basis_active2all,
			const std::vector<size_t>& basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t>& elem_active2all,
			const std::vector<size_t>& elem_all2active) const
		{
			rhs.setZero();

			#pragma omp parallel for
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				//reference to current active element
				const Element_t& ELEM = elements[elem_active2all[e_idx]];

				//integrate against active basis_s functions
				for (int i=0; i<ELEM.cursor_basis_s; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_s[i]];
					if (!FUN.is_active) {continue;}

					const size_t global_i = basis_all2active[FUN.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					std::function<double(Point_t)> basis_fun = [&FUN](Point_t point) -> double {return FUN.eval(point);};
					double integral = gauss_quad(basis_fun, rhs_fun, ELEM.bbox());

					#pragma omp atomic
					rhs[global_i] += integral;
				}

				//integrate against active basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_a[i]];
					if (!FUN.is_active) {continue;}

					const size_t global_i = basis_all2active[FUN.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					std::function<double(Point_t)> basis_fun = [&FUN](Point_t point) -> double {return FUN.eval(point);};
					double integral = gauss_quad(basis_fun, rhs_fun, ELEM.bbox());

					#pragma omp atomic
					rhs[global_i] += integral;
				}
			}
		}


		//integrate a function against each basis vector in a domain defined by the signed distance function
		template <int n>
		void integrate_fun(Eigen::Vector<double,n> &rhs, const ScalarFun_t &rhs_fun,
			const BasisList_t& basis,
			const std::vector<size_t>& basis_active2all,
			const std::vector<size_t>& basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t>& elem_active2all,
			const std::vector<size_t>& elem_all2active,
			const ScalarFun_t& HEAVISIDE) const
		{
			rhs.setZero();
			// const ScalarFun_t HEAVISIDE = [this, sgndist](Point_t point) -> double {return heaviside(sgndist(point), this->epsilon);};

			#pragma omp parallel for
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				//reference to current active element
				const Element_t& ELEM = elements[elem_active2all[e_idx]];

				//integrate against active basis_s functions
				for (int i=0; i<ELEM.cursor_basis_s; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_s[i]];
					if (!FUN.is_active) {continue;}

					const size_t global_i = basis_all2active[FUN.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					std::function<double(Point_t)> basis_fun = [&FUN](Point_t point) -> double {return FUN.eval(point);};
					double integral = gauss_quad_interface(basis_fun, rhs_fun, HEAVISIDE, ELEM.bbox());

					#pragma omp atomic
					rhs[global_i] += integral;
				}

				//integrate against active basis_a functions
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_a[i]];
					if (!FUN.is_active) {continue;}

					const size_t global_i = basis_all2active[FUN.list_index]; //DOF and matrix index
					assert(global_i<basis_active2all.size());

					std::function<double(Point_t)> basis_fun = [&FUN](Point_t point) -> double {return FUN.eval(point);};
					double integral = gauss_quad_interface(basis_fun, rhs_fun, HEAVISIDE, ELEM.bbox());

					#pragma omp atomic
					rhs[global_i] += integral;
				}
			}
		}

		//set rows for Dirichlet BC
		void set_dirichlet_bc(Eigen::SparseMatrix<double, Eigen::RowMajor> &mat, const std::vector<size_t> &nodes)
		{
			//prepare matrix to shorten loops. note for an integrating matrix the diagonals will never be 0 and will never be cleared here.
			mat.makeCompressed();

			//loop through rows and zero them out
			#pragma omp parallel
			for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
			{
				//get start of row storage
				size_t row_idx   = nodes[n_idx];
				size_t idx_start = mat.outerIndexPtr()[row_idx];
				size_t idx_stop  = mat.outerIndexPtr()[row_idx+1]; //this stop index only works when mat is compressed

				//zero out row
				for (size_t j=idx_start; j<idx_stop; j++)
				{
					mat.valuePtr()[j] = 0.0;
				}
			}

			//loop through rows and set main diagonal entry to 1
			// #pragma omp parallel
			for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
			{
				size_t row_idx = nodes[n_idx];
				mat.coeffRef(row_idx, row_idx) = 1.0;
			}

			//compress matrix to free storage
			mat.prune(0.0);
			mat.makeCompressed();
		}


		//set rows for weakly imposed (penalty method) Dirichlet BC in a domain defined by a signed distance function
		template <int Format_t>
		void set_dirichlet_bc(Eigen::SparseMatrix<double,Format_t> &mat,
			const BasisList_t& basis,
			const std::vector<size_t>& basis_active2all,
			const std::vector<size_t>& basis_all2active,
			const ElementList_t& elements,
			const std::vector<size_t>& elem_active2all,
			const std::vector<size_t>& elem_all2active,
			const ScalarFun_t& DIRAC_DELTA)
		{
			assert(penalty>0);

			// const ScalarFun_t DIRAC_DELTA = [this, sgndist](Point_t point) -> double {return dirac_DIRAC_delta(sgndist(point), this->epsilon);};

			//loop through rows of the matrix where the corresponding node is outside of the domain
			#pragma omp parallel for
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
					double integral = penalty * gauss_quad_interface(fun_i, fun_i, DIRAC_DELTA, ELEM.bbox());

					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop though the other basis_s functions
					for (int j=i+1; j<ELEM.cursor_basis_s; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_s[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());

						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
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
						integral = penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());
						
						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
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
					double integral = penalty * gauss_quad_interface(fun_i, fun_i, DIRAC_DELTA, ELEM.bbox());
					
					#pragma omp atomic
					mat.coeffRef(global_i,global_i) += integral;

					//loop through any ancestor basis_a functions
					for (int j=i+1; j<ELEM.cursor_basis_a; j++)
					{
						const BasisFun_t& FUN_j = basis[ELEM.basis_a[j]];
						if (!FUN_j.is_active) {continue;}
						const size_t global_j = basis_all2active[FUN_j.list_index]; //DOF and matrix index
						assert(global_j<basis_active2all.size());

						//integrate FUN_i * FUN_j
						std::function<double(Point_t)> fun_j = [&FUN_j](Point_t point) -> double {return FUN_j.eval(point);};
						integral = penalty * gauss_quad_interface(fun_i, fun_j, DIRAC_DELTA, ELEM.bbox());
						
						#pragma omp atomic
						mat.coeffRef(global_i,global_j) += integral;

						#pragma omp atomic
						mat.coeffRef(global_j,global_i) += integral;
					}
				}
			}
		}
	};
}