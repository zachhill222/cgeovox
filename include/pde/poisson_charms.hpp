#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "util/point.hpp"

#include "charms/assembly_charmsQ1.hpp"
#include "charms/charms_fem_util.hpp"

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include <omp.h>

namespace gv::pde
{
	double constant_fun(gv::util::Point<3,double> point) {return 1;}
	double zero_fun(gv::util::Point<3,double> point) {return 0;}

	//solve -Laplace(u) = f over a domain meshed with quasi-hierarchical voxel Q1 elements
	template <typename Assembly_t>
	class PoissonCharms
	{
	public:
		using Mesh_t        = gv::charms::AssemblyCharmsQ1Mesh<Assembly_t>;
		using Element_t     = typename Mesh_t::Element_t;  		//gv::charms::CharmsQ1Element;
		using BasisFun_t    = typename Mesh_t::BasisFun_t; 		//gv::charms::CharmsQ1BasisFun;
		using BasisList_t   = typename Mesh_t::BasisList_t; 	//gv::charms::CharmsQ1BasisFunOctree;
		using ElementList_t = typename Mesh_t::ElementList_t; 	//gv::charms::CharmsQ1ElementOctree;
		using VertexList_t  = typename Mesh_t::VertexList_t; 	//gv::charms::CharmsQ1BasisFun::VertexList_t;
		using Point_t       = typename Mesh_t::Point_t; 		//gv::util::Point<3,double>;
		using Box_t         = typename Mesh_t::Box_t; 			//gv::util::Box<3>;
		using Index_t       = typename Mesh_t::Index_t; 		//gv::util::Point<3,size_t>;
		using MeshOpts      = typename Mesh_t::MeshOpts; 		//gv::geometry::AssemblyMeshOptions;	
		using ScalarFun_t   = typename Mesh_t::ScalarFun_t; 	//std::function<double(Point_t)> //function from Point_t to double
		using VectorFun_t   = typename Mesh_t::VectorFun_t; 	//std::function<double(Point_t)> //function from Point_t to Point_t

		using SpRow_t       = Eigen::SparseMatrix<double, Eigen::RowMajor>; //sparse type for the stiffness matrix
		using SpCol_t       = Eigen::SparseMatrix<double, Eigen::ColMajor>; //sparse type for the mass matrix

		Mesh_t mesh;
		ScalarFun_t rhs_fun = constant_fun;
		ScalarFun_t bc_fun  = zero_fun;
		std::vector<double> u;

		PoissonCharms(const Box_t& domain, const Assembly_t& assembly, MeshOpts opts) :
			mesh(domain, opts, assembly)
			{
				assert(mesh.opts.include_interface); //the interface must be included for the mesh refinement
				u.resize(mesh.basis.size(), 0);
			}

		Box_t domain() const {return mesh.domain;}

		void refine_interface()
		{
			//refine all active elements with the interface marker
			for (size_t i=0; i<mesh.elem_active2all.size(); i++)
			{
				size_t e_idx = mesh.elem_active2all[i];
				if (mesh.element_marker[e_idx] == mesh.opts.interface_marker) {mesh.refine(e_idx, u);} //pass u so that the coefficients get updated to the fine mesh
			}
		}

		void refine_all()
		{
			//refine all active elements
			for (size_t i=0; i<mesh.elements.size(); i++)
			{
				if (mesh.elements[i].is_active) {mesh.refine(i, u);}
			}
		}

		void solve(int verbose=0)
		{
			//set up timer
			std::time_t start = std::time(nullptr);
			std::time_t end = std::time(nullptr);

			//ensure that the active basis functions and elements are recorded
			mesh.get_active_indices();


			//initialize matrices and vectors
			gv::charms::CharmsGalerkinMatrixConstructor matrix_constructor(mesh);
			SpRow_t stifMat;
			// SpCol_t massMat;
			Eigen::VectorXd U(mesh.basis_active2all.size());
			Eigen::VectorXd F(mesh.basis_active2all.size());

			//set up RHS and initial guess
			std::vector<double> f_vec(mesh.basis.size());
			// mesh._init_scalar_field(f_vec, rhs_fun);
			for (size_t i=0; i<mesh.basis_active2all.size(); i++)
			{
				size_t b_idx = mesh.basis_active2all[i];
				// F[i]  = f_vec[b_idx];
				U[i]  = u[b_idx];
			}
			
			// if (verbose>0) {std::cout << "making integrating matrices " << std::flush; start = std::time(nullptr);}
			// matrix_constructor.make_integrating_matrices(massMat, stifMat,
			// 	mesh.basis,
			// 	mesh.basis_active2all,
			// 	mesh.basis_all2active,
			// 	mesh.elements,
			// 	mesh.elem_active2all,
			// 	mesh.elem_all2active);
			// if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}

			if (verbose>0) {std::cout << "making stiffness matrix " << std::flush; start = std::time(nullptr);}
			matrix_constructor.make_stiff_matrix(stifMat,
				mesh.basis,
				mesh.basis_active2all,
				mesh.basis_all2active,
				mesh.elements,
				mesh.elem_active2all,
				mesh.elem_all2active);
			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}


			//create RHS vector
			// F = massMat*F;
			if (verbose>0) {std::cout << "integrating RHS vector " << std::flush; start = std::time(nullptr);}
			matrix_constructor.integrate_fun(F, rhs_fun,
				mesh.basis,
				mesh.basis_active2all,
				mesh.basis_all2active,
				mesh.elements,
				mesh.elem_active2all,
				mesh.elem_all2active);
			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}


			//find interior boundary
			if (verbose>0) {std::cout << "computing boundary " << std::flush; start = std::time(nullptr);}

			std::vector<size_t> boundary_global = mesh.active_basis_interior_boundary();
			std::vector<size_t> boundary; //the elements of boundary are ACTIVE indices
			for (size_t i=0; i<boundary_global.size(); i++)
			{
				boundary.push_back(mesh.basis_all2active[boundary_global[i]]);
			}

			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}


			//apply Dirichlet BC
			if (verbose>0) {std::cout << "applying Dirichlet BC " << std::flush; start = std::time(nullptr);}
			matrix_constructor.set_dirichlet_bc(stifMat, boundary);
			for (size_t i=0; i<boundary.size(); i++)
			{
				size_t b_idx = mesh.basis_active2all[boundary[i]];
				const Point_t point = mesh.basis[b_idx].coord();
				F[boundary[i]] = bc_fun(point);
			}
			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}

			//solve
			if (verbose>0) {std::cout << "solving " << std::flush; start = std::time(nullptr);}
			Eigen::ConjugateGradient<SpRow_t, Eigen::Lower|Eigen::Upper> solver;
			solver.compute(stifMat);
			U = solver.solveWithGuess(F, U);

			//update recorded u
			for (size_t i=0; i<mesh.basis_active2all.size(); i++)
			{
				u[mesh.basis_active2all[i]] = U[i];
			}

			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}
		}


		void save_as(std::string filename) const
		{
			assert(u.size()==mesh.basis.size());
			mesh.save_as(filename, u);
		}

		double error_L1(const ScalarFun_t& exact) const
		{
			assert(u.size()==mesh.basis.size());

			std::vector<double> gauss_locations {-0.7745966692414834, 0, 0.7745966692414834};
			std::vector<double> gauss_weights {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

			double error = 0;
			#pragma omp parallel for reduction(+:error)
			for (size_t e_idx=0; e_idx<mesh.elem_active2all.size(); e_idx++)
			{
				double local_err = 0;
				const Box_t voxel = mesh.elements[mesh.elem_active2all[e_idx]].bbox();
				const Point_t H = voxel.sidelength();
				const Point_t& center = voxel.center();

				Point_t ref_point;
				for (size_t i=0; i<gauss_locations.size(); i++) {
					ref_point[0] = gauss_locations[i];
					for (size_t j=0; j<gauss_locations.size(); j++) {
						ref_point[1] = gauss_locations[j];
						for (size_t k=0; k<gauss_locations.size(); k++) {
							ref_point[2] = gauss_locations[k];
							Point_t point = center + 0.5*H*ref_point;
							local_err += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * std::fabs(mesh._interpolate_scalar_field(u,point) - exact(point));
						}
					}
				}
				error += 0.125*H[0]*H[1]*H[2]*local_err;
			}

			return error;
		}

		double mean_error_L1(const ScalarFun_t& exact) const {return error_L1(exact)/mesh.volume();}
	};


	//solve -Laplace(u) = f over a domain meshed with quasi-hierarchical voxel Q1 elements
	template <typename Assembly_t>
	class PoissonCharmsInterface
	{
	public:
		using Mesh_t        = gv::charms::AssemblyCharmsQ1Mesh<Assembly_t>;
		using Element_t     = typename Mesh_t::Element_t;  		//gv::charms::CharmsQ1Element;
		using BasisFun_t    = typename Mesh_t::BasisFun_t; 		//gv::charms::CharmsQ1BasisFun;
		using BasisList_t   = typename Mesh_t::BasisList_t; 	//gv::charms::CharmsQ1BasisFunOctree;
		using ElementList_t = typename Mesh_t::ElementList_t; 	//gv::charms::CharmsQ1ElementOctree;
		using VertexList_t  = typename Mesh_t::VertexList_t; 	//gv::charms::CharmsQ1BasisFun::VertexList_t;
		using Point_t       = typename Mesh_t::Point_t; 		//gv::util::Point<3,double>;
		using Box_t         = typename Mesh_t::Box_t; 			//gv::util::Box<3>;
		using Index_t       = typename Mesh_t::Index_t; 		//gv::util::Point<3,size_t>;
		using MeshOpts      = typename Mesh_t::MeshOpts; 		//gv::geometry::AssemblyMeshOptions;	
		using ScalarFun_t   = typename Mesh_t::ScalarFun_t; 	//double (*)(Point_t); //function from Point_t to double
		using VectorFun_t   = typename Mesh_t::VectorFun_t; 	//Point_t (*)(Point_t); //function from Point_t to Point_t

		using SpRow_t       = Eigen::SparseMatrix<double, Eigen::RowMajor>; //sparse type for the stiffness matrix
		using SpCol_t       = Eigen::SparseMatrix<double, Eigen::ColMajor>; //sparse type for the mass matrix

		Mesh_t mesh;
		ScalarFun_t rhs_fun = constant_fun;
		ScalarFun_t bc_fun  = zero_fun;
		std::vector<double> u;

		double epsilon; //used for interface width
		double penalty; //used for Dirichlet BC
		int domain_marker = 0; //used to determine if H or 1-H should be used as the heaviside function

		PoissonCharmsInterface(const Box_t& domain, const Assembly_t& assembly, MeshOpts opts) :
			mesh(domain, opts, assembly)
			{
				assert(mesh.opts.include_interface); //the interface must be included for the mesh refinement
				u.resize(mesh.basis.size(), 0);

				epsilon = 1.5 * 0.333 * gv::util::norm1(domain.sidelength() / (Point_t) opts.N);
				penalty = 1.0/epsilon;
			}

		Box_t domain() const {return mesh.domain;}

		void refine_interface()
		{
			epsilon *= 0.5;
			penalty *= 2.0;


			//refine all active elements with the interface marker
			for (size_t i=0; i<mesh.elem_active2all.size(); i++)
			{
				size_t e_idx = mesh.elem_active2all[i];
				if (mesh.element_marker[e_idx] == mesh.opts.interface_marker) {mesh.refine(e_idx, u);} //pass u so that the coefficients get updated to the fine mesh
			}
		}

		void refine_all()
		{
			epsilon *= 0.5;
			penalty *= 2.0;

			//refine all active elements
			for (size_t i=0; i<mesh.elements.size(); i++)
			{
				if (mesh.elements[i].is_active) {mesh.refine(i, u);}
			}
		}

		void solve(int verbose=0)
		{
			//set up timer
			std::time_t start = std::time(nullptr);
			std::time_t end = std::time(nullptr);

			//ensure that the active basis functions and elements are recorded
			mesh.get_active_indices();


			//initialize matrices and vectors
			gv::charms::CharmsGalerkinMatrixConstructor matrix_constructor(mesh);
			matrix_constructor.penalty = penalty;

			SpRow_t stifMat;
			SpCol_t massMat;
			Eigen::VectorXd U(mesh.basis_active2all.size());
			Eigen::VectorXd F(mesh.basis_active2all.size());

			//set up RHS and initial guess
			// std::vector<double> f_vec(mesh.basis.size());
			// mesh._init_scalar_field(f_vec, rhs_fun);
			for (size_t i=0; i<mesh.basis_active2all.size(); i++)
			{
				size_t b_idx = mesh.basis_active2all[i];
				// F[i]  = f_vec[b_idx];
				U[i]  = u[b_idx];
			}

			if (verbose>0) {std::cout << "making stiffness matrix with weak Dirichlet BC " << std::flush; start = std::time(nullptr);}
			
			mesh._init_coarse_heaviside();
			ScalarFun_t heaviside;
			if (domain_marker==mesh.opts.solid_marker)
			{
				heaviside = [this](const Point_t& point) {return 1.0 - this->mesh.assembly.heaviside(point, this->epsilon);};
			}else if (domain_marker==mesh.opts.void_marker)
			{
				heaviside = [this](const Point_t& point) {return this->mesh.assembly.heaviside(point, this->epsilon);};
			}
			else {assert(false);}

			// ScalarFun_t heaviside = [this](const Point_t& point) {return 1.0 - this->mesh.coarse_heaviside(point);};
			// ScalarFun_t heaviside = [this](const Point_t& point) {return 1.0-this->mesh.assembly.heaviside(point, this->epsilon);};
			// ScalarFun_t penalty_kernel = [this](const Point_t& point) {return -this->mesh.assembly.dirac_delta(point, this->epsilon);}; //dirac delta, surface penalty
			ScalarFun_t penalty_kernel = [heaviside](const Point_t& point){return 1.0 - heaviside(point);}; //volume penalty

			// matrix_constructor.make_stiff_matrix(stifMat,
			// 	mesh.basis,
			// 	mesh.basis_active2all,
			// 	mesh.basis_all2active,
			// 	mesh.elements,
			// 	mesh.elem_active2all,
			// 	mesh.elem_all2active,
			// 	heaviside);

			matrix_constructor.make_stiff_matrix_with_BC(stifMat,
				mesh.basis,
				mesh.basis_active2all,
				mesh.basis_all2active,
				mesh.elements,
				mesh.elem_active2all,
				mesh.elem_all2active,
				heaviside,
				penalty_kernel);

			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}
			if (verbose>0) {std::cout << "epsilon= " << this->epsilon << "\t penalty= " << this->penalty << std::endl;}

			//create RHS vector
			if (verbose>0) {std::cout << "assembling rhs vector " << std::flush; start = std::time(nullptr);}
			ScalarFun_t rhs_fun_h = [heaviside, this](const Point_t& point) {return this->rhs_fun(point) * heaviside(point);};
			matrix_constructor.integrate_fun(F, rhs_fun,
				mesh.basis,
				mesh.basis_active2all,
				mesh.basis_all2active,
				mesh.elements,
				mesh.elem_active2all,
				mesh.elem_all2active,
				heaviside);
			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}

			//find interior boundary
			if (verbose>0) {std::cout << "computing boundary " << std::flush; start = std::time(nullptr);}

			std::vector<size_t> boundary_global = mesh.active_basis_interior_boundary();
			std::vector<size_t> boundary; //the elements of boundary are ACTIVE indices
			for (size_t i=0; i<boundary_global.size(); i++)
			{
				boundary.push_back(mesh.basis_all2active[boundary_global[i]]);
			}

			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}


			//apply Dirichlet BC
			// if (verbose>0) {std::cout << "applying weak (penalty) Dirichlet BC " << std::flush; start = std::time(nullptr);}
			
			// matrix_constructor.set_dirichlet_bc(stifMat,
			// 	mesh.basis,
			// 	mesh.basis_active2all,
			// 	mesh.basis_all2active,
			// 	mesh.elements,
			// 	mesh.elem_active2all,
			// 	mesh.elem_all2active,
			// 	dirac_delta);

			//SKIP IF THE BC IS HOMOGENEOUS
			// Eigen::VectorXd rhs_penalty(mesh.basis_active2all.size());
			// matrix_constructor.integrate_fun(rhs_penalty, bc_fun,
			// 	mesh.basis,
			// 	mesh.basis_active2all,
			// 	mesh.basis_all2active,
			// 	mesh.elements,
			// 	mesh.elem_active2all,
			// 	mesh.elem_all2active,
			// 	dirac_delta);
			// F += penalty * rhs_penalty;

			// if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}

			//solve
			if (verbose>0) {std::cout << "solving " << std::flush; start = std::time(nullptr);}
			Eigen::ConjugateGradient<SpRow_t, Eigen::Lower|Eigen::Upper> solver;
			solver.compute(stifMat);
			U = solver.solveWithGuess(F, U);

			//update recorded u
			for (size_t i=0; i<mesh.basis_active2all.size(); i++)
			{
				u[mesh.basis_active2all[i]] = U[i];
			}

			if (verbose>0) {end = std::time(nullptr); std::cout << "(" << std::difftime(end,start) << " seconds)" << std::endl;}
		}


		void save_as(std::string filename) const
		{
			assert(u.size()==mesh.basis.size());
			mesh.save_as(filename, u, epsilon);
		}

		double error_L1(const ScalarFun_t& exact) const
		{
			assert(u.size()==mesh.basis.size());

			std::vector<double> gauss_locations {-0.7745966692414834, 0, 0.7745966692414834};
			std::vector<double> gauss_weights {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

			ScalarFun_t heaviside = [this](Point_t point) {return 1.0-this->mesh.assembly.heaviside(point, 0.1*this->epsilon);};

			double error = 0;
			#pragma omp parallel for reduction(+:error)
			for (size_t e_idx=0; e_idx<mesh.elem_active2all.size(); e_idx++)
			{
				double local_err = 0;
				const Box_t voxel = mesh.elements[mesh.elem_active2all[e_idx]].bbox();
				const Point_t H = voxel.sidelength();
				const Point_t& center = voxel.center();

				Point_t ref_point;
				for (size_t i=0; i<gauss_locations.size(); i++) {
					ref_point[0] = gauss_locations[i];
					for (size_t j=0; j<gauss_locations.size(); j++) {
						ref_point[1] = gauss_locations[j];
						for (size_t k=0; k<gauss_locations.size(); k++) {
							ref_point[2] = gauss_locations[k];
							Point_t point = center + 0.5*H*ref_point;
							local_err += gauss_weights[i]*gauss_weights[j]*gauss_weights[k] * std::fabs(mesh._interpolate_scalar_field(u,point) - exact(point)) * heaviside(point);
						}
					}
				}
				error += 0.125*H[0]*H[1]*H[2]*local_err;
			}

			return error;
		}

		double mean_error_L1(const ScalarFun_t& exact) const {return error_L1(exact)/mesh.volume();}
	};
}