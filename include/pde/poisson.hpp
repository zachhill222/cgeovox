#pragma once

#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/Core>

#include "mesh/Q1.hpp"
#include "fem/spmatrix_util.hpp"
#include "fem/assemble_homogeneous_matrices.hpp"


namespace gv::pde
{
	//solve -Laplace(u) = f over a domain meshed with voxels
	class Poisson
	{
	private:
		typedef gv::mesh::VoxelMeshQ1 Mesh_t;
		typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRow_t;
		typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpCol_t;

		SpRow_t stiff_mat_row;
		SpCol_t mass_mat_col;
		Eigen::VectorXd RHS;

		Eigen::ConjugateGradient<SpRow_t, Eigen::Lower|Eigen::Upper> solver;

		
	public:
		Poisson(Mesh_t &_mesh) : RHS(_mesh.nNodes()), mesh(_mesh), f(_mesh.nNodes()), u(_mesh.nNodes()) {}
		
		Mesh_t& mesh;
		Eigen::VectorXd f; //evaluation at each node
		Eigen::VectorXd u; //solution

		std::vector<size_t> dirichlet_bc_nodes;

		void setup()
		{
			//assemble integrating matrices
			gv::fem::make_integrating_matrices(mesh, mass_mat_col, stiff_mat_row);
			
			//set dirichlet boundary condition on the stiffness matrix (zero out rows corresponding to known data then set main diagonal entry to 1)
			gv::fem::set_dirichlet_bc(stiff_mat_row, dirichlet_bc_nodes);
			
			//set up right hand side. apply default dirichlet data of 0.
			RHS = mass_mat_col*f;
			for (size_t n=0; n<dirichlet_bc_nodes.size(); n++) {RHS[dirichlet_bc_nodes[n]] = 0.0;}

			//prepare solver
			solver.compute(stiff_mat_row);
		}


		void solve()
		{
			//set up initial guess to respect BC
			Eigen::VectorXd init_guess(mesh.nNodes());
			init_guess.fill(0.0);
			for (size_t n=0; n<dirichlet_bc_nodes.size(); n++) {init_guess[dirichlet_bc_nodes[n]] = RHS[dirichlet_bc_nodes[n]];}

			//solve
			u = solver.solveWithGuess(RHS, init_guess);
		}


		void save_as(const char* filename) const
		{
			mesh.save_as(filename);
			mesh._append_node_scalar_data(filename, u, "u", true);
			mesh._append_node_scalar_data(filename, f, "f", false);
			if (mesh.elem_marker.size()==mesh.nElems())
			{
				mesh._append_element_scalar_data(filename, mesh.elem_marker, "element_markers", true);
			}
		}
	};
}