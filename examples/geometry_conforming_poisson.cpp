#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/Core>

#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "util/Point.hpp"

#include "mesh/Q1.hpp"


#include <ctime>
#include <iostream>








int main(int argc, char* argv[])
{
	//set assembly file
	const char* file = "assemblies/sphere.txt";
	if (argc>1) {file = argv[1];}

	//set assembly file read flags
	const char* read_flags = "-rrr-eps-xyz-q"; 

	//set discretization size
	size_t n = 32;
	if (argc>2) {n = atoi(argv[2]);}

	//set padding ratio
	double r = 1.0;
	if (argc>3) {r = atof(argv[3]);}

	//set up timer
	std::time_t start;
	std::time_t end;


	//read geometry
	std::cout << "construct assembly: " << std::flush;
	using Particle_t = gv::geometry::Prism;
	start = std::time(nullptr);
	gv::geometry::Assembly<Particle_t> assembly(file, read_flags);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = true;
	opts.N[0] = n;
	opts.N[1] = n;
	opts.N[2] = n;

	//mesh
	std::cout << "create Q1 mesh: " << std::flush;
	start = std::time(nullptr);
	gv::mesh::VoxelMeshQ1 mesh;
	assembly.create_voxel_mesh_Q1(mesh, r*assembly.bbox(), opts);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnNodes= " << mesh.nNodes() << "\tnElems= " << mesh.nElems() << "\n";

	//construct connectivity
	std::cout << "construct mesh connectivity (node2elem): " << std::flush;
	start = std::time(nullptr);
	mesh.compute_connectivity();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//construct boundary
	std::cout << "identify boundary: " << std::flush;
	start = std::time(nullptr);
	mesh.make_boundary();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnBoundaryNodes= " << mesh.boundary(0).size() << "\n";

	//construct integrating matrices
	std::cout << "construct integrating matrices: " << std::flush;
	start = std::time(nullptr);
	Eigen::SparseMatrix<double, Eigen::RowMajor> A;
	Eigen::SparseMatrix<double, Eigen::ColMajor> M;
	mesh.make_integrating_matrices(M, A);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnnz= " << M.nonZeros() << " (" << 100.0*M.nonZeros()/(mesh.nNodes()*mesh.nNodes()) << "%)\n";

	//set up RHS
	Eigen::VectorXd RHS(mesh.nNodes());
	RHS.fill(0.0);

	// double vol = RHS.dot(M*RHS);
	// double scalar_rhs = 100.0/vol;
	// RHS.fill(scalar_rhs);
	// RHS  = M*RHS;


	//apply boundary conditions
	std::cout << "apply boundary conditions: " << std::flush;
	start = std::time(nullptr);
	for (size_t i=0; i<mesh.boundary(0).size(); i++)
	{
		size_t node_idx = mesh.boundary(0)[i];
		if (mesh.nodes(node_idx)[0]==1.0)
		{
			RHS[node_idx] = 1.0;
		}
		else
		{
			RHS[node_idx] = 0.0;
		}

		// RHS[node_idx] = 0.0;
		A.row(node_idx) *= 0.0;
		A.coeffRef(node_idx,node_idx) = 1.0;
	}
	// RHS[1320] = 0.0;
	// A.row(1320) *= 0.0;
	// A.coeffRef(1320,1320) = 1.0;
	A.makeCompressed();

	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";


	//solve problem
	std::cout << "solve Poisson problem: " << std::flush;
	Eigen::SparseMatrix<double, Eigen::ColMajor> LHS = A;
	LHS.makeCompressed();

	start = std::time(nullptr);
	Eigen::VectorXd u(mesh.nNodes());

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver;
	// Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<double>> solver;
	// Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	// Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
	solver.compute(LHS);

	// Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	// solver.analyzePattern(LHS);
	// solver.factorize(LHS);

	u = solver.solveWithGuess(RHS,RHS);

	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnIter= " << solver.iterations() << "\test. err= " << solver.error() << "\n"; //only for iterative solvers


	//save solution
	std::cout << "save solution: " << std::flush;
	start = std::time(nullptr);
	mesh.save_as("solution.vtk");
	mesh._append_node_scalar_data("solution.vtk", u, "poisson_solution");
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";


	//save boundary_node marker for verification
	Eigen::VectorXd is_boundary(mesh.nNodes());
	is_boundary.fill(0.0);
	for (size_t i=0; i<mesh.boundary(0).size(); i++)
	{
		is_boundary[mesh.boundary(0)[i]] = 1.0;
	}
	mesh._append_node_scalar_data("solution.vtk", is_boundary, "is_boundary", false);

	//save exact solution (sphere with radius 1)
	// Eigen::VectorXd exact_solution(mesh.nNodes());
	// for (size_t i=0; i<mesh.nNodes(); i++)
	// {
	// 	gv::util::Point<3,double> x = mesh.nodes(i);
	// 	// exact_solution[i] = scalar_rhs*(1-x.squaredNorm())/6;
	// 	// exact_solution[i] = scalar_rhs*(1-x[0]*x[0])*(1-x[1]*x[1])*(1-x[2]*x[2])/8;
	// }
	// mesh._append_node_scalar_data("solution.vtk", exact_solution, "exact_solution", false);


	return 0;
}