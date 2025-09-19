#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "charms/assembly_charmsQ1.hpp"
#include "charms/charms_fem_util.hpp"

#include <ctime>
#include <iostream>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>


int main(int argc, char* argv[])
{
	using Particle_t = gv::geometry::SuperEllipsoid;
	using Point_t = gv::util::Point<3,double>;
	typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRow_t;
	typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpCol_t;



	//set assembly file
	const char* file = "assemblies/sphere.txt";
	if (argc>1) {file = argv[1];}

	//set assembly file read flags
	const char* read_flags = "-rrr-eps-xyz-q"; 

	//set initial discretization size
	size_t n = 16;
	if (argc>2) {n = atoi(argv[2]);}

	//set padding ratio
	double r = 1.0;
	if (argc>3) {r = atof(argv[3]);}

	//set up timer
	std::time_t start;
	std::time_t end;


	//read geometry
	std::cout << "construct assembly: " << std::flush;
	// using Particle_t = gv::geometry::Prism;


	start = std::time(nullptr);
	gv::geometry::Assembly<Particle_t> assembly(file, read_flags);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N[0] = n;
	opts.N[1] = n;
	opts.N[2] = n;

	//mesh
	std::cout << "create coarse charms mesh: " << std::flush;
	start = std::time(nullptr);
	gv::charms::AssemblyCharmsQ1Mesh mesh(r*assembly.bbox(), opts, assembly);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << mesh << std::endl;


	std::cout << "refine coarse charms mesh: " << std::flush;
	mesh.p.resize(mesh.coarse_basis.size());
	start = std::time(nullptr);
	size_t inner_start=0;
	size_t next_inner_start=0;
	for (size_t i=1; i<=3; i++)
	{
		
		inner_start = next_inner_start;
		next_inner_start = mesh.coarse_elements.size();
		std::cout << "\n\nrefine depth: " << i << " (" << next_inner_start-inner_start << " elements)" << std::endl;

		for (size_t j=inner_start; j<next_inner_start; j++)
		{
			if (mesh.coarse_element_marker[j]==opts.interface_marker)
			{
				mesh.refine(j);
			}
		}
	}
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	mesh.get_active_indices();
	std::cout << mesh << std::endl;


	//get solution on coarsest mesh
	gv::charms::CharmsGalerkinMatrixConstructor matrix_constructor(mesh);
	SpRow_t stifMat;
	SpCol_t massMat;
	Eigen::VectorXd p(mesh.coarse_basis_active2all.size());
	Eigen::VectorXd f(mesh.coarse_basis_active2all.size());

	//set RHS vector
	auto fun = [](Point_t point) -> double {return 1;};
	std::vector<double> f_vec; f_vec.resize(mesh.coarse_basis.size());
	mesh._init_coarse_scalar_field(f_vec, fun);
	for (size_t i=0; i<mesh.coarse_basis_active2all.size(); i++)
	{
		size_t b_idx = mesh.coarse_basis_active2all[i];
		f[i]  = f_vec[b_idx];
	}
	
	//construct boundary
	std::cout << "identify boundary: " << std::flush;
	start = std::time(nullptr);
	std::vector<size_t> boundary_global = mesh.active_coarse_basis_interior_boundary();
	std::vector<size_t> boundary;
	for (size_t i=0; i<boundary_global.size(); i++)
	{
		boundary.push_back(mesh.coarse_basis_all2active[boundary_global[i]]);
	}


	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnBoundaryNodes= " << boundary.size() << "\n";

	//set up Poisson problem
	std::cout << "set up Poisson problem: " << std::endl;
	start = std::time(nullptr);

	std::cout << "\tmass matrix: " << std::flush;
	matrix_constructor.make_mass_matrix(massMat,
		mesh.coarse_basis,
		mesh.coarse_basis_active2all,
		mesh.coarse_basis_all2active,
		mesh.coarse_elements,
		mesh.coarse_elem_active2all,
		mesh.coarse_elem_all2active);
	std::cout << " done" << std::endl;

	std::cout << "\tstiffness matrix: " << std::flush;
	matrix_constructor.make_stiff_matrix(stifMat,
		mesh.coarse_basis,
		mesh.coarse_basis_active2all,
		mesh.coarse_basis_all2active,
		mesh.coarse_elements,
		mesh.coarse_elem_active2all,
		mesh.coarse_elem_all2active);
	std::cout << " done" << std::endl;

	std::cout << "\tapply dirichlet bc: " << std::flush;
	matrix_constructor.set_dirichlet_bc(stifMat, boundary);
	std::cout << " done" << std::endl;

	end = std::time(nullptr);
	std::cout << "\t" << std::difftime(end,start) << " seconds\n";
	


	//solve
	std::cout << "solve Poisson problem: " << std::flush;
	start = std::time(nullptr);
	Eigen::ConjugateGradient<SpRow_t, Eigen::Lower|Eigen::Upper> solver;
	solver.compute(stifMat);

	f = massMat*f;
	for (size_t i=0; i<boundary.size(); i++)
	{
		f[boundary[i]] = 0;
	}

	p = solver.solveWithGuess(f, p);

	for (size_t i=0; i<mesh.coarse_basis_active2all.size(); i++)
	{
		mesh.p[mesh.coarse_basis_active2all[i]] = p[i];
	}


	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//save solution
	std::cout << "save solution: " << std::flush;
	start = std::time(nullptr);
	mesh.save_as("outfiles/solution.vtk");
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	return 0;
}