#include "geometry/assembly.hpp"  //for creating assemblies of particles to be used as the problem domain
#include "geometry/particles.hpp" //for specifying what type of particles the assembly is made of
#include "pde/poisson_charms.hpp" //poisson solvers that use the charms method (both standard and level set/interface)

#include <ctime>
#include <iostream>
#include <string>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>


template <typename Particle_t, double pad=0, bool domain_is_void=true>
void charms_interface(const uint n_start, const uint n_refine, const std::string filename, const std::string experiment_name, const std::string outdirectory)
{
	std::cout << "\n===== BEGIN (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;
	std::cout << "METHOD = CHARMS_INTERFACE" << std::endl;


	//set up timer
	std::time_t start = std::time(nullptr);

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	//define assembly 
	const double scale = 1000;
	std::cout << "getting assembly/domain from: " << filename << std::endl;
	gv::geometry::Assembly<Particle_t,8> assembly(filename, "-rrr-eps-xyz-q", scale);

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void      = true;
	opts.include_interface = true;
	opts.include_solid     = true;
	opts.check_centroid    = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;
	opts.void_marker      = 0;
	opts.interface_marker = 1;
	opts.solid_marker     = 2;

	//set up problem
	gv::pde::PoissonCharmsInterface problem((1.0+pad)*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS
	// problem.penalty = 16;

	if (domain_is_void) {problem.domain_marker = opts.void_marker;}
	else {problem.domain_marker = opts.solid_marker;}
	
	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
	problem.mesh.get_active_indices();
	std::cout << problem.mesh << std::endl;
	problem.solve(1);
	problem.save_as(outdirectory + experiment_name + "_0.vtk");
	//refine and solve
	for (uint i=1; i<=n_refine; i++)
	{
		std::cout << "\n===== refinement " << i << " ===== " << std::flush;
		std::time_t inner_start = std::time(nullptr);
		problem.refine_interface();
		std::time_t inner_end = std::time(nullptr);
		std::cout << "(" << std::difftime(inner_end,inner_start) << " seconds to refine) =====" << std::endl;

		problem.mesh.get_active_indices();
		std::cout << problem.mesh << std::endl;

		problem.solve(1);
		problem.save_as(outdirectory + experiment_name + "_" + std::to_string(i) + ".vtk");
	}

	//print time
	std::time_t end = std::time(nullptr);
	std::cout << "===== END " << " (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}

template <typename Particle_t, double pad=0, bool domain_is_void=true>
void charms_standard(const uint n_start, const uint n_refine, const std::string filename, const std::string experiment_name, const std::string outdirectory)
{
	std::cout << "\n===== BEGIN (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;
	std::cout << "METHOD = CHARMS_STANDARD" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	//define assembly 
	const double scale = 1000;
	std::cout << "getting assembly/domain from: " << filename << std::endl;
	gv::geometry::Assembly<Particle_t,8> assembly(filename, "-rrr-eps-xyz-q", scale);

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_interface = true;
	if (domain_is_void) { opts.include_solid = false; opts.include_void = true;}
	else { opts.include_solid = true; opts.include_void = false;}

	opts.check_centroid    = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;
	opts.void_marker      = 0;
	opts.interface_marker = 1;
	opts.solid_marker     = 2;

	//set up problem
	gv::pde::PoissonCharms problem((1.0+pad)*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS

	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
	problem.mesh.get_active_indices();
	std::cout << problem.mesh << std::endl;
	problem.solve(1);
	problem.save_as(outdirectory + experiment_name + "_0.vtk");
	//refine and solve
	for (uint i=1; i<=n_refine; i++)
	{
		std::cout << "\n===== refinement " << i << " ===== " << std::flush;
		std::time_t inner_start = std::time(nullptr);
		problem.refine_interface();
		std::time_t inner_end = std::time(nullptr);
		std::cout << "(" << std::difftime(inner_end,inner_start) << " seconds to refine) =====" << std::endl;

		problem.mesh.get_active_indices();
		std::cout << problem.mesh << std::endl;

		problem.solve(1);
		problem.save_as(outdirectory + experiment_name + "_" + std::to_string(i) + ".vtk");
	}

	//print time
	std::time_t end = std::time(nullptr);
	std::cout << "===== END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}



int main(int argc, char* argv[])
{
	uint n_start                 = 4;
	uint n_refine                = 4;
	std::string filename         = "./assmblies/sphere.txt";
	std::string experiment_name  = "sphere";

	if (argc>1) {n_start         = atoi(argv[1]);}
	if (argc>2) {n_refine        = atoi(argv[2]);}
	if (argc>3) {filename        = argv[3];}
	if (argc>4) {experiment_name = argv[4];}

	using Particle_t = gv::geometry::Ellipsoid; //treat all particles as spheres. SuperEllipsoids seem to have a problem.
	constexpr double pad = 0; //add extra space around the particles (0.5 for small assemblies and 0 for large)
	constexpr bool domain_is_void = true; //determine if the problem domain is the void+interface or solid+interface

	//compile program to use the interface method
	#ifdef INTERFACE
		//set up directory for solutions
		std::string outdirectory = "./charms_interface/" + experiment_name + "/";
		if (argc>5) {outdirectory = argv[5];}

		try {
			charms_interface<Particle_t,pad,domain_is_void>(n_start, n_refine, filename, experiment_name, outdirectory);
		} catch (...) {
			std::cout << "\n\tPROGRAM CRASHED!" << std::endl;
			return 1;}
		
	#endif

	//compile program to use the standard method
	#ifdef STANDARD
		//set up directory for solutions
		std::string outdirectory = "./charms_standard/" + experiment_name + "/";
		if (argc>5) {outdirectory = argv[5];}

		try {
			charms_standard<Particle_t,pad,domain_is_void>(n_start, n_refine, filename, experiment_name, outdirectory);
		} catch (...) {
			std::cout << "\n\tPROGRAM CRASHED!" << std::endl;
			return 1;
		}
		
	#endif
	return 0;
}