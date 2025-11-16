#include "util/point.hpp"
#include "util/quaternion.hpp"

#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "pde/poisson_charms.hpp"

#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>


void standard(const uint n_start, const uint n_refine)
{
	std::cout << "\n===== STANDARD START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	//set up DOF and error records
	std::vector<size_t> DOF(1+n_refine);
	std::vector<double> ERR(1+n_refine);

	using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::SuperEllipsoid;
	std::string experiment_name = "convergence.standard";
	std::string outdirectory = "./outfiles/";

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	//set up exact solution (homogeneous Dirichlet BC, radius 1, f(x)=1)
	auto exact = [](Point_t point) -> double {return 0.166666666666*(1.0-gv::util::squaredNorm(point));};

	//define assembly 
	Particle_t sphere(Point_t{1,1,1}, Point_t{0,0,0});
	gv::geometry::Assembly<Particle_t> assembly {sphere};

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void      = false;
	opts.include_interface = true;
	opts.include_solid     = true;
	opts.check_centroid    = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;

	//set up problem
	gv::pde::PoissonCharms problem(assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS


	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
	problem.mesh.get_active_indices();
	std::cout << problem.mesh << std::endl;
	problem.solve(1);
	ERR[0] = problem.mean_error_L1(exact);
	DOF[0] = problem.mesh.basis_active2all.size();
	std::cout << "mean L1 error= " << ERR[0] << std::endl;
	problem.save_as(outdirectory + experiment_name + "_0.vtk");
	//refine and solve
	for (uint i=1; i<=n_refine; i++)
	{
		std::cout << "\n===== refinement " << i << " =====" << std::endl;

		problem.refine_interface();
		problem.mesh.get_active_indices();
		std::cout << problem.mesh << std::endl;

		problem.solve(1);
		ERR[i] = problem.mean_error_L1(exact);
		DOF[i] = problem.mesh.basis_active2all.size();
		std::cout << "mean L1 error= " << ERR[i] << std::endl;
		problem.save_as(outdirectory + experiment_name + "_" + std::to_string(i) + ".vtk");
	}


	//print summary
	std::cout << "\nsummary:" << std::endl;
	std::cout << "Deg. of Freedom:\t";
	for (size_t i=0; i<n_refine+1; i++) {std::cout << DOF[i] << "\t";}
	std::cout << std::endl;

	std::cout << "Mean L1 Error  :\t";
	for (size_t i=0; i<n_refine+1; i++) {std::cout << ERR[i] << "\t";}
	std::cout << std::endl;

	//print time
	std::time_t end = std::time(nullptr);
	std::cout << "===== STANDARD END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}



void interface(const uint n_start, const uint n_refine)
{
	std::cout << "\n===== INTERFACE START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	//set up DOF and error records
	std::vector<size_t> DOF(1+n_refine);
	std::vector<double> ERR(1+n_refine);

	using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::Sphere;
	std::string experiment_name = "convergence.interface";
	std::string outdirectory = "./outfiles/";

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	//set up exact solution (homogeneous Dirichlet BC, radius 1, f(x)=1)
	auto exact = [](Point_t point) -> double {return 0.166666666666*(1.0-gv::util::squaredNorm(point));};

	//define assembly 
	Particle_t sphere(Point_t{1,1,1}, Point_t{0,0,0});
	gv::geometry::Assembly<Particle_t> assembly {sphere};

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void      = true;
	opts.include_interface = true;
	opts.include_solid     = true;
	opts.check_centroid    = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;

	//set up problem
	gv::pde::PoissonCharmsInterface problem(2.0*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS
	problem.domain_marker = opts.solid_marker;

	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
	problem.mesh.get_active_indices();
	std::cout << problem.mesh << std::endl;
	problem.solve(1);
	ERR[0] = problem.mean_error_L1(exact);
	DOF[0] = problem.mesh.basis_active2all.size();
	std::cout << "mean L1 error= " << ERR[0] << std::endl;
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
		ERR[i] = problem.mean_error_L1(exact);
		DOF[i] = problem.mesh.basis_active2all.size();
		std::cout << "mean L1 error= " << ERR[i] << std::endl;
		problem.save_as(outdirectory + experiment_name + "_" + std::to_string(i) + ".vtk");
	}


	//print summary
	std::cout << "\nsummary:" << std::endl;
	std::cout << "Deg. of Freedom:\t";
	for (size_t i=0; i<n_refine+1; i++) {std::cout << DOF[i] << "\t";}
	std::cout << std::endl;

	std::cout << "Mean L1 Error  :\t";
	for (size_t i=0; i<n_refine+1; i++) {std::cout << ERR[i] << "\t";}
	std::cout << std::endl;

	//print time
	std::time_t end = std::time(nullptr);
	std::cout << "===== INTERFACE END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}







int main(int argc, char* argv[])
{
	uint n_start  = 4;
	uint n_refine = 4;

	if (argc>1) {n_start  = atoi(argv[1]);}
	if (argc>2) {n_refine = atoi(argv[2]);}

	standard(n_start, n_refine);
	interface(n_start, n_refine);

	return 0;
}