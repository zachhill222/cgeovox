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

#define NDEBUG

void unit_sphere(const uint n_start, const uint n_refine)
{
	std::cout << "\n===== UNIT SPHERE START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	//set up DOF and error records
	std::vector<size_t> DOF(1+n_refine);
	std::vector<double> ERR(1+n_refine);

	using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::Sphere;
	std::string experiment_name = "charms_interface_unitSphere";
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
	opts.include_void = true;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;

	//set up problem
	gv::pde::PoissonCharmsInterface problem(2.0*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS
	problem.domain_marker = opts.solid_marker;
	// problem.penalty = 16;
	// problem.epsilon = 0.25;

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
	std::cout << "===== UNIT SPHERE END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}


void axial_prism(const uint n_start, const uint n_refine)
{
	std::cout << "\n===== AXIAL PRISM START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	//set up DOF and error records
	std::vector<size_t> DOF(1+n_refine);
	std::vector<double> ERR(1+n_refine);

	using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::Prism;
	std::string experiment_name = "charms_interface_axialPrism";
	std::string outdirectory = "./outfiles/";

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	Point_t radius {1.5, 1.0, 0.5};
	const Point_t radius_inv = Point_t{1,1,1}/radius;

	//set up exact solution (homogeneous Dirichlet BC, radius 1, f(x)=C*exact())
	auto exact = [radius_inv](Point_t point) -> double
	{
		Point_t cos_args = 1.570796327 * radius_inv * point;
		return std::cos(cos_args[0]) * std::cos(cos_args[1]) * std::cos(cos_args[2]);
	};

	//define assembly 
	Particle_t prism(radius, Point_t{0,0,0});
	gv::geometry::Assembly<Particle_t> assembly {prism};

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;

	//set up problem
	//increase meshed domain so that there are interface elements to be refined
	gv::pde::PoissonCharmsInterface problem(1.1*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS

	const double C = gv::util::squaredNorm(1.570796327*radius_inv);
	problem.rhs_fun = [C, radius_inv](Point_t point)
	{
		Point_t cos_args = 1.570796327 * radius_inv * point;
		return C * std::cos(cos_args[0]) * std::cos(cos_args[1]) * std::cos(cos_args[2]);
	};

	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
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

	std::cout << "===== AXIAL PRISM END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}



void prism(const uint n_start, const uint n_refine)
{
	std::cout << "\n===== PRISM START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	std::string experiment_name = "charms_interface_prism";
	std::string outdirectory = "./outfiles/";

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	//set up DOF and error records
	std::vector<size_t> DOF(1+n_refine);
	std::vector<double> ERR(1+n_refine);


	//define assembly 
	using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::Prism;
	using Quaternion_t = gv::util::Quaternion<double>;

	Quaternion_t quat {1,2,3,4};
	quat.normalize();

	Point_t radius {1.5, 1.0, 0.5};
	const Point_t radius_inv = Point_t{1,1,1}/radius;

	Particle_t prism(radius, Point_t{0,0,0}, quat);
	gv::geometry::Assembly<Particle_t> assembly {prism};
	
	

	

	//set up exact solution (homogeneous Dirichlet BC, radius 1, f(x)=C*exact())
	auto exact = [radius_inv, quat](Point_t point) -> double
	{
		Point_t cos_args = 1.570796327 * radius_inv * quat.rotate(point);
		return std::cos(cos_args[0]) * std::cos(cos_args[1]) * std::cos(cos_args[2]);
	};

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;

	//set up problem
	//increase meshed domain so that there are interface elements to be refined
	gv::pde::PoissonCharmsInterface problem(assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS

	const double C = gv::util::squaredNorm(1.570796327*radius_inv);
	problem.rhs_fun = [C, radius_inv, quat](Point_t point) -> double
	{
		Point_t cos_args = 1.570796327 * radius_inv * quat.rotate(point);
		return C * std::cos(cos_args[0]) * std::cos(cos_args[1]) * std::cos(cos_args[2]);
	};

	//solve on coarsest mesh
	std::cout << "===== coarse mesh =====" << std::endl;
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

	std::cout << "===== PRISM END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}


void assembly(const uint n_start, const uint n_refine, const std::string filename)
{
	std::cout << "\n===== ASSEMBLY START (n_start=" << n_start << ", n_refine= " << n_refine << ") =====" << std::endl;

	//set up timer
	std::time_t start = std::time(nullptr);

	// using Point_t = gv::util::Point<3,double>;
	using Particle_t = gv::geometry::SuperEllipsoid;
	std::string experiment_name = "charms_interface_assembly";
	std::string outdirectory = "./outfiles/";

	std::cout << "saving to: " << outdirectory << std::endl;
	std::cout << "prefix: " << experiment_name << std::endl;


	//define assembly 
	// std::string filename = "assemblies/particles_100.txt";
	std::cout << "getting assembly from: " << filename << std::endl;
	gv::geometry::Assembly<Particle_t,8> assembly(filename, "-rrr-eps-xyz-q");

	//set up meshing options
	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = true;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N[0] = n_start;
	opts.N[1] = n_start;
	opts.N[2] = n_start;
	opts.solid_marker = 1;
	opts.void_marker = 0;
	opts.interface_marker = 2;

	//set up problem
	gv::pde::PoissonCharmsInterface problem(1.5*assembly.bbox(), assembly, opts); //defualt homogeneous BC and f(x)=1 on the RHS
	problem.domain_marker = opts.void_marker;
	problem.penalty = 16;
	// problem.epsilon = 0.25;

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
	std::cout << "===== ASSEMBLY END (" << std::difftime(end,start) << " seconds) =====\n" << std::endl;
}



int main(int argc, char* argv[])
{
	uint n_start = 4;
	uint n_refine = 4;
	std::string filename = "./assmblies/particles_50.txt";

	if (argc>1) {n_start = atoi(argv[1]);}
	if (argc>2) {n_refine = atoi(argv[2]);}
	if (argc>3) {filename = argv[3];}

	// unit_sphere(n_start, n_refine);
	// axial_prism(n_start, n_refine);
	// prism(n_start, n_refine);
	assembly(n_start, n_refine, filename);

	return 0;
}