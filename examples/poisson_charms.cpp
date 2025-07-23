#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "mesh/Q1.hpp"
#include "fem/charmsQ1.hpp"

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
	// using Particle_t = gv::geometry::Prism;
	using Particle_t = gv::geometry::SuperEllipsoid;
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
	std::cout << "create standard voxel Q1 mesh: " << std::flush;
	start = std::time(nullptr);
	// gv::mesh::VoxelMeshQ1 mesh(gv::util::Point<3,double>{r,r,r}, gv::util::Point<3,size_t>{n,n,n});
	gv::mesh::VoxelMeshQ1 mesh;
	assembly.create_voxel_mesh_Q1(mesh, r*assembly.bbox(), opts);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnNodes= " << mesh.nNodes() << "\tnElems= " << mesh.nElems() << "\n";

	//convert standard mesh to charms mesh
	std::cout << "convert standard voxel Q1 mesh to a CHARMS voxel Q1 mesh: " << std::flush;
	start = std::time(nullptr);
	gv::fem::CharmsQ1Mesh charms_mesh(mesh);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds" << std::endl;


	// //construct connectivity
	// std::cout << "construct mesh connectivity (node2elem): " << std::flush;
	// start = std::time(nullptr);
	// mesh.compute_node2elem();
	// end = std::time(nullptr);
	// std::cout << std::difftime(end,start) << " seconds\n";

	// //construct boundary
	// std::cout << "identify boundary: " << std::flush;
	// start = std::time(nullptr);
	// mesh.make_boundary();
	// end = std::time(nullptr);
	// std::cout << std::difftime(end,start) << " seconds\n";
	// std::cout << "\tnBoundaryNodes= " << mesh.boundary(0).size() << "\n";

	// //set up Poisson problem
	// std::cout << "set up Poisson problem: " << std::flush;
	// start = std::time(nullptr);
	// gv::pde::Poisson problem(mesh);
	// problem.dirichlet_bc_nodes = mesh.boundary(0);
	// problem.f.fill(1.0);
	// problem.setup();
	// end = std::time(nullptr);
	// std::cout << std::difftime(end,start) << " seconds\n";
	
	// //solve
	// std::cout << "solve Poisson problem: " << std::flush;
	// start = std::time(nullptr);
	// problem.solve();
	// end = std::time(nullptr);
	// std::cout << std::difftime(end,start) << " seconds\n";

	//save solution
	std::cout << "save solution: " << std::flush;
	start = std::time(nullptr);
	// problem.save_as("solution.vtk");
	mesh.save_as("mesh.vtk");
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	return 0;
}