#include "geometry/assembly.hpp"
#include "geometry/particles.hpp"

#include "mesh/Q1.hpp"

#include "util/octree_util.hpp"

#include "pde/poisson.hpp"

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
	size_t n = 8;
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
	// using Particle_t = gv::geometry::SuperEllipsoid;
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
	opts.scale = 1.0;

	//mesh
	std::cout << "create Q1 mesh: " << std::flush;
	start = std::time(nullptr);
	// gv::mesh::VoxelMeshQ1 mesh(gv::util::Point<3,double>{r,r,r}, gv::util::Point<3,size_t>{n,n,n});
	gv::mesh::VoxelMeshQ1 mesh;
	assembly.create_voxel_mesh_Q1(mesh, r*assembly.bbox(), opts);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnNodes= " << mesh.nNodes() << "\tnElems= " << mesh.nElems() << "\n";

	//save mesh
	std::cout << "save Q1 mesh: " << std::flush;
	start = std::time(nullptr);
	mesh.save_as("mesh.vtk");
	// gv::util::view_octree_vtk(mesh._nodes);
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds" << std::endl;

	//check mesh for duplicate points
	for (size_t i=0; i<mesh.nNodes(); i++)
	{
		for (size_t j=i+1; j<mesh.nNodes(); j++)
		{
			if (mesh.node(i)==mesh.node(j))
			{
				std::cout << "duplicate point:\n\tvertex[" << i << "]= (" << mesh.node(i) << ")\n\tvertex[" << j << "]= (" << mesh.node(j) << ")" << std::endl;
			}
		}
	}


	//construct connectivity
	std::cout << "construct mesh connectivity (node2elem): " << std::flush;
	start = std::time(nullptr);
	mesh.compute_node2elem();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//construct boundary
	std::cout << "identify boundary: " << std::flush;
	start = std::time(nullptr);
	mesh.make_boundary();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	std::cout << "\tnBoundaryNodes= " << mesh.boundary(0).size() << "\n";

	//set up Poisson problem
	std::cout << "set up Poisson problem: " << std::flush;
	start = std::time(nullptr);
	gv::pde::Poisson problem(mesh);
	problem.dirichlet_bc_nodes = mesh.boundary(0);
	problem.f.fill(1.0);
	problem.setup();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";
	
	//solve
	std::cout << "solve Poisson problem: " << std::flush;
	start = std::time(nullptr);
	problem.solve();
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	//save solution
	std::cout << "save solution: " << std::flush;
	start = std::time(nullptr);
	problem.save_as("solution.vtk");
	end = std::time(nullptr);
	std::cout << std::difftime(end,start) << " seconds\n";

	return 0;
}