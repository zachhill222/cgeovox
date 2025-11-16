#include "util/point_octree.hpp"
#include "geometry/assembly.hpp"
#include "mesh/Q1.hpp"

// #include <armadillo>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <iostream>


void test_geometry(std::string filename, size_t N[3])
{
	std::cout << "load assembly\n";
	gv::geometry::Assembly<gv::geometry::SuperEllipsoid, 8> assembly(filename, "-rrr-eps-xyz-q");
	
	std::cout << "save solid\n";
	assembly.save_solid("outfiles/assembly_Geometry.vtk", assembly.bbox(), N);

	std::cout << "save octree structure\n";
	assembly.view_octree_vtk("outfiles/assembly_OctreeStructure.vtk");
}

void test_mesh(std::string filename, size_t N[3])
{
	std::cout << "load assembly\n";
	gv::geometry::Assembly<gv::geometry::SuperEllipsoid, 8> assembly(filename, "-rrr-eps-xyz-q");
	gv::util::Box<3> mesh_box = assembly.bbox();

	std::cout << "create unstructured voxel mesh\n";
	gv::mesh::VoxelMeshQ1 mesh;

	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = false;
	opts.include_solid = true;
	opts.check_centroid = true;
	opts.N[0] = N[0];
	opts.N[1] = N[1];
	opts.N[2] = N[2];
	opts.scale = 1;

	assembly.create_voxel_mesh_Q1(mesh, mesh_box, opts);
	std::cout << "\tnNodes= " << mesh.nNodes() << std::endl;
	std::cout << "\tnElem= "  << mesh.nElems() << std::endl;

	std::cout << "save mesh\n";
	mesh.save_as("outfiles/assembly_voxel_mesh.vtk");
}



int main(int argc, char* argv[])
{	
	size_t N[3] {50, 50, 50};
	if (argc>=2)
	{
		N[0] = atoi(argv[1]);
		N[1] = atoi(argv[1]);
		N[2] = atoi(argv[1]);
	}


	std::string filename = "./testdata/particles_50.txt";
	if (argc>=3) {filename=argv[2];}

	// test_geometry(filename, N);
	test_mesh(filename, N);

	return 0;
}