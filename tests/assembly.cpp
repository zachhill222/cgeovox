#include "geometry/assembly.hpp"
#include "mesh/mesh.hpp"
#include "mesh/Q1.hpp"

#include <armadillo>
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
	arma::sp_mat mass_mat;
	assembly.create_voxel_mesh_Q1(mesh, mesh_box, N);
	
	std::cout << "extract solid region\n";
	gv::mesh::VoxelMeshQ1 solid_mesh;
	mesh.get_subdomain(1, solid_mesh);

	std::cout << "make mass matrix\n";
	std::cout << "\tnNodes= " << solid_mesh.nNodes() << std::endl;
	std::cout << "\tnElem= "  << solid_mesh.nElems() << std::endl;
	
	solid_mesh.make_mass_matrix(mass_mat);
	arma::vec ones(solid_mesh.nNodes(), arma::fill::ones);
	double approx_total_volume = arma::dot(ones, mass_mat*ones);
	std::cout << "\t1*M*1= " << approx_total_volume << std::endl;
	std::cout << "\tn_nonzero= " << mass_mat.n_nonzero << std::endl;

	if (solid_mesh.nNodes() < 100)
	{
		std::cout << "\nmass_matrix:\n";
		arma::mat dense(27*mass_mat);
		// dense.print();
		dense.save( arma::csv_name("mass_matrix.csv"));
	}


	std::cout << "save solid_mesh\n";
	solid_mesh.saveas("outfiles/assembly_voxel_mesh.vtk");

	// std::cout << "create and save solid region\n";
	// gv::mesh::VoxelMeshQ1 solid_mesh;
	// mesh.get_subdomain(1, solid_mesh);

	// // solid_mesh.saveas("outfiles/assembly_solid.vtk");

	// std::cout << "create and save void region\n";
	// gv::mesh::VoxelMeshQ1 void_mesh;
	// mesh.get_subdomain(0, void_mesh);
	// // void_mesh.saveas("outfiles/assembly_void.vtk");

	// std::cout << "create and save interface region\n";
	// gv::mesh::VoxelMeshQ1 interface_mesh;
	// mesh.get_subdomain(2, interface_mesh);
	// interface_mesh.saveas("outfiles/assembly_interface.vtk");
}



int main(int argc, char* argv[])
{	
	size_t N[3] {50, 50, 50};
	if (argc>4)
	{
		N[0] = atoi(argv[2]);
		N[1] = atoi(argv[3]);
		N[2] = atoi(argv[4]);
	}
	else if (argc==3)
	{
		N[0] = atoi(argv[2]);
		N[1] = atoi(argv[2]);
		N[2] = atoi(argv[2]);
	}

	// test_geometry(argv[1], N);
	test_mesh(argv[1], N);

	return 0;
}