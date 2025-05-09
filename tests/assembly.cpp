#include "geometry/assembly.hpp"
#include "mesh/mesh.hpp"
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

	assembly.create_voxel_mesh_Q1(mesh, mesh_box, opts);
	std::cout << "\tnNodes= " << mesh.nNodes() << std::endl;
	std::cout << "\tnElem= "  << mesh.nElems() << std::endl;


	std::cout << "\nmake mass matrix\n";

	Eigen::SparseMatrix<double> spmat;
	mesh.make_mass_matrix(spmat);

	Eigen::VectorXd ones(mesh.nNodes());
	ones.fill(1.0);
	double approx_total_volume = ones.dot(spmat*ones);
	
	std::cout << "\t1*M*1= " << approx_total_volume << std::endl;
	std::cout << "\tn_nonzero= " << spmat.nonZeros() << std::endl;

	std::cout << "\nmake stiffness matrix\n";
	mesh.make_stiffness_matrix(spmat);
	
	//create vector of x_1 locations at each degree of freedom
	Eigen::VectorXd x(mesh.nNodes());
	for (size_t i=0; i<mesh.nNodes(); i++)
	{
		x[i] = mesh.nodes(i)[1];
	}

	approx_total_volume = x.dot(spmat*x);
	std::cout << "\tx*M*x= " << approx_total_volume << std::endl;

	std::cout << "save mesh\n";
	mesh.save_as("outfiles/assembly_voxel_mesh.vtk");
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