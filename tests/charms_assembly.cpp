#include "charms/assembly_charmsQ1.hpp"
#include "util/octree_util.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <string>

//uncomment to disable assert()
// #define NDEBUG

template <class Mesh_t>
void print_mesh_info(const Mesh_t &mesh)
{
	std::cout << "n_vertices= " << mesh.vertices.size() << std::endl;
	std::cout << "n_coarse_basis_functions= " << mesh.coarse_basis.size() << std::endl;
	std::cout << "n_active_coarse_basis_functions= " << mesh.coarse_basis_active2all.size() << std::endl;
	std::cout << "n_coarse_elements= " << mesh.coarse_elements.size() << std::endl;
	std::cout << "n_active_coarse_elements= " << mesh.coarse_elem_active2all.size() << std::endl;

	std::cout << "n_fine_basis_functions= " << mesh.fine_basis.size() << std::endl;
	
	std::cout << "n_fine_elements= " << mesh.fine_elements.size() << std::endl;
}

double fun(const gv::util::Point<3,double> point) {return 1;}


// template <class Mesh_t>
// void print_integrating_matrix_values(const Mesh_t &mesh)
// {
// 	//create mass and stiffness matrices
// 	std::vector<size_t> basis_idx = mesh.active_coarse_basis();
// 	Eigen::SparseMatrix<double> massMat, stifMat;
// 	Eigen::VectorXd ones(basis_idx.size());
// 	Eigen::VectorXd x(basis_idx.size());
// 	ones.fill(0.0);
// 	x.fill(0.0);
// 	for (size_t i=0; i<basis_idx.size(); i++)
// 	{
// 		size_t b_idx = basis_idx[i];
// 		if (mesh.coarse_basis[b_idx].depth==0)
// 		{
// 			ones[b_idx] = 1.0;
// 			x[b_idx]    = mesh.coarse_basis[b_idx].coord()[0];
// 		}
// 	}


// 	gv::charms::make_mass_matrix(massMat, mesh);
// 	std::cout << "1*M*1= " << ones.transpose()*(massMat*ones) << std::endl;

// 	// mesh.make_stiff_matrix(stifMat);
// 	// std::cout << "x*A*x= " << x.transpose()*(stifMat*x) << std::endl;
// }




int main(int argc, char const *argv[])
{
	//set domain parameters
	std::string filename = "testdata/sphere.txt";
	gv::geometry::Assembly<gv::geometry::SuperEllipsoid,8> assembly(filename, "-rrr-eps-xyz-q");

	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = true;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N = gv::util::Point<3,size_t> {16,16,16};
	opts.void_marker = 0;
	opts.solid_marker = 1;
	opts.interface_marker = 2;


	//initialize coarsest mesh
	gv::charms::AssemblyCharmsQ1Mesh mesh(2*assembly.bbox(), opts, assembly);
	std::cout << "initialized mesh" << std::endl;

	size_t n = 0;
	if (argc > 1) {n=atoi(argv[1]);}


	//print coarse mesh info
	mesh.get_active_indices();
	mesh._init_coarse_scalar_field(mesh.p, fun);
	print_mesh_info(mesh);
	// print_integrating_matrix_values(mesh);
	mesh.save_as("./outfiles/assembly_charms_mesh_refined_0.vtk");

	//refine mesh
	size_t inner_start=0;
	size_t next_inner_start=0;
	for (size_t i=1; i<=n; i++)
	{
		
		inner_start = next_inner_start;
		next_inner_start = mesh.coarse_elements.size();
		std::cout << "\n\nrefine depth: " << i << " (" << next_inner_start-inner_start << " elements)" << std::endl;

		for (size_t j=inner_start; j<next_inner_start; j++)
		{
			if (mesh.coarse_element_marker[j]==opts.interface_marker)
			{
				// std::cout << "\trefine element " << j << std::endl;
				mesh.refine(j);
			}
		}

		mesh.get_active_indices();
		print_mesh_info(mesh);
		// print_integrating_matrix_values(mesh);
		mesh.save_as("./outfiles/assembly_charms_mesh_refined_" + std::to_string(i) + ".vtk");
	}
	std::cout << "refined mesh" << std::endl;
	
	


	//check for duplicate vertices
	// std::cout << "DUPLICATE VERTICES: " << std::flush;
	// size_t n_duplicates = 0;
	// for (size_t i=0; i<mesh.nNodes(); i++)
	// {
	// 	for (size_t j=i+1; j<mesh.nNodes(); j++)
	// 	{
	// 		if (mesh.vertices[i]==mesh.vertices[j]) {n_duplicates+=1;}
	// 	}
	// }
	// std::cout << n_duplicates << std::endl;


	// std::cout << "\nMASS MATRIX:\n" << massMat << std::endl;
	// std::cout << "\nSTIFFNESS MATRIX:\n" << stifMat << std::endl;

	// std::cout << "\n===============================================" << std::endl;
	// std::cout << "========== PRINT ALL BASIS FUNCTIONS ==========" << std::endl;
	// std::cout << "===============================================\n" << std::endl;
	// for (size_t i=0; i<mesh.nBasis(); i++)
	// {
	// 	std::cout << "basis: " << i << std::endl;
	// 	std::cout << mesh.basis[i] << std::endl;
	// }

	// std::cout << "================================================" << std::endl;
	// std::cout << "============== PRINT ALL ELEMENTS ==============" << std::endl;
	// std::cout << "================================================\n" << std::endl;
	// for (size_t i=0; i<mesh.nElems(); i++)
	// {
	// 	std::cout << "element: " << i << std::endl;
	// 	std::cout << mesh.elements[i] << std::endl;
	// }

	//save octree structures for debugging
	// gv::util::view_octree_vtk(mesh.elements, "./outfiles/charms_element_octree.vtk");
	// gv::util::view_octree_vtk(mesh.vertices, "./outfiles/vertices_octree.vtk");
	// gv::util::view_octree_vtk(mesh.basis,    "./outfiles/basis_octree.vtk");
	return 0;
}