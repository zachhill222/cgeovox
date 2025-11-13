#include "charms/charmsQ1.hpp"
#include "util/octree_util.hpp"
#include <Eigen/SparseCore>
#include <iostream>

//uncomment to disable assert()
// #define NDEBUG

int main(int argc, char const *argv[])
{
	//set domain parameters
	gv::util::Box<3> domain(gv::util::Point<3,double> {0,0,0}, gv::util::Point<3,double> {1,2,3}); //box domain to be meshed
	gv::util::Point<3,size_t> N {32, 32, 32}; //number of elements in each axis direction
	
	//initialize coarsest mesh
	gv::charms::CharmsQ1Mesh mesh(domain,N);
	std::cout << "initialized mesh" << std::endl;

	size_t n = 0;
	if (argc > 1) {n=atoi(argv[1]);}

	//refine mesh
	for (size_t i=0; i<n; i++)
	{
		size_t idx = mesh.nBasis()/2;
		std::cout << "refine basis " << idx << " at " << mesh.basis[idx].coord() << std::endl;
		// mesh.h_refine(idx);
		mesh.q_refine(idx);
	}
	std::cout << "refined mesh" << std::endl;

	//save mesh to view in ParaView
	mesh.save_as("./outfiles/charms_mesh_refined.vtk");
	std::cout << "saved mesh" << std::endl;
	
	//create mass and stiffness matrices
	Eigen::SparseMatrix<double> massMat, stifMat;
	Eigen::VectorXd ones(mesh.nBasis());
	Eigen::VectorXd x(mesh.nBasis());
	ones.fill(0.0);
	x.fill(0.0);
	for (size_t b_idx=0; b_idx<mesh.nBasis(); b_idx++)
	{
		if (mesh.basis[b_idx].depth==0)
		{
			ones[b_idx] = 1.0;
			x[b_idx]    = mesh.basis[b_idx].coord()[0];
		}
	}


	mesh.make_mass_matrix(massMat);
	std::cout << "1*M*1= " << ones.transpose()*(massMat*ones) << std::endl;

	mesh.make_stiff_matrix(stifMat);
	std::cout << "x*A*x= " << x.transpose()*(stifMat*x) << std::endl;


	//check for duplicate vertices
	std::cout << "DUPLICATE VERTICES: " << std::flush;
	size_t n_duplicates = 0;
	for (size_t i=0; i<mesh.nNodes(); i++)
	{
		for (size_t j=i+1; j<mesh.nNodes(); j++)
		{
			if (mesh.vertices[i]==mesh.vertices[j]) {n_duplicates+=1;}
		}
	}
	std::cout << n_duplicates << std::endl;


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