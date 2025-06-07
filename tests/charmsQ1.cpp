#include "util/point.hpp"
#include "util/box.hpp"
#include "fem/charmsQ1.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
	//set domain parameters
	gv::util::Box<3> domain(gv::util::Point<3,double> {0,0,0}, gv::util::Point<3,double> {1,2,3}); //box domain to be meshed
	gv::util::Point<3,size_t> N {1, 2, 3}; //number of elements in each cardinal direction
	
	//initialize coarsest mesh
	gv::fem::CharmsQ1Mesh mesh(domain,N);
	// mesh.vertices.reserve(100000);
	// mesh.elements.reserve(100000);
	// mesh.basis.reserve(100000);
	std::cout << "initialized mesh" << std::endl;
	
	//refine mesh
	mesh.refine_basis(9);
	mesh.refine_basis(25);
	mesh.refine_basis(32);
	mesh.refine_basis(65);
	mesh.refine_basis(11);
	mesh.refine_basis(7);
	// for (int k=0; k<2; k++) {mesh.refine_basis(mesh.nBasis()-1);}
	std::cout << "refined mesh" << std::endl;

	//save mesh to view in ParaView
	mesh.save_as("test_mesh.vtk");

	return 0;
}