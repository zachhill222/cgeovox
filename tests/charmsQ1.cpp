#include "fem/charmsQ1.hpp"
#include "util/octree_util.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
	//set domain parameters
	gv::util::Box<3> domain(gv::util::Point<3,double> {0,0,0}, gv::util::Point<3,double> {1,2,3}); //box domain to be meshed
	gv::util::Point<3,size_t> N {3, 5, 7}; //number of elements in each cardinal direction
	
	//initialize coarsest mesh
	gv::fem::CharmsQ1Mesh mesh(domain,N);
	std::cout << "initialized mesh" << std::endl;
	
	// mesh.elements.reserve(1000);
	// mesh.vertices.reserve(1000);
	// mesh.basis.reserve(1000);

	//refine mesh
	mesh.h_refine(9);
	mesh.h_refine(8);
	mesh.h_refine(32);
	mesh.h_refine(65);
	mesh.h_refine(11);
	mesh.h_refine(7);
	mesh.h_refine(0);
	mesh.h_refine(91);
	mesh.h_refine(83);
	std::cout << "refined mesh" << std::endl;

	// //save mesh to view in ParaView
	mesh.save_as("./outfiles/charms_mesh.vtk");
	std::cout << "saved mesh" << std::endl;

	// for (size_t i=0; i<mesh.nBasis(); i++)
	// {
	// 	std::cout << mesh.basis[i] << std::endl;
	// }

	// // for (size_t i=0; i<mesh.basis.size(); i++)
	// // {
	// // 	std::cout << i << ":\t( " << mesh.basis[i].coord() << ")\tdepth= " << mesh.basis[i].depth << "\tactive= " << mesh.basis[i].is_active << std::endl;
	// // }

	//save octree structures for debugging
	// gv::util::view_octree_vtk(mesh.elements, "./outfiles/charms_element_octree.vtk");
	// // gv::util::view_octree_vtk(mesh.vertices, "vertices_octree.vtk");
	// // gv::util::view_octree_vtk(mesh.basis,    "basis_octree.vtk");
	return 0;
}