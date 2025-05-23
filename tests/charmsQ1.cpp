#include "util/point.hpp"
#include "fem/charms_Q1_mesh.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
	
	gv::fem::CharmsQ1_3DMesh mesh;
	mesh.divide(mesh.active_elements[0]);
	mesh.divide(mesh.active_elements[1]);
	mesh.divide(mesh.active_elements[2]);
	mesh.divide(mesh.active_elements[8]);

	mesh.save_as("test_mesh.vtk");


	return 0;
}