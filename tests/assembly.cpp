#include "geometry/assembly.hpp"
#include "mesh/mesh.hpp"

#include <iostream>


void test_geometry(std::string filename, size_t N[3])
{
	std::cout << "load assembly\n";
	gv::geometry::Assembly<gv::geometry::SuperEllipsoid, 8> assembly(filename, "-rrr-eps-xyz-q");
	
	std::cout << "save solid\n";
	assembly.save_solid("outfiles/assembly_Geometry.vtk", 0.9*assembly.bbox(), N);

	std::cout << "save octree structure\n";
	assembly.view_octree_vtk("outfiles/assembly_OctreeStructure.vtk");
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

	test_geometry(argv[1], N);

	return 0;
}