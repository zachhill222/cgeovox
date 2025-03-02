#include "util/point.hpp"
#include "util/box.hpp"
#include "mesh/octree_mesh.hpp"

#include <iostream>
#include <random>


void test_octree_mesh(size_t N)
{
	using Point_t = gv::util::Point<3,double>;
	gv::util::Box<3> bbox(Point_t {0,0,0}, Point_t {1,1,1});

	//set up octree mesh
	gv::mesh::OctreeMesh octmesh(bbox);
	octmesh.reserve(5*N);

	
	//set up random number generator
	// std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    //divide mesh
    for (size_t i=0; i<N; i++)
    {
    	Point_t point;
    	for (int j=0; j<3; j++){
    		point[j] = dis(gen);
    	}

    	octmesh.divide(point);
    }

    std::cout << "saving mesh\n";
    octmesh.save("octmesh.vtk");
}


int main(int argc, char* argv[])
{	
	int N = 10;
	if (argc>1) {N=atoi(argv[1]);}

	test_octree_mesh(N);


	return 0;
}