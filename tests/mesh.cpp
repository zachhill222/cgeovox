#include "util/point.hpp"
#include "util/box.hpp"
#include "mesh/octree_mesh.hpp"

#include <iostream>
#include <random>


void test_octree_mesh(size_t N)
{
	using Point_t = gv::util::Point<3,float>;
	gv::util::Box bbox(Point_t {0,0,0}, Point_t {1,1,1});

	//set up octree mesh
	gv::mesh::OctreeMesh<float> octmesh(bbox);
	octmesh.reserve(5*N);

	
	//set up random number generator
	// std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    //divide mesh
    for (size_t i=0; i<N; i++)
    {
    	Point_t point;
    	for (int j=0; j<3; j++){
    		point[j] = dis(gen);
    	}

    	// std::cout << i << " (started) " << point << std::endl;

    	octmesh.divide_center(point);
    	// octmesh.divide(point);

    	// std::cout << i << " (finished) " << point << " size= " << octmesh.size() << std::endl;
    }

    std::cout << "saving mesh\n";
    octmesh.save("octmesh.vtk");

}


int main(int argc, char* argv[])
{
	test_octree_mesh(atoi(argv[1]));

	return 0;
}