#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "mesh/homo_mesh.hpp"
// #include "util/octree_util.hpp"

#include <iostream>
#include <random>
#include <vector>

//helpful data types
using Point_t = gv::util::Point<3,double>;
using Box_t = gv::util::Box<3>;

void test_octree(size_t N)
{
	//set up octree container for points
	Box_t bbox(Point_t {0,0,0}, Point_t {1,1,1});
	const int n_data_per_leaf = 4;
	gv::util::PointOctree<3,n_data_per_leaf>* octree = new gv::util::PointOctree<3,n_data_per_leaf>(bbox,16); //test using a pointer
	// gv::util::OctreeInspector inspector(*octree); //print and vtk functions
	std::vector<Point_t> vector;

	//set up random number generator
	// std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    //insert random points
    for (size_t i=0; i<N; i++)
    {
    	Point_t point;
    	for (int j=0; j<3; j++){
    		point[j] = dis(gen);
    	}

    	octree->push_back(point);
    	vector.push_back(point);
    	std::cout << "index  " << i << " (" << point << ") capacity: " << octree->size() << "/" << octree->capacity() << std::endl;
    }

    //verify that the octree and vector have the same linear access
    if (vector.size()!=octree->size() or octree->size()!=N)
    {
    	std::cout << "ERROR: incorrect size after octree insertion\n";
    }
    bool success = true;
    for (size_t i=0; i<N; i++)
    {
    	bool same_point = vector[i]==octree->at(i);
    	if (!same_point) {std::cout << "ERROR: vector and octree mismatch at index " << i << std::endl;}
    	success = success and same_point;
    }
    if (success) {std::cout << "\nSUCCESS: vector and octree linear access is the same" << std::endl;}

    //verify that the octree can find every element with the correct index
    success = true;
	for (size_t i=0; i<octree->size(); i++)
    {
    	bool same_index = i==octree->find(octree->at(i));
    	if (!same_index) {std::cout << "ERROR: octree.find(octree[i])!=i  (i=" << i << ")" << std::endl;}
    	success = success and same_index;
    }
    if (success) {std::cout << "\nSUCCESS: octree.find(octree[i])=i for all i" << std::endl;}

    //verify that we can extract all of the correct data associated with boxes
    Box_t new_bbox(Box_t{Point_t{0,0,0},Point_t{0.5,0.5,0.5}});
    std::vector<size_t> all_points = octree->get_data_indices(octree->bbox());
    if (all_points.size()!=octree->size()) {std::cout << "ERROR: did not get all data";}

    std::vector<size_t> some_points = octree->get_data_indices(new_bbox);
	std::cout << "there are " << some_points.size() << " points in the box " << new_bbox << std::endl;


    //print octree structure
    // std::cout << "\nPRINT OCTREE STRUCTURE" << std::endl;
    // inspector.print();

    //resize and trim bounding box
    // gv::util::PointOctree<3,n_data_per_leaf>* octree2 = new gv::util::PointOctree<3,n_data_per_leaf>(new_bbox, octree->size());
    // for (size_t idx=0; idx<octree->size(); idx++) {octree2->push_back(octree->at(idx));}
    // gv::util::PointOctree<3,n_data_per_leaf>* octree2 = new gv::util::PointOctree<3,n_data_per_leaf>(*octree, new_bbox);

    gv::util::PointOctree<3,n_data_per_leaf>* octree2 = octree;
    octree2->set_bbox(new_bbox);

    std::cout << "\nmaking smaller octree (octree2): " << octree2->size() << "/" << octree2->capacity() << std::endl;
    std::cout << "\tinitial bbox= " << bbox << std::endl;
    std::cout << "\tnew bbox= " << new_bbox << std::endl;

    some_points.clear();
    some_points = octree2->get_data_indices(new_bbox);
	std::cout << "there are (still?) " << some_points.size() << " points in the box " << new_bbox << std::endl;

    //print octree structure
    // std::cout << "\nPRINT OCTREE STRUCTURE" << std::endl;
    // inspector.print();

    //verify that the octree can find every element with the correct index
    success = true;
	for (size_t i=0; i<octree2->size(); i++)
    {
    	bool same_index = i==octree2->find(octree2->at(i));
    	if (!same_index) {std::cout << "ERROR: octree2.find(octree2[i])!=i  (i=" << i << ")" << std::endl;}
    	success = success and same_index;
    }
    if (success) {std::cout << "\nSUCCESS: octree2.find(octree2[i])=i for all i" << std::endl;}


    
    for (size_t i=0; i<N; i++)
    {
    	const Point_t& point = vector[i];
    	if (octree2->contains(point))
    	{
    		size_t j = octree2->find(point);
	    	std::cout << "vector: " << i << "\t" << vector[i] << "\t| octree: " << j << "\t" << octree2->at(j);
	    	std::cout << "\t (diff= " << vector[i]-octree2->at(j) << ")" << std::endl;
    	}
    	else
    	{
    		std::cout << "vector: " << i << "\t" << vector[i] << " trimmed from octree2" << std::endl;
    		continue;
    	}
	}

    //print all of the octee nodes
   	std::cout << "\n\nlist octree2 data" << std::endl;
   	success = true;
    for (size_t i=0; i<octree2->size(); i++)
    {
    	bool same_index = i==octree2->find(octree2->at(i));
    	if (!same_index) {std::cout << "ERROR: can't find data at index " << i << " (" << octree2->at(i) << ")" << std::endl;}
    	success = success and same_index;
    }
    if (success) {std::cout << "\nSUCCESS: octree2.find(octree2[i])=i for all i" << std::endl;}


    //free memory
    delete octree;
    // delete octree2;
}

int main(int argc, char* argv[])
{	
	int N = 10;
	if (argc>1) {N=atoi(argv[1]);}

	// test_octree_mesh(N);
	test_octree(N);

	return 0;
}