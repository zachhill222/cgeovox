#include "util/point.hpp"
#include "util/box.hpp"
#include "util/quaternion.hpp"
#include "util/octree.hpp"
#include "util/polytope.hpp"

#include "util/octree_util.hpp"

#include <iostream>
#include <random>
#include <vector>

void test_point()
{
	gv::util::Point<3,double> p1 {1.0,2.0,3.0};
	gv::util::Point<3,double> p2(2.1*p1);
	p1+=p2;
	std::cout << (p1-p2).normalized() << std::endl;
	p1=1.001*p2;
	std::cout << p1 << " == " << p2 << " is " << (p1==p2) << std::endl;
	std::cout << "-p1: " << (-p1) << std::endl;
}

void test_box()
{
	const int dim=3;

	//define bounds
	gv::util::Point<dim,double> low;
	gv::util::Point<dim,double> high;
	for (int i=0; i<dim; i++){
		low[i]  = -1.0;
		high[i] = 1.0;
	}


	gv::util::Box<dim> box(low,high);
	// std::cout << "box:\n" << box.tostr() << std::endl;

	// box *= 0.1;
	std::cout << box << std::endl;
	for (int i=0; i<std::pow(2,dim); i++){
		std::cout << box.hexvertex(i) << std::endl;
	}

	std::cout << "center= " << box.center() << std::endl;
}


void test_quaternion()
{
	gv::util::Quaternion<double> p(1,2,0,0);
	
	gv::util::Quaternion<double> q;
	gv::util::Point<3,double> axis {0,0,1};
	q.setrotation(3.1415926/2,axis);

	std::cout << "p= " << p << std::endl;
	std::cout << "q= " << q << std::endl;
	std::cout << "p+q= " << p+q << std::endl;
	std::cout << "pq= " << p*q << std::endl;
	std::cout << "qp= " << q*p << std::endl;
	p.normalize();
	std::cout << "p.normalize()= " << p << std::endl;
}


int main(int argc, char* argv[])
{
	// test_point();
	test_box();
	// test_quaternion();
	// test_octree(atoi(argv[1]));

	return 0;
}