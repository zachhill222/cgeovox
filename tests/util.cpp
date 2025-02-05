#include "util/box.hpp"
#include "util/quaternion.hpp"

#include <iostream>

void test_box(){
	const int dim=3;

	//define bounds
	GeoVox::util::Point<dim> low;
	GeoVox::util::Point<dim> high;
	for (int i=0; i<dim; i++){
		low[i] = 0.0;
		high[i] = 1.0;
	}

	GeoVox::util::Box<dim> box(low,high);
	box *= 0.1;
	std::cout << box.tostr() << std::endl;
	for (int i=0; i<std::pow(2,dim); i++){
		std::cout << box.hexvertex(i) << std::endl;
	}
}


void test_quaternion(){
	GeoVox::util::Quaternion p(1,2,0,0);
	
	GeoVox::util::Quaternion q;
	GeoVox::util::Point3 axis(0,0,1);
	q.setrotation(3.1415926/2,axis);

	std::cout << "p= " << p << std::endl;
	std::cout << "q= " << q << std::endl;
	std::cout << "p+q= " << p+q << std::endl;
	std::cout << "pq= " << p*q << std::endl;
	std::cout << "qp= " << q*p << std::endl;
	p.normalize();
	std::cout << "p.normalize()= " << p << std::endl;
}



int main(int argc, char* argv[]){
	// test_box();
	test_quaternion();
	return 0;
}