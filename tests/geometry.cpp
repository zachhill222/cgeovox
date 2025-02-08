#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "geometry/particles.hpp"

#include <iostream>


using Point_t = gv::util::Point<3,double>;
using Quat_t = gv::util::Quaternion<double>;

void test_particles(double x, double y, double z)
{
	Point_t center {0,0,0};
	Point_t radius {1,2,3};
	Quat_t quaternion;
	quaternion.setrotation(0.5*3.1415926, Point_t {0,0,1});



	gv::geometry::Prism particle(radius, center, quaternion);

	Point_t point {x,y,z};
	std::cout << "particle contains " << point << "? " << particle.contains(point) << std::endl;
}


int main(int argc, char* argv[])
{	
	double x=1;
	double y=1;
	double z=1;

	if (argc>3)
	{
		x = atof(argv[1]);
		y = atof(argv[2]);
		z = atof(argv[3]);
	}

	test_particles(x,y,z);


	return 0;
}