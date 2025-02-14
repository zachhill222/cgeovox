#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"
#include "geometry/assembly.hpp"

#include <iostream>


using Point = gv::util::Point<3,double>;
using Quat = gv::util::Quaternion<double>;

void test_particles(double x, double y, double z)
{
	Point center {0,0,0};
	Point radius {1,1,1};
	Quat quaternion {1,0,0,0};
	// quaternion.setrotation(0.25*3.1415926, Point {0,0,1});



	gv::geometry::Prism P1(radius, center, quaternion);
	double eps[2] {1,1};
	gv::geometry::SuperEllipsoid P2(1.73*radius, eps, {2,2,2}, {1,0,0,0});

	Point point {x,y,z};
	std::cout << "P1 contains " << point << "? " << P1.contains(point) << std::endl;
	std::cout << "P2 contains " << point << "? " << P2.contains(point) << std::endl;
	std::cout << "P1 intersects P2 ? " << gv::geometry::collides_GJK(P1.bbox(),P2) << std::endl;
	std::cout << "P2 intersects P1 ? " << gv::geometry::collides_GJK(P2,P1) << std::endl;
	bool flag = P2==P2;
	std::cout << "P2==P2 ? " << flag << std::endl;
}


void test_geometry(std::string filename, size_t N[3])
{
	using PARTICLE = gv::geometry::SuperEllipsoid<double>;

	std::cout << "load assembly\n";
	gv::geometry::Assembly<PARTICLE, double> assembly(filename, "-rrr-eps-xyz-q");
	
	std::cout << "save solid\n";
	assembly.save_solid("Geometry.vtk", N);
}


int main(int argc, char* argv[])
{	
	// double x=1;
	// double y=1;
	// double z=1;
	// if (argc>3)
	// {
	// 	x = atof(argv[1]);
	// 	y = atof(argv[2]);
	// 	z = atof(argv[3]);
	// }
	// test_particles(x,y,z);
	
	size_t N[3] {50, 50, 50};
	if (argc>4)
	{
		N[0] = atoi(argv[2]);
		N[1] = atoi(argv[3]);
		N[2] = atoi(argv[4]);
	}

	test_geometry(argv[1], N);

	return 0;
}