#include "util/point.hpp"
#include "fem/charms_util.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
	
	gv::fem::CharmsActiveElements<3> elements;
	elements.refine_at(elements.bbox().center());
	gv::util::Point<3,double> point(0.25);
	elements.refine_at(point);
	std::cout << elements;

	return 0;
}