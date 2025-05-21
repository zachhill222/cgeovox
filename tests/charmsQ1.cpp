#include "util/point.hpp"
#include "fem/charms_util.hpp"
#include <iostream>
#include <vector>

int main(int argc, char const *argv[])
{
	
	gv::fem::CharmsMesh<3> elements;
	elements.refine_at(elements.bbox().center());
	gv::util::Point<3,double> point(0.25);
	elements.refine_at(point);
	std::cout << elements;

	std::cout << "\n=============================\n";
	for (size_t k=0; k<elements.size(); k++)
	{
		std::cout << "\nidx= " << k << "\n" << elements[k] << std::endl;
	}


	return 0;
}