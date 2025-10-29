#include "util/box.hpp"
#include "util/point.hpp"

#include "mesh/topological_mesh.hpp"

int main(int argc, char* argv[])
{	
	using Point_t = gv::util::Point<3,double>;
	using Box_t   = gv::util::Box<3,double>;
	using Mesh_t  = gv::mesh::TopologicalMesh<3,double>;
	using Index_t = gv::util::Point<3,size_t>;

	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Mesh_t mesh(domain, Index_t{10,20,30});

	mesh.recolor(true);
	mesh.save_as("./outfiles/topological_mesh.vtk", true, true);

	return 0;
}