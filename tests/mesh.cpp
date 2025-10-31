#include "util/box.hpp"
#include "util/point.hpp"

#include "mesh/topological_mesh.hpp"

int main(int argc, char* argv[])
{	
	const int dim = 3;
	using T = double;

	using Point_t = gv::util::Point<dim,T>;
	using Box_t   = gv::util::Box<dim,T>;
	using Mesh_t  = gv::mesh::TopologicalMesh<T>;
	using Index_t = gv::util::Point<dim,size_t>;

	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Mesh_t mesh(domain, Index_t{1,2,3});

	for (int n=0; n<6; n++){
		const size_t nElems = mesh.nElems(false);
		for (size_t i=0; i<nElems; i+=2) {
			mesh.splitElement(i);
		}
	}
	
	//unrefine
	// mesh.joinDescendents(0);
	// mesh.joinDescendents(488);
	// mesh.recolor();
	mesh.save_as("./outfiles/topological_mesh.vtk", true, true);

	return 0;
}