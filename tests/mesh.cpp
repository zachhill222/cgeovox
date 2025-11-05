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
	Mesh_t mesh(domain, Index_t{10,20,30});

	for (int n=0; n<4; n++){
		const size_t nElems = mesh.nElems(false);
		for (size_t i=0; i<nElems; i+=2) {
			mesh.split_element(i);
		}
	}
	
	// unrefine
	// mesh.join_descendents(0);
	// mesh.join_descendents(488);
	// mesh.recolor();

	// mesh.compute_boundary();
	// Mesh_t boundary = mesh.boundary_mesh();
	std::cout << mesh << std::endl;
	
	mesh.save_as("./outfiles/topological_mesh.vtk", true);
	// boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true);


	return 0;
}