#include "util/box.hpp"
#include "util/point.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/basic_mesh.hpp"
#include "mesh/topological_mesh.hpp"

int main(int argc, char* argv[])
{	
	const int dim = 3;
	using T = double;

	using Point_t = gv::util::Point<dim,T>;
	using Box_t   = gv::util::Box<dim,T>;
	using Index_t = gv::util::Point<dim,size_t>;

	using Node_t  = gv::mesh::BasicNode<Point_t>;
	using Element_t = gv::mesh::ColorableElement;
	constexpr gv::mesh::ColorMethod method = gv::mesh::ColorMethod::BALANCED;
	// using Mesh_t  = gv::mesh::TopologicalMesh<Node_t,Element_t,method>;
	using Mesh_t  = gv::mesh::BasicMesh<Node_t,Element_t>;

	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N{10, 10, 10};
	Mesh_t mesh(domain,N);

	// for (int n=0; n<1; n++){
	// 	const size_t nElems = mesh.nElems(false);
	// 	for (size_t i=0; i<nElems; i+=2) {
	// 		mesh.split_element(i);
	// 	}
	// 	mesh.process_refinement();
	// }


	// unrefine
	// mesh.join_descendents(0);
	// mesh.join_descendents(488);
	// mesh.recolor();

	// mesh.compute_boundary();
	// Mesh_t boundary = mesh.boundary_mesh();
	

	std::cout << mesh << std::endl;
	
	// std::cout << "colors are valid? " << mesh.colors_are_valid() << std::endl;

	// for (auto it=mesh.boundaryBegin(); it!=mesh.boundaryEnd(); ++it) {std::cout << *it << std::endl;}

	mesh.save_as("./outfiles/topological_mesh.vtk", true, true);
	// boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true);


	return 0;
}