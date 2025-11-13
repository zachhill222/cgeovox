#include "util/box.hpp"
#include "util/point.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"

#include "mesh/mesh_view.hpp"

int main(int argc, char* argv[])
{	
	const int dim = 3;
	using T = double;

	using Point_t  = gv::util::Point<dim,T>;
	using Box_t    = gv::util::Box<dim,T>;
	using Index_t  = gv::util::Point<dim,size_t>;

	using Vertex_t  = gv::util::Point<3,T>;
	using Node_t    = gv::mesh::BasicNode<Vertex_t>;
	using Face_t    = gv::mesh::HierarchicalElement;
	using Element_t = gv::mesh::HierarchicalColoredElement;
	// using Element_t = gv::mesh::BasicElement;

	constexpr gv::mesh::ColorMethod method = gv::mesh::ColorMethod::BALANCED;
	using Mesh_t  = gv::mesh::HierarchicalMesh<Node_t,Element_t,Face_t,method>;
	// using Mesh_t  = gv::mesh::BasicMesh<Node_t,Element_t,Face_t>;

	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N{2, 2, 1};
	Mesh_t mesh(domain,N,false);


	// gv::mesh::LogicalMesh logical_mesh(mesh);


	for (int n=0; n<4; n++){
		const size_t nElems = mesh.nElems();
		for (size_t i=0; i<nElems; i+=1) {
			mesh.splitElement(i);
		}
		mesh.processSplit();
	}


	auto fun = [](Vertex_t old) -> Vertex_t {old[2] += 0.25*(1-old[0])*(1-old[1]); return old;};
	for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {mesh.moveVertex(it->index, 2*fun(it->vertex));}

	// unrefine
	mesh.joinDescendents(0);
	// mesh.join_descendents(488);
	// mesh.recolor();

	// mesh.compute_boundary();
	// Mesh_t boundary = mesh.boundary_mesh();
	
	// std::cout << "colors are valid? " << mesh.colors_are_valid() << std::endl;

	//loop though boundary elements
	// for (auto it=mesh.boundaryBegin(); it!=mesh.boundaryEnd(); ++it) {std::cout << *it << std::endl;}

	//loop though elements
	// for (const auto &ELEM : mesh) {std::cout << ELEM << std::endl;}

	std::cout << mesh << std::endl;
	gv::mesh::memorySummary(mesh);

	

	mesh.save_as("./outfiles/topological_mesh.vtk", true, true);
	// boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true);


	return 0;
}