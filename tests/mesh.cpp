#include "util/box.hpp"
#include "util/point.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"

#include "mesh/mesh_view.hpp"


const int dim = 2;
using T = double;
	
template<int n>
using Box_t    = gv::util::Box<n,T>;



void test()
{	
	

	using Point_t  = gv::util::Point<dim,T>;

	
	using Index_t  = gv::util::Point<dim,size_t>;

	using Vertex_t  = gv::util::Point<3,T>;
	using Node_t    = gv::mesh::BasicNode<Vertex_t>;
	using Face_t    = gv::mesh::HierarchicalElement;
	using Element_t = gv::mesh::HierarchicalColoredElement;
	// using Element_t = gv::mesh::BasicElement;

	constexpr gv::mesh::ColorMethod method = gv::mesh::ColorMethod::BALANCED;
	// using Mesh_t  = gv::mesh::ColoredMesh<Node_t,Element_t,Face_t,method>;
	using Mesh_t  = gv::mesh::HierarchicalMesh<Node_t,Element_t,Face_t,method>;
	// using Mesh_t  = gv::mesh::BasicMesh<Node_t,Element_t,Face_t>;

	Point_t corner {1.0,1.0,1.0};
	Box_t<dim> domain(-corner, corner);
	Index_t N{1, 1, 1};
	Mesh_t mesh(domain,N,true);

	// mesh.reserveElements((size_t) 1 << 21); //128x128x128
	// mesh.reserveNodes((size_t) 1 << 22); //just over 129x129x129

	// gv::mesh::LogicalMesh logical_mesh(mesh);

	for (int n=0; n<3; n++){
		for (const auto &ELEM : mesh) {mesh.splitElement(ELEM.index);}
		mesh.processSplit();
	}


	auto fun = [](Vertex_t old) -> Vertex_t {
		if (old[2]<0) {return old;}

		double r = std::sqrt(old[0]*old[0] + old[2]*old[2]);
		double theta = std::atan2(old[2],old[0]);
		theta += 0.75*old[1];

		old[0] = r*std::cos(theta);
		old[2] = r*std::sin(theta); 
		return old;
	};
	for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
		// if (it->vertex[1]>0) {
			mesh.moveVertex(it->index, fun(it->vertex));
		// }
	}


	for (int n=0; n<3; n++){
		for (const auto &ELEM : mesh) {mesh.splitElement(ELEM.index);}
		mesh.processSplit();
	}


	// unrefine
	// mesh.joinDescendents(0);
	// mesh.join_descendents(488);
	// mesh.recolor();


	Box_t<3> bbox = mesh.bbox();
	Mesh_t boundary(bbox);
	mesh.getBoundaryMesh(boundary);
	
	// std::cout << "colors are valid? " << mesh.colors_are_valid() << std::endl;

	//loop though boundary elements
	// for (auto it=mesh.boundaryBegin(); it!=mesh.boundaryEnd(); ++it) {std::cout << *it << std::endl;}

	// std::cout << "NODES\n";
	// for (size_t i=0; i<mesh.nNodes(); i++) {
	// 	for (size_t j=i+1; j<mesh.nNodes(); j++) {
	// 		if (mesh.getNode(i).vertex == mesh.getNode(j).vertex) {
	// 			std::cout << "=============\n" << mesh.getNode(i) << "\n" << mesh.getNode(j) << "\n================";
	// 		}
	// 	}
	// }


	//loop though elements
	// std::cout << "ELEMENTS\n";
	// for (const auto &ELEM : mesh) {std::cout << ELEM << std::endl;}



	std::cout << mesh << std::endl;
	gv::mesh::memorySummary(mesh);



	std::cout << std::endl << boundary << std::endl;
	gv::mesh::memorySummary(boundary);

	mesh.save_as("./outfiles/topological_mesh.vtk", true, false);
	boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true, false);
}


int main(int argc, char* argv[])
{
	int nTests = 1;
	if (argc > 1) {nTests = atoi(argv[1]);}
	for (int i = 0; i < nTests; i++) {test();}

	return 0;
}