#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"

const int dim = 3;
using T = double;
// using T = gutil::FixedPoint<int64_t,-15>;

void test() {
	using Index_t    = gutil::Point<dim,size_t>;
	using RefPoint_t = gutil::Point<dim,T>;
	using Point_t    = gutil::Point<3,T>;
	using Box_t      = gutil::Box<dim,T>;
	using Element_t  = gv::mesh::HierarchicalColoredElement<11>;
	using Vertex_t   = gv::mesh::BasicVertex<Point_t>;

	constexpr gv::mesh::ColorMethod method = gv::mesh::ColorMethod::BALANCED;
	using Mesh_t  = gv::mesh::HierarchicalMesh<Element_t,T,method>;
	

	Point_t corner {1.0,1.0,1.0};
	Box_t   domain(-corner, corner);
	Index_t N{4,4,4};
	Mesh_t  mesh(domain,N);

	auto H = corner;
	for (int n=0; n<8; n++){
		H*=0.75;
		mesh.refineRegion(Box_t{corner-H,corner+H});
		mesh.processSplit();
	}

	// unrefine
	// mesh.joinDescendents(1);


	//print mesh summary
	std::cout << "\n\n";
	std::cout << mesh << std::endl;
	gv::mesh::memorySummary(mesh);


	// Box_t bbox = mesh.bbox();
	// BoundaryMesh_t boundary(bbox);
	// mesh.getBoundaryMesh(boundary);
	// std::cout << "\n\n";
	// std::cout << std::endl << boundary << std::endl;
	// gv::mesh::memorySummary(boundary);

	mesh.save_as("./outfiles/topological_mesh.vtk", true, false);
	// boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true, true);
	// gv::util::makeOctreeLeafMesh(mesh.getNodeOctree(), "./outfiles/topological_mesh_node_octree.vtk");
}


int main(int argc, char* argv[])
{
	#ifdef _OPENMP
		std::cout << "PARALLEL (OPENMP)" << std::endl;
	#else
		std::cout << "SERIAL" << std::endl;
	#endif


	int nTests = 1;
	if (argc > 1) {nTests = atoi(argv[1]);}
	for (int i = 0; i < nTests; i++) {test();}

	return 0;
}