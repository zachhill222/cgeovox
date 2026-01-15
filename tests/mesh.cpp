// #include "util/box.hpp"
// #include "util/point.hpp"
// #include "util/scalars/fixed_point.hpp"

#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"
#include "mesh/mesh_view.hpp"

// #include "util/octree_stats.hpp"

const int dim = 3;
// using T = double;
using T = gutil::FixedPoint<int64_t,-15>;

void test() {
	using Index_t    = gutil::Point<dim,size_t>;
	using RefPoint_t = gutil::Point<dim,T>;
	using Point_t    = gutil::Point<3,T>;
	using Box_t      = gutil::Box<dim,T>;
	using Element_t  = gv::mesh::HierarchicalColoredElement;
	
	constexpr gv::mesh::ColorMethod method = gv::mesh::ColorMethod::BALANCED;
	using Mesh_t  = gv::mesh::HierarchicalMesh<3,3,T,Element_t,method>;
	using BoundaryMesh_t = gv::mesh::HierarchicalMesh<3,2,T,Element_t,method>;
	// using Mesh_t  = gv::mesh::BasicMesh<Node_t,Element_t,Face_t>;

	RefPoint_t corner {0.1,0.125,0.1235};
	Box_t   domain(-corner, corner);
	Index_t N{1, 1, 1};
	Mesh_t mesh(domain,N,false);

	// gv::mesh::LogicalMesh logical_mesh(mesh);

	for (int n=0; n<5; n++){
		for (const auto &ELEM : mesh) {mesh.splitElement(ELEM.index);}
		mesh.processSplit();
	}

	auto fun = [corner](Point_t old) -> Point_t {
		double r = std::sqrt(old[0]*old[0] + old[1]*old[1]);
		double theta = std::atan2(old[1],old[0]);
		theta += 2.0*0.78539816339*old[2]/corner[2];

		old[0] = r*std::cos(theta);
		old[1] = r*std::sin(theta); 
		return old;
	};
	for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
		if (it->coord[2]>0.0) {
			mesh.moveVertex(it->index, fun(it->coord));
		}
	}


	for (int n=0; n<1; n++){
		for (const auto &ELEM : mesh) {mesh.splitElement(ELEM.index);}
		mesh.processSplit();
	}

	// unrefine
	// mesh.joinDescendents(1);


	//print mesh summary
	std::cout << "\n\n";
	std::cout << mesh << std::endl;
	gv::mesh::memorySummary(mesh);


	Box_t bbox = mesh.bbox();
	BoundaryMesh_t boundary(bbox);
	mesh.getBoundaryMesh(boundary);
	std::cout << "\n\n";
	std::cout << std::endl << boundary << std::endl;
	gv::mesh::memorySummary(boundary);

	mesh.save_as("./outfiles/topological_mesh.vtk", true, false);
	boundary.save_as("./outfiles/topological_mesh_boundary.vtk", true, true);
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