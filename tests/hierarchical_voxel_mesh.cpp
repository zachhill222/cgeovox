#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh.hpp"

using Point_t = gutil::Point<3,double>;
using Mesh_t  = gv::vmesh::HierarchicalVoxelMesh<6>;
using Elem_t  = Mesh_t::VoxelElement;
using Vert_t  = Mesh_t::VoxelVertex;

int main(int argc, char* argv[])
{
	//build test mesh
	Point_t low{0,0,0};
	Point_t high{1,1,1};
	Mesh_t  mesh(low, high);

	//refine a few elements
	mesh.set_depth(1);
	
	//check if the parent_child relation is correct
	bool correct = true;
	auto action_e = [&correct](Elem_t el) {
		//check if it is a valid element
		correct = correct and el.is_valid();

		//check parent logic
		for (int c=0; c<8; ++c) {
			Elem_t child = el.child(c);
			correct = correct and (child.parent() == el);
		}

		//check vertices are valid
		for (int vtx=0; vtx<8; ++vtx) {
			auto vertex = el.vertex(vtx);
			correct = correct and vertex.is_valid();
		}
	};
	auto action_v = [](Vert_t vtx) {
		std::cout << "\n== Start \n";
		std::cout << "Vertex: " << vtx.linear_index() << "\n" << vtx << "\n";
		std::cout << "coord: " << vtx.normalized_coordinate() << "\n";
		std::cout << "\nReduced: " << vtx.reduced_key().linear_index() << "\n" << vtx.reduced_key() << "\n";
		assert(vtx.is_same_coord(vtx.reduced_key()));
	};

	// mesh.for_each_vertex_depth(1, action_v);

	if (correct) {
		std::cout << "SUCCESS\n";
	}
	else {
		std::cout << "FAIL\n";
	}

	Elem_t el(1, 0, 0, 0);
	std::cerr << "EL: " << el.linear_index() << "\n";
	for (int v=0; v<8; ++v) {
	    auto vtx = el.vertex(v).reduced_key();
	    std::cerr << "v" << v << ": i=" << vtx.i() << " j=" << vtx.j() << " k=" << vtx.k() << " d=" << vtx.depth() << " idx=" << vtx.linear_index() << "\n";
	}

	// mesh.save_hierarchy("./outfiles/hierarchical_voxel_mesh_");
	mesh.save_unstructured_mesh("./outfiles/hierarchical_voxel_mesh_unstructured.vtk");
	return 0;
}