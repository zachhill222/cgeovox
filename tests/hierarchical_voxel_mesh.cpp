#include "gutil.hpp"
#include "voxel_mesh/voxel_mesh.hpp"

using Point_t = gutil::Point<3,double>;
using Mesh_t = gv::vmesh::HierarchicalVoxelMesh;
using Elem_t = gv::vmesh::VoxelElementKey;

int main(int argc, char* argv[])
{
	//build test mesh
	Point_t low{-1,-1,-1};
	Point_t high{1,1,1};
	Mesh_t  mesh(6, low, high);

	//refine a few elements

	//check if the parent_child relation is correct
	bool correct = true;
	auto action = [&correct](Elem_t el) {
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
	mesh.for_each_element(action);

	if (correct) {
		std::cout << "SUCCESS\n";
	}
	else {
		std::cout << "FAIL\n";
	}

	auto pred = [](Elem_t el) {return el.is_active();};
	mesh.build_unstructured_mesh();
	
	// mesh.save_hierarchy("./outfiles/hierarchical_voxel_mesh_");
	mesh.save_unstructured_mesh("./outfiles/hierarchical_voxel_mesh_unstructured.vtk");
	return 0;
}