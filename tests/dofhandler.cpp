#include "gutil.hpp"

#include <format>

#include "fem/charms_dofhandler.hpp"
#include "fem/charms_dofs.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"
#include "mesh/mesh_view.hpp"

//define types
using T = gutil::FixedPoint<int64_t,0>;
using Mesh_t    = gv::mesh::HierarchicalMesh<3,3,T>;

//import types from the mesh
using Box_t   = typename Mesh_t::DomainBox_t;
using Point_t = typename Mesh_t::Point_t;
using Index_t = typename Mesh_t::Index_t;

int main(int argc, char* argv[])
{
	//build test mesh
	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N {8,8,8};
	Mesh_t mesh(domain, N, false);

	gv::fem::CharmsDOFhandler<Mesh_t, gv::fem::CharmsVoxelQ1, double> dofhandler(mesh);
	
	//assign coefficients
	for (size_t i=0; i<dofhandler.ndof(); ++i) {
		dofhandler.coef(i) = 1.0;
	}

	dofhandler.distribute();
	dofhandler.make_dof_map();
	
	//refine elements in the [0,1/2^k]^3 region
	double high=0.5;
	for (int k=0; k<6; k++) {
		Box_t box(Point_t{0,0,0}, Point_t{high,high,high});
		high*=0.5;
		dofhandler.mark_refine(box);
		mesh.processSplit();
		mesh.save_as(std::format("./outfiles/start_refine_{}.vtk", k), true);
		dofhandler.process_refine<true>();
	}

	dofhandler.make_dof_map();

	std::cout << dofhandler << std::endl;
	std::cout << mesh << std::endl;
	// gv::mesh::memorySummary(mesh);
	mesh.save_as("./outfiles/topological_mesh.vtk", true, false);

	return 0;
}