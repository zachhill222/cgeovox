#include "gutil.hpp"
#include <string>

#include "fem/charms_dofhandler.hpp"
#include "fem/charms_dofs.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"
#include "mesh/mesh_view.hpp"

//define types
// using T = gutil::FixedPoint<int64_t,0>;
using T = double;
using Point_t   = gutil::Point<3,T>;
using Vertex_t  = gv::mesh::BasicVertex<Point_t>;
using Element_t = gv::mesh::HierarchicalColoredElement;
using Mesh_t    = gv::mesh::HierarchicalMesh<3,Element_t,Vertex_t>;

//import types from the mesh
using Box_t   = typename Mesh_t::DomainBox_t;
using Index_t = typename Mesh_t::Index_t;

int main(int argc, char* argv[])
{
	//build test mesh
	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N {8,8,64};
	Mesh_t mesh(domain, N, false);

	gv::fem::CharmsDOFhandler<Mesh_t, gv::fem::CharmsVoxelQ1, double> dofhandler(mesh);
	
	dofhandler.distribute();
	dofhandler.make_dof_map();
	
	//assign coefficients
	for (size_t i=0; i<dofhandler.ndof(); ++i) {
		dofhandler.coef(i) = 1.0;
	}

	//refine elements in the [0,1/2^k]^3 region
	double high=0.5;
	for (int k=0; k<4; k++) {
		std::cout << "Refine: " << k << std::endl;
		Box_t box(Point_t{0,0,0}, Point_t{high,high,high});
		high*=0.5;
		dofhandler.mark_refine(box);
		mesh.processSplit();
		dofhandler.process_refine<false>();
		// dofhandler.save_as(std::string("./outfiles/start_refine_")+std::to_string(k)+std::string(".vtk"), 1, true);
	}

	dofhandler.make_dof_map();

	std::cout << dofhandler << std::endl;
	std::cout << mesh << std::endl;
	// gv::mesh::memorySummary(mesh);
	dofhandler.save_as("./outfiles/charms_mesh.vtk", 1, true);

	return 0;
}