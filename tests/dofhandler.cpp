#include "gutil.hpp"

#include "fem/charms_dofhandler.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"
#include "mesh/mesh_hierarchical.hpp"
#include "mesh/mesh_view.hpp"

//define types
using T = gutil::FixedPoint<int64_t,0>;
using Element_t = gv::mesh::BasicElement;
using Mesh_t  = gv::mesh::BasicMesh<3,3,T,Element_t>;

//import types from the mesh
using Box_t   = typename Mesh_t::DomainBox_t;
using Point_t = typename Mesh_t::Point_t;
using Index_t = typename Mesh_t::Index_t;

int main(int argc, char* argv[])
{
	//build test mesh
	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N {16,16,16};
	Mesh_t mesh(domain, N, false);

	gv::fem::CharmsDOFhandler<Mesh_t, gv::fem::VoxelQ1, double> dofhandler(mesh);
	dofhandler.distribute();

	//assign coefficients
	for (size_t i=0; i<dofhandler.n_dofs(); ++i) {
		dofhandler.coef(i) = 1.0;
	}

	dofhandler.make_dof_map();
	std::cout << mesh << std::endl;
	return 0;
}