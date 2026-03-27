#include "gutil.hpp"
#include <string>

#include "fem/charms_dofhandler.hpp"
#include "fem/charms_dofs.hpp"

#include "mesh/mesh_hierarchical.hpp"

#include "pde/helmholtz.hpp"

//define types
using T = gutil::FixedPoint<int64_t,0>;
// using T = double;
using Point_t   = gutil::Point<3,T>;
using Vertex_t  = gv::mesh::BasicVertex<Point_t>;
using Element_t = gv::mesh::HierarchicalColoredElement;
using Mesh_t    = gv::mesh::HierarchicalMesh<3,Element_t,Vertex_t>;

//import types from the mesh
using Box_t   = typename Mesh_t::DomainBox_t;
using Index_t = typename Mesh_t::Index_t;



int main(int argc, char* argv[])
{
	//build mesh
	std::cout << "Build Mesh\n";
	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N {1,1,1};
	Mesh_t mesh(domain, N, false);

	//build dofs
	std::cout << "Build Initial DOFs\n";
	gv::fem::CharmsDOFhandler<Mesh_t, gv::fem::CharmsVoxelQ1, double> dofhandler(mesh);
	dofhandler.distribute();

	//refine all elements element
	std::cout << "Refine DOFs\n";
	for (int k=0; k<2; k++) {
		dofhandler.mark_refine(2*domain);
		mesh.processSplit();
		dofhandler.process_refine<true>();
	}

	dofhandler.make_dof_map();

	std::cout << dofhandler << std::endl;
	std::cout << mesh << std::endl;

	//set up helmholtz problem
	gv::pde::HelmholtzSolver problem(mesh, dofhandler);

	std::cout << "Assemble Matrices\n";
	problem.assemble_mats();

	std::cout << "Compute Eigenvalues\n";
	problem.compute_eigenvals(2);

	std::cout << "Save Solutions\n";
	problem.save_to_vtk();

	return 0;
}