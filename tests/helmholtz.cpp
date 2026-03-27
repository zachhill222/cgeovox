#include "gutil.hpp"
#include <string>
#include <chrono>

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

//timing types
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;


int main(int argc, char* argv[])
{
	//build mesh
	std::cout << "Build Mesh: " << std::flush;
	auto t0 = Clock::now();
	Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
	Index_t N {1,1,1};
	Mesh_t mesh(domain, N, false);
	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";

	//build dofs
	std::cout << "Build Initial DOFs: " << std::flush;
	t0 = Clock::now();
	gv::fem::CharmsDOFhandler<Mesh_t, gv::fem::CharmsVoxelQ1, double> dofhandler(mesh);
	dofhandler.distribute();
	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";

	//refine all elements element
	std::cout << "Refine DOFs: " << std::flush;
	t0 = Clock::now();
	for (int k=0; k<5; k++) {
		dofhandler.mark_refine(2*domain);
		mesh.processSplit();
		dofhandler.process_refine<true>();
	}

	dofhandler.make_dof_map();

	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";
	std::cout << dofhandler << std::endl;
	std::cout << mesh << std::endl;

	//set up helmholtz problem
	gv::pde::HelmholtzSolver problem(mesh, dofhandler);

	std::cout << "Assemble Matrices: " << std::flush;
	t0 = Clock::now();
	problem.assemble_mats();
	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";

	std::cout << "Compute Eigenvalues: " << std::flush;
	t0 = Clock::now();
	problem.compute_eigenvals(6);
	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";

	std::cout << "Save Solutions: " << std::flush;
	t0 = Clock::now();
	problem.save_to_vtk("./outfiles/");
	std::cout << Seconds(Clock::now()-t0).count() << " seconds\n";

	return 0;
}