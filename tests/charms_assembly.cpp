#include "charms/assembly_charmsQ1.hpp"
#include "util/octree_util.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"
#include "charms/charms_fem_util.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <string>
#include <vector>
#include <functional>

//uncomment to disable assert()
// #define NDEBUG

template <class Mesh_t>
void print_mesh_info(const Mesh_t &mesh)
{
	std::cout << mesh << std::endl;
}

double fun_const(const gv::util::Point<3,double> point) {return 1.0;}
double fun_x(const gv::util::Point<3,double> point) {return point[0];}

template <class Mesh_t>
void print_integrating_matrix_values(Mesh_t &mesh)
{
	//create mass and stiffness matrices
	mesh.get_active_indices();
	gv::charms::CharmsGalerkinMatrixConstructor<Mesh_t> matrix_constructor;


	Eigen::SparseMatrix<double> massMat, stifMat;
	Eigen::VectorXd ones(mesh.basis_active2all.size());
	Eigen::VectorXd x(mesh.basis_active2all.size());
	

	std::vector<double> one_vec, x_vec;

	one_vec.resize(mesh.basis.size());
	mesh._init_scalar_field(one_vec, fun_const);

	x_vec.resize(mesh.basis.size());
	mesh._init_scalar_field(x_vec, fun_x);

	for (size_t i=0; i<mesh.basis_active2all.size(); i++)
	{
		size_t b_idx = mesh.basis_active2all[i];
		ones[i]  = one_vec[b_idx];
		x[i] = x_vec[b_idx];
	}

	std::cout << "creating integrating matrices" << std::endl;
	matrix_constructor.make_integrating_matrices(massMat, stifMat,
		mesh.basis,
		mesh.basis_active2all,
		mesh.basis_all2active,
		mesh.elements,
		mesh.elem_active2all,
		mesh.elem_all2active);

	double total = mesh.volume();
	std::cout << "volume of active elements: " << total << std::endl;


	double mass_vol = ones.transpose()*(massMat*ones);
	std::cout << "1*M*1= " << mass_vol << " (error = " << std::fabs(total-mass_vol) << ")" << std::endl;

	double stiff_vol = x.transpose()*(stifMat*x);
	std::cout << "x*A*x= " << stiff_vol << " (error = " << std::fabs(total-stiff_vol) << ")" << std::endl;

	double zero_val = x.transpose()*(stifMat*ones);
	std::cout << "x*A*1= " << zero_val << " (error = " << std::fabs(zero_val) << ")" << std::endl;
}




int main(int argc, char const *argv[])
{
	//get user arguments
	size_t N = 16; //number of elements per side in initial domain
	if (argc>1) {N=atoi(argv[1]);}

	size_t n = 0; //number of refinements
	if (argc>2) {n=atoi(argv[2]);}

	


	//set domain parameters
	std::string filename = "testdata/particles_50.txt";
	using Particle_t = gv::geometry::Ellipsoid;
	gv::geometry::Assembly<Particle_t,8> assembly(filename, "-rrr-eps-xyz-q", 1000);

	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = true;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N = gv::util::Point<3,size_t> {N,N,N};
	opts.void_marker = 0;
	opts.solid_marker = 1;
	opts.interface_marker = 2;

	//initialize coarsest mesh
	gv::charms::AssemblyCharmsQ1Mesh mesh(assembly.bbox(), opts, assembly);

	//define signed distance function to the first particle
	// const Particle_t particle = assembly._particles[0];
	// std::function<double(gv::util::Point<3,double>)> sgndist = [particle](const gv::util::Point<3,double>& point)->double {return particle.signed_distance(point);};


	//print coarse mesh info
	mesh.get_active_indices();
	std::vector<double> u;
	u.resize(mesh.basis.size());
	mesh._init_scalar_field(u, fun_x);
	std::cout << "coarse mesh" << std::endl;
	print_mesh_info(mesh);
	// print_integrating_matrix_values(mesh);
	mesh.save_as("./outfiles/assembly_charms_mesh_refined_0.vtk", u);

	//refine mesh
	size_t inner_start=0;
	size_t next_inner_start=0;
	for (size_t i=1; i<=n; i++)
	{
		
		inner_start = next_inner_start;
		next_inner_start = mesh.elements.size();
		std::cout << "\n\nrefine depth: " << i << " (" << next_inner_start-inner_start << " elements)" << std::endl;

		for (size_t j=inner_start; j<next_inner_start; j++)
		{
			if (mesh.element_marker[j]==opts.interface_marker)
			{
				// std::cout << "\trefine element " << j << std::endl;
				mesh.refine(j,u);
			}
		}

		mesh.get_active_indices();
		// mesh._init_scalar_field(mesh.p, fun_const);
		print_mesh_info(mesh);
		// print_integrating_matrix_values(mesh);
		mesh.save_as("./outfiles/assembly_charms_mesh_refined_" + std::to_string(i) + ".vtk", u);
	}

	//save octree structures for debugging
	// gv::util::view_octree_vtk(mesh.elements, "./outfiles/charms_element_octree.vtk");
	// gv::util::view_octree_vtk(mesh.vertices, "./outfiles/vertices_octree.vtk");
	// gv::util::view_octree_vtk(mesh.basis,    "./outfiles/basis_octree.vtk");
	gv::util::view_octree_vtk(assembly._particles, "./outfiles/assembly_octree.vtk");
	return 0;
}