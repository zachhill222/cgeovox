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
	std::cout << "n_vertices= " << mesh.vertices.size() << std::endl;
	std::cout << "n_coarse_basis_functions= " << mesh.coarse_basis.size() << std::endl;
	std::cout << "n_active_coarse_basis_functions= " << mesh.coarse_basis_active2all.size() << std::endl;
	std::cout << "n_coarse_elements= " << mesh.coarse_elements.size() << std::endl;
	std::cout << "n_active_coarse_elements= " << mesh.coarse_elem_active2all.size() << std::endl;

	std::cout << "n_fine_basis_functions= " << mesh.fine_basis.size() << std::endl;
	
	std::cout << "n_fine_elements= " << mesh.fine_elements.size() << std::endl;
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
	Eigen::VectorXd ones(mesh.coarse_basis_active2all.size());
	Eigen::VectorXd x(mesh.coarse_basis_active2all.size());
	

	std::vector<double> one_vec, x_vec;

	one_vec.resize(mesh.coarse_basis.size());
	mesh._init_coarse_scalar_field(one_vec, fun_const);

	x_vec.resize(mesh.coarse_basis.size());
	mesh._init_coarse_scalar_field(x_vec, fun_x);

	for (size_t i=0; i<mesh.coarse_basis_active2all.size(); i++)
	{
		size_t b_idx = mesh.coarse_basis_active2all[i];
		ones[i]  = one_vec[b_idx];
		x[i] = x_vec[b_idx];
	}

	std::cout << "creating coarse mass matrix" << std::endl;
	matrix_constructor.make_mass_matrix(massMat,
		mesh.coarse_basis,
		mesh.coarse_basis_active2all,
		mesh.coarse_basis_all2active,
		mesh.coarse_elements,
		mesh.coarse_elem_active2all,
		mesh.coarse_elem_all2active);

	std::cout << "creating coarse stiffness matrix" << std::endl;
	matrix_constructor.make_stiff_matrix(stifMat,
		mesh.coarse_basis,
		mesh.coarse_basis_active2all,
		mesh.coarse_basis_all2active,
		mesh.coarse_elements,
		mesh.coarse_elem_active2all,
		mesh.coarse_elem_all2active);

	//compute actual volume of active elements at each depth
	std::vector<double> volume;
	for (size_t i=0; i<mesh.coarse_elem_active2all.size(); i++)
	{
		typename Mesh_t::Element_t ELEM = mesh.coarse_elements[mesh.coarse_elem_active2all[i]];
		typename Mesh_t::Point_t H = ELEM.H();
		if (ELEM.depth<volume.size()) {volume[ELEM.depth] += H[0]*H[1]*H[2];}
		else {volume.push_back(H[0]*H[1]*H[2]);}
	}

	std::cout << "volume of active elements: " << std::endl;
	double total = 0;
	for (size_t i=0; i<volume.size(); i++)
	{
		std::cout << "\tdepth " << i << " :\t" << volume[i] << std::endl;
		total += volume[i];
	}
	std::cout << "\t------------------------------\n";
	std::cout <<      "\ttotal   :\t" << total << std::endl;


	double mass_vol = ones.transpose()*(massMat*ones);
	std::cout << "1*M*1= " << mass_vol << " (error = " << total-mass_vol << ")" << std::endl;

	double stiff_vol = x.transpose()*(stifMat*x);
	std::cout << "x*A*x= " << stiff_vol << " (error = " << total-stiff_vol << ")" << std::endl;

	double zero_val = x.transpose()*(stifMat*ones);
	std::cout << "x*A*1= " << zero_val << " (error = " << zero_val << ")" << std::endl;
}




int main(int argc, char const *argv[])
{
	//get user arguments
	size_t N = 16; //number of elements per side in initial domain
	if (argc>1) {N=atoi(argv[1]);}

	size_t n = 0; //number of refinements
	if (argc>2) {n=atoi(argv[2]);}

	


	//set domain parameters
	std::string filename = "testdata/sphere.txt";
	gv::geometry::Assembly<gv::geometry::SuperEllipsoid,8> assembly(filename, "-rrr-eps-xyz-q");

	gv::geometry::AssemblyMeshOptions opts;
	opts.include_void = false;
	opts.include_interface = true;
	opts.include_solid = true;
	opts.check_centroid = false;
	opts.N = gv::util::Point<3,size_t> {N,N,N};
	opts.void_marker = 0;
	opts.solid_marker = 1;
	opts.interface_marker = 2;
	// opts.unknown_marker = -1;

	//initialize coarsest mesh
	gv::charms::AssemblyCharmsQ1Mesh mesh(assembly.bbox(), opts, assembly);
	std::cout << "initialized mesh" << std::endl;



	//print coarse mesh info
	mesh.get_active_indices();
	mesh.p.resize(mesh.coarse_basis_active2all.size());
	mesh._init_coarse_scalar_field(mesh.p, fun_const);
	print_mesh_info(mesh);
	// print_integrating_matrix_values(mesh);
	mesh.save_as("./outfiles/assembly_charms_mesh_refined_0.vtk");

	//refine mesh
	size_t inner_start=0;
	size_t next_inner_start=0;
	for (size_t i=1; i<=n; i++)
	{
		
		inner_start = next_inner_start;
		next_inner_start = mesh.coarse_elements.size();
		std::cout << "\n\nrefine depth: " << i << " (" << next_inner_start-inner_start << " elements)" << std::endl;

		for (size_t j=inner_start; j<next_inner_start; j++)
		{
			if (mesh.coarse_element_marker[j]==opts.interface_marker)
			{
				// std::cout << "\trefine element " << j << std::endl;
				mesh.refine(j);
			}
		}

		mesh.get_active_indices();
		// mesh._init_coarse_scalar_field(mesh.p, fun_const);
		print_mesh_info(mesh);
		double percent = (double) mesh.coarse_elem_active2all.size() / (double) std::pow( (double) N * std::pow(2,i), 3);
		std::cout << "ratio= " << percent << std::endl;
		// print_integrating_matrix_values(mesh);
		mesh.save_as("./outfiles/assembly_charms_mesh_refined_" + std::to_string(i) + ".vtk");
	}
	std::cout << "refined mesh" << std::endl;
	
	


	//check for duplicate vertices
	// std::cout << "DUPLICATE VERTICES: " << std::flush;
	// size_t n_duplicates = 0;
	// for (size_t i=0; i<mesh.nNodes(); i++)
	// {
	// 	for (size_t j=i+1; j<mesh.nNodes(); j++)
	// 	{
	// 		if (mesh.vertices[i]==mesh.vertices[j]) {n_duplicates+=1;}
	// 	}
	// }
	// std::cout << n_duplicates << std::endl;


	// std::cout << "\nMASS MATRIX:\n" << massMat << std::endl;
	// std::cout << "\nSTIFFNESS MATRIX:\n" << stifMat << std::endl;

	// std::cout << "\n===============================================" << std::endl;
	// std::cout << "========== PRINT ALL BASIS FUNCTIONS ==========" << std::endl;
	// std::cout << "===============================================\n" << std::endl;
	// for (size_t i=0; i<mesh.nBasis(); i++)
	// {
	// 	std::cout << "basis: " << i << std::endl;
	// 	std::cout << mesh.basis[i] << std::endl;
	// }

	// std::cout << "================================================" << std::endl;
	// std::cout << "============== PRINT ALL ELEMENTS ==============" << std::endl;
	// std::cout << "================================================\n" << std::endl;
	// for (size_t i=0; i<mesh.nElems(); i++)
	// {
	// 	std::cout << "element: " << i << std::endl;
	// 	std::cout << mesh.elements[i] << std::endl;
	// }

	//save octree structures for debugging
	// gv::util::view_octree_vtk(mesh.elements, "./outfiles/charms_element_octree.vtk");
	// gv::util::view_octree_vtk(mesh.vertices, "./outfiles/vertices_octree.vtk");
	// gv::util::view_octree_vtk(mesh.basis,    "./outfiles/basis_octree.vtk");
	gv::util::view_octree_vtk(assembly._particles, "./outfiles/assembly_octree.vtk");
	return 0;
}