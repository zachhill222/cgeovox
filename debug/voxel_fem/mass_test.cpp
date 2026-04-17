#include "voxel_mesh/fem/dofhandler.hpp"
#include "voxel_mesh/fem/dofs/voxel_dof_Q1.hpp"
#include "voxel_mesh/fem/kernel.hpp"
#include "voxel_mesh/mesh/voxel_mesh.hpp"
#include "voxel_mesh/pde/forms/L2_inner.hpp"
#include "voxel_mesh/pde/forms/H1_semi.hpp"
#include "util/log_time.hpp"

using Mesh_t    = GV::HierarchicalVoxelMesh<10>;
using Elem_t    = Mesh_t::VoxelElement;
using Vert_t    = Mesh_t::VoxelVertex;
using DofKey_t  = GV::VoxelVertexKey<11,0,0>;
using DOF_t     = GV::VoxelQ1<DofKey_t>;
using Handler_t = GV::DofHandler<Mesh_t,DOF_t>;

using BiMass_t  = GV::SymmetricL2<DOF_t>;
using BiStiff_t = GV::SymmetricH1<DOF_t>;
using Kernel_t  = GV::Kernel<4,BiMass_t,BiStiff_t>;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	//uniform depth
	const int depth = 6;

	//define mesh
	Mesh_t mesh({0,0,0}, {1,2,3});
	mesh.set_depth(depth);

	//define dofhandler
	Handler_t dofhandler(mesh);
	dofhandler.set_depth(depth);
	dofhandler.save_dof_list();

	//define kernel (includes bilinear form)
	const auto diag = mesh.high - mesh.low;
	BiMass_t mass_bl;
	BiStiff_t stiff_bl;
	Kernel_t kernel(diag[0], diag[1], diag[2], mass_bl, stiff_bl);

	//integrate bilinear forms
	auto integrate = [&kernel, &dofhandler](Elem_t el) {
		const auto el_basis = dofhandler.basis_active(el);
		kernel.set_element(el);
		kernel.form<0>().set_basis(el_basis,el_basis);
		kernel.form<1>().set_basis(el_basis,el_basis);
		kernel.compute_scatter<0>();
		kernel.compute_scatter<1>();
	};

	{
		GV::LogTime time{"build COO_CSR"};
		mesh.for_each_depth<Elem_t>(depth,integrate);
	}
	
	Eigen::SparseMatrix<double,Eigen::RowMajor,int> mass_mat, stiff_mat;

	#ifdef _OPENMP
	omp_set_max_active_levels(2);
	omp_set_nested(1);
	#pragma omp parallel
	#pragma omp single
	#endif
	{
		#ifdef _OPENMP
		#pragma omp task
		#endif
		{
			mass_mat = mass_bl.to_eigen_csr(dofhandler.last_compressed_dofs(),dofhandler.last_compressed_dofs());
		}

		#ifdef _OPENMP
		#pragma omp task
		#endif
		{
			stiff_mat = stiff_bl.to_eigen_csr(dofhandler.last_compressed_dofs(),dofhandler.last_compressed_dofs());
		}
	}

	Eigen::VectorXd vec = Eigen::VectorXd::Ones(mass_mat.rows());
	std::cout << "mass: " << (mass_mat * vec).transpose() * vec << std::endl;


	//populate the vec with the x coordinates of each dof to test the stiffness matrix
	for (size_t i=0; i<dofhandler.last_compressed_dofs().size(); ++i) {
		double x = dofhandler.last_compressed_dofs()[i].key.x();
		vec[i] = (1.0-x)*mesh.low[0] + x*mesh.high[0];
	}

	std::cout << "stiff: " << (stiff_mat * vec).transpose() * vec << std::endl;

}