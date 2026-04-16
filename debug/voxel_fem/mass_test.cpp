#include "voxel_mesh/fem/dofhandler.hpp"
#include "voxel_mesh/fem/dofs/voxel_dof_Q1.hpp"
#include "voxel_mesh/fem/kernel.hpp"
#include "voxel_mesh/mesh/voxel_mesh.hpp"
#include "voxel_mesh/pde/bilinear_mass.hpp"


using Mesh_t    = gv::vmesh::HierarchicalVoxelMesh<10>;
using Elem_t    = Mesh_t::VoxelElement;
using Vert_t    = Mesh_t::VoxelVertex;
using DofKey_t  = gv::vmesh::VoxelVertexKey<11,0,0>;
using DOF_t     = gv::vmesh::VoxelQ1<DofKey_t>;
using Handler_t = gv::vmesh::DofHandler<Mesh_t,DOF_t>;

using BiMass_t  = gv::vmesh::SymmetricMassForm<DOF_t>;
using Kernel_t  = gv::vmesh::Kernel<4,BiMass_t>;

int main(int argc, char* argv[]) {
	//uniform depth
	const int depth = 6;

	//define mesh
	Mesh_t mesh({0,0,0}, {1,1,1});
	mesh.set_depth(depth);

	//define dofhandler
	Handler_t dofhandler(mesh);
	dofhandler.set_depth(depth);
	dofhandler.snapshot_dof_list();

	//define kernel (includes bilinear form)
	const auto diag = mesh.high - mesh.low;
	Kernel_t kernel(diag[0], diag[1], diag[2]);

	//integrate mass matrix
	auto make_mass = [&kernel, &dofhandler](Elem_t el) {
		const auto el_basis = dofhandler.basis_active(el);
		kernel.set_element(el);
		kernel.form<0>().set_basis(el_basis,el_basis);
		kernel.compute_scatter<0>();
	};

	mesh.for_each_depth<Elem_t>(depth,make_mass);
	auto mass_mat = kernel.form<0>().to_eigen_csr(dofhandler.last_compressed_dofs(),dofhandler.last_compressed_dofs());


	auto ones = Eigen::VectorXd::Ones(mass_mat.rows());
	std::cout << (mass_mat * ones).transpose() * ones << std::endl;

}