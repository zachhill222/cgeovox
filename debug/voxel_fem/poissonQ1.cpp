
#include "voxel_mesh/pde/problems/poisson.hpp"

using Poisson = GV::PoissonQ1<0,10>;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	Poisson problem({0,0,0}, {1,1,1});
	problem.set_depth(4);
	problem.integrate();
	problem.build_matrices();

	auto bc_fun1  = [](typename Poisson::DOF_t dof) {return dof.key.x();};
	auto bc_pred1 = [](typename Poisson::DOF_t dof) {return dof.key.y()==1.0;};

	auto bc_fun2  = [](typename Poisson::DOF_t dof) {return dof.key.y();};
	auto bc_pred2 = [](typename Poisson::DOF_t dof) {return dof.key.x()==1.0;};

	problem.apply_dirichlet(bc_fun1, bc_pred1);
	problem.apply_dirichlet(bc_fun2, bc_pred2);
	problem.solve();
	problem.save_as("poissonQ1.vtk");
}