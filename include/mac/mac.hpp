#ifndef MAC_H
#define MAC_H

#include "constants.hpp"
#include "mesh/vtk_structured.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"
#include "util/box.hpp"
#include "geometry/voxel_particle_geometry.hpp"

#include "Eigen/Core"

#include <string>
#include <sstream>
#include <fstream>
#include <functional> //for passing method functions to solvers

#include <omp.h>

//Convenient typedefs
using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Assembly = GeoVox::geometry::Assembly;
using VectorXd = Eigen::VectorXd;
using VoxelParticleGeometry = GeoVox::geometry::VoxelParticleGeometry;

namespace GeoVox::mac{
	class MacMesh : public VoxelParticleGeometry {
	public:
		MacMesh(const Box& box, const long unsigned int N[3], const Assembly* const assembly) : VoxelParticleGeometry(box, N), mu(1.0), tol(MAC_DEFAULT_TOL), assembly(assembly) {
			// mark cells
			#pragma omp parallel for collapse(3)
			for (long unsigned int k=0; k<N[2]; k++){
				for (long unsigned int j=0; j<N[1]; j++){
					for (long unsigned int i=0; i<N[0]; i++){
						if (assembly->in_particle(idx2point(i,j,k))){
							markers[index(i,j,k)] = SOLID_PHASE_MARKER;
						}else{
							markers[index(i,j,k)] = DEFAULT_FLUID_PHASE_MARKER;
						}
					}
				}
			}
		}

		
		//Viscosity
		double mu;

		//Solver parameters
		int max_outer_iterations; //Maximum number of outer Distributive Gauss-Seidel (DGS) iterations
		double tol; //Outer iteration residual tolerance for each equation (u,v,w) momentum equations (p) mass conservation equation
		// double SCALE; //Scaling factor to multiply both sides. Should be O(h^2).

		//DOF checks
		//interior DOF -> return 0
		//boundary DOF -> return 1
		//not in domain DOF -> return -1 (i.e., in rock phase)
		int dof_type_p(const long unsigned int i, const long unsigned int j, const long unsigned int k) const;
		int dof_type_u(const long unsigned int i, const long unsigned int j, const long unsigned int k) const;
		int dof_type_v(const long unsigned int i, const long unsigned int j, const long unsigned int k) const;
		int dof_type_w(const long unsigned int i, const long unsigned int j, const long unsigned int k) const;


		//DOF values
		VectorXd p; //pressure values (reduced indexing): number of nonzeros in p_mask
		VectorXd u; //x-velocity values (reduced indexing): number of nonzeros in
		VectorXd v; //y-velocity values (reduced indexing)
		VectorXd w; //z-velocity values (reduced indexing)

		//RHS values
		VectorXd f1; //forcing term in x-direction at x-DOFs
		VectorXd f2; //forcing term in y-direction at y-DOFs
		VectorXd f3; //forcing term in z-direction at z-DOFsS
		VectorXd g;  //forcing term for p at p-DOFs
		
		//Geometry
		const Assembly* const assembly;

		//FIRST DERIVATIVES
		VectorXd dPdX(const VectorXd& variable) const;
		VectorXd dPdY(const VectorXd& variable) const;
		VectorXd dPdZ(const VectorXd& variable) const;

		VectorXd dUdX(const VectorXd& variable) const;
		VectorXd dVdY(const VectorXd& variable) const;
		VectorXd dWdZ(const VectorXd& variable) const;


		//SECOND DERIVATIVES
		VectorXd A(const VectorXd& variable) const; //same template for u,v,w
		VectorXd Ap(const VectorXd& variable) const;

		//SOLUTION
		void GS_relax_velocity();
		void GS_relax_velocity_reverse();
		VectorXd GS_relax_p() const; //Return ep = Ap_inv(g-Bx(u)-By(v)-Bz(w))
		VectorXd GS_relax_p_reverse() const;
		void setRockVelocity();

		void DGS();
		void DGS_reverse(); //for symmetric Gauss-Seidel iterations
		void solve(const int max_iter=MAC_DEFAULT_MAX_OUTER_ITERATIONS);
		void solve_reverse(const int max_iter=MAC_DEFAULT_MAX_OUTER_ITERATIONS);
		
		void solve_multigrid(int m);
		void coarsen(VectorXd& U, VectorXd& V, VectorXd& W, VectorXd& P);
		void refine(VectorXd& U, VectorXd& V, VectorXd& W, VectorXd& P);

		//save solution interpolated to pressure DOFs
		void saveas(const std::string filename) const;
	};
}


#endif