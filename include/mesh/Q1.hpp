#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/homo_mesh.hpp"
#include "mesh/vtk_voxel.hpp"

#include <vector>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>

namespace gv::mesh
{
	//voxels embedded in 3D with piecewise d-linear shape functions.
	class VoxelMeshQ1 : public HomoMesh<gv::mesh::Voxel> {
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3>;
		using Index_t = gv::util::Point<3,size_t>;

		VoxelMeshQ1() : HomoMesh<gv::mesh::Voxel>() {}
		VoxelMeshQ1(const Box_t &bbox) : HomoMesh<gv::mesh::Voxel>(bbox) {}

		//mesh a box region uniformly
		VoxelMeshQ1(Point_t length, Index_t N) : HomoMesh<gv::mesh::Voxel>(Box_t(Point_t(0), length))
		{
			reserve_elems(N[0]*N[1]*N[2]);
			reserve_nodes((N[0]+1)*(N[1]+1)*(N[2]+1));

			gv::util::Point<3,double> H {length[0]/((double) N[0]), length[1]/((double) N[1]), length[2]/((double) N[2])};
			gv::util::Point<3,double> LOW;
			for (size_t i=0; i<N[0]; i++)
			{
				LOW[0] = H[0] * (double) i;
				for (size_t j=0; j<N[1]; j++)
				{
					LOW[1] = H[1] * (double) j;
					for (size_t k=0; k<N[2]; k++)
					{
						LOW[2] = H[2] * (double) k;
						
						gv::util::Box<3> box(LOW,LOW+H);

						gv::util::Point<3,double> voxel[8];
						for (size_t l=0; l<8; l++) {voxel[l] = box[l];}

						this->add_element(voxel);
					}
				}
			}
		}

		//construct exterior boundary
		void make_boundary(const int mkr=-1)
		{
			//set up new boundary
			size_t boundary_idx = this->_boundary.size();
			this->create_new_boundary();

			//construct boundary			
			if (mkr==-1) //get boundary of entire mesh
			{
				//loop through all nodes
				//in a Q1 voxel mesh, nodes are on the boundary if they belong to fewer than 8 elements

				for (size_t n_idx=0; n_idx<this->nNodes(); n_idx++)
				{
					if (this->local_nElem(n_idx) < 8)
					{
						this->_boundary[boundary_idx].push_back(n_idx);
					}
				}
			}
		}
	};
}