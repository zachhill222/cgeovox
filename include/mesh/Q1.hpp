#pragma once

#include "util/point.hpp"

#include "mesh/mesh.hpp"
#include "mesh/vtkVoxel.hpp"

#include <vector>
#include <algorithm>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>
#include <armadillo>

namespace gv::mesh
{
	//voxels embedded in 3D with piecewise d-linear shape functions.
	class VoxelMeshQ1 : public Mesh<8> {
	private:
		//reference element to handle integration
		static const gv::mesh::Voxel referenceElement;

	public:
		VoxelMeshQ1() : Mesh<8>() {this->vtkID=11;}

		//create (isotropic) mass and stiffness matrices.
		void makeMassMatrix(arma::sp_mat &massMat) const;
		void makeStifMatrix(arma::sp_mat &stifMat) const;
		void makeMassStifMatrix(arma::sp_mat &massMat, arma::sp_mat &stifMat) const;
	};


	void VoxelMeshQ1::makeMassMatrix(arma::sp_mat &massMat) const
	{
		//set up index tracking to allow parallel looping over elements when computing integrals
		arma::umat(2,64*nElems()) locations;
		arma::vec(64*nElems()) values;

		//integrate over each element
		#pragma omp parallel
		for (size_t el=0; el<nElems(); el++)
		{
			//set parameters for this element
			gv::util::Point<3,double> H = _nodes[elem2node(el,7)] - _nodes[elem2node(el,0)]; //size of voxel
			size_t start = el*64; //start of this element's block in locations and values.

			//compute contributions to mass matrix
			for (size_t i=0; i<8; i++)
			{
				//global node number for local node i
				size_t global_i = elem2node(el,i);

				//diagonal (i,i) entry
				values.at(start + _ij2lin(i,i)) = referenceElement.integrate_mass(i,i,H);
				locations.at(0,start + _ij2lin(i,i)) = global_i;
				locations.at(1,start + _ij2lin(i,i)) = global_i;

				//off-diagonal entries
				for (size_t j=i+1; j<8; j++)
				{
					//global node number for local node j
					size_t global_j = elem2node(el,j);

					//get value
					double val = referenceElement.integrate_mass(i,j,H);
					
					//store location 1
					values[_ij2lin(i,j)] = val;
					locations.at(0,start + _ij2lin(i,j)) = global_i;
					locations.at(1,start + _ij2lin(i,j)) = global_j;

					//store location 2
					values[_ij2lin(j,i)] = val;
					locations.at(0,start + _ij2lin(j,i)) = global_j;
					locations.at(1,start + _ij2lin(j,i)) = global_i;
				}
			}
		}

		//construct matrix
		massMat = arma::sp_mat(true, locations, values, nNodes(), nNodes(), true, false);
	}
}