#pragma once

#include "mesh/homo_mesh.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/SparseCore>
#include <omp.h>

namespace gv::fem
{
	///convert (i,j) index pair to a linear index
	template <size_t nodes_per_element>
	inline size_t _ij2lin(const size_t i, const size_t j) {return nodes_per_element*i + j;}

	
	///construct mass integrating matrix
	template <typename Mesh_t, int Format_t>
	void make_mass_matrix(const Mesh_t &mesh, Eigen::SparseMatrix<double, Format_t> &massMat)
	{
		//ensure only homogeneous meshes are passed
		// static_assert(std::is_base_of<gv::mesh::HomoMesh, Mesh_t>::value, "Mesh_t is not derived from HomoMesh.");
		if (!mesh.isHomogeneous) {throw std::runtime_error("Mesh_t must be derived from gv::mesh::HomoMesh.");}
		const size_t nodes_per_element = mesh.referenceElement.nNodes;

		//set up index tracking to allow parallel looping over elements when computing integrals
		typedef Eigen::Triplet<double> T;
		std::vector<T> triplets;
		triplets.resize(nodes_per_element*nodes_per_element*mesh.nElems()); //may be slow. initializes Eigen::Triplets with default constructor.

		//integrate over each element
		#pragma omp parallel
		for (size_t el=0; el<mesh.nElems(); el++)
		{
			//construct logical element
			auto local_element = mesh.new_element();
			for (size_t n_idx=0; n_idx<nodes_per_element; n_idx++)
			{
				local_element.nodes[n_idx] = &(mesh.nodes(mesh.elem2node(el,n_idx)));
			}

			//set parameters for this element
			size_t start = el*nodes_per_element*nodes_per_element; //start of this element's block in locations and values.

			//compute contributions to mass matrix
			for (size_t i=0; i<nodes_per_element; i++)
			{
				//global node number for local node i
				size_t global_i = mesh.elem2node(el,i);

				//diagonal (i,i) entry
				double val = local_element.integrate_mass(i,i);

				//record location index and value
				triplets[start + _ij2lin<nodes_per_element>(i,i)] = T(global_i, global_i, val);

				//off-diagonal entries
				for (size_t j=i+1; j<nodes_per_element; j++)
				{
					//global node number for local node j
					size_t global_j = mesh.elem2node(el,j);

					//get value
					val = local_element.integrate_mass(i,j);

					//record location index and values
					triplets[start + _ij2lin<nodes_per_element>(i,j)] = T(global_i, global_j, val);
					triplets[start + _ij2lin<nodes_per_element>(j,i)] = T(global_j, global_i, val);
				}
			}
		}

		//construct matrix
		massMat.setZero();
		massMat.resize(mesh.nNodes(),mesh.nNodes());
		massMat.setFromTriplets(triplets.begin(), triplets.end());
	}


	///construct stiffness integrating matrix
	template <typename Mesh_t, int Format_t>
	void make_stiffness_matrix(const Mesh_t &mesh, Eigen::SparseMatrix<double, Format_t> &stiffMat)
	{
		//ensure only homogeneous meshes are passed
		// static_assert(std::is_base_of<gv::mesh::HomoMesh<mesh.Element_type>, Mesh_t>::value, "Mesh_t is not derived from HomoMesh.");
		if (!mesh.isHomogeneous) {throw std::runtime_error("Mesh_t must be derived from gv::mesh::HomoMesh.");}
		const size_t nodes_per_element = mesh.referenceElement.nNodes;

		//set up index tracking to allow parallel looping over elements when computing integrals
		typedef Eigen::Triplet<double> T;
		std::vector<T> triplets;
		triplets.resize(nodes_per_element*nodes_per_element*mesh.nElems()); //may be slow. initializes Eigen::Triplets with default constructor.

		//integrate over each element
		#pragma omp parallel
		for (size_t el=0; el<mesh.nElems(); el++)
		{
			//construct logical element
			auto local_element = mesh.new_element();
			for (size_t n_idx=0; n_idx<nodes_per_element; n_idx++)
			{
				local_element.nodes[n_idx] = &(mesh.nodes(mesh.elem2node(el,n_idx)));
			}

			//set parameters for this element
			size_t start = el*nodes_per_element*nodes_per_element; //start of this element's block in locations and values.

			//compute contributions to mass matrix
			for (size_t i=0; i<nodes_per_element; i++)
			{
				//global node number for local node i
				size_t global_i = mesh.elem2node(el,i);

				//diagonal (i,i) entry
				double val = local_element.integrate_stiff(i,i);

				//store location index and value
				triplets[start + _ij2lin<nodes_per_element>(i,i)] = T(global_i, global_i, val);

				//off-diagonal entries
				for (size_t j=i+1; j<nodes_per_element; j++)
				{
					//global node number for local node j
					size_t global_j = mesh.elem2node(el,j);

					//get value
					val = local_element.integrate_stiff(i,j);

					//record location index and values
					triplets[start + _ij2lin<nodes_per_element>(i,j)] = T(global_i, global_j, val);
					triplets[start + _ij2lin<nodes_per_element>(j,i)] = T(global_j, global_i, val);
				}
			}
		}

		//construct matrix
		stiffMat.setZero();
		stiffMat.resize(mesh.nNodes(),mesh.nNodes());
		stiffMat.setFromTriplets(triplets.begin(), triplets.end());
	}


	///construct mass and stiffness integrating matrices
	template <typename Mesh_t, int Format1_t, int Format2_t>
	void make_integrating_matrices(const Mesh_t &mesh, Eigen::SparseMatrix<double,Format1_t> &massMat, Eigen::SparseMatrix<double,Format2_t> &stiffMat)
	{
		//ensure only homogeneous meshes are passed
		if (!mesh.isHomogeneous) {throw std::runtime_error("Mesh_t must be derived from gv::mesh::HomoMesh.");}
		const size_t nodes_per_element = mesh.referenceElement.nNodes;

		//set up index tracking to allow parallel looping over elements when computing integrals
		typedef Eigen::Triplet<double> T;
		std::vector<T> triplets_m, triplets_s;
		triplets_m.resize(nodes_per_element*nodes_per_element*mesh.nElems()); //may be slow. initializes Eigen::Triplets with default constructor.
		triplets_s.resize(nodes_per_element*nodes_per_element*mesh.nElems()); //may be slow. initializes Eigen::Triplets with default constructor.

		//integrate over each element
		#pragma omp parallel
		for (size_t el=0; el<mesh.nElems(); el++)
		{
			//construct logical element
			auto local_element = mesh.new_element();
			for (size_t n_idx=0; n_idx<nodes_per_element; n_idx++)
			{
				local_element.nodes[n_idx] = &(mesh.nodes(mesh.elem2node(el,n_idx)));
			}

			//set parameters for this element
			size_t start = el*nodes_per_element*nodes_per_element; //start of this element's block in locations and values.

			//compute contributions to mass matrix
			for (size_t i=0; i<nodes_per_element; i++)
			{
				//global node number for local node i
				size_t global_i = mesh.elem2node(el,i);

				//diagonal (i,i) entry
				double val_m = local_element.integrate_mass(i,i);
				double val_s = local_element.integrate_stiff(i,i);

				//record location index and value
				triplets_m[start + _ij2lin<nodes_per_element>(i,i)] = T(global_i, global_i, val_m);
				triplets_s[start + _ij2lin<nodes_per_element>(i,i)] = T(global_i, global_i, val_s);

				//off-diagonal entries
				for (size_t j=i+1; j<nodes_per_element; j++)
				{
					//global node number for local node j
					size_t global_j = mesh.elem2node(el,j);

					//get value
					val_m = local_element.integrate_mass(i,j);
					val_s = local_element.integrate_stiff(i,j);

					//record location index and values
					triplets_m[start + _ij2lin<nodes_per_element>(i,j)] = T(global_i, global_j, val_m);
					triplets_m[start + _ij2lin<nodes_per_element>(j,i)] = T(global_j, global_i, val_m);

					triplets_s[start + _ij2lin<nodes_per_element>(i,j)] = T(global_i, global_j, val_s);
					triplets_s[start + _ij2lin<nodes_per_element>(j,i)] = T(global_j, global_i, val_s);
				}
			}
		}

		//construct mass matrix
		massMat.setZero();
		massMat.resize(mesh.nNodes(),mesh.nNodes());
		massMat.setFromTriplets(triplets_m.begin(), triplets_m.end());

		//construct stiffness matrix
		stiffMat.setZero();
		stiffMat.resize(mesh.nNodes(),mesh.nNodes());
		stiffMat.setFromTriplets(triplets_s.begin(), triplets_s.end());
	}
}

