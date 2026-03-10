#pragma once

#include "fem/dofs.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "gutil.hpp"

#include <type_traits>
#include <vector>
#include <set>
#include <array>
#include <string>
#include <cassert>

#include <Eigen/SparseCore>
#include <omp.h>

namespace gv::fem
{

	//DOF map from [0:N-1] into [0:M-1] where M is the number of DOFs tracked by the handler
	//The [0:N-1] index corresponds to the DOF number for the FEM matrices
	struct DOFMap {
		std::vector<size_t> compressed2global;
		std::vector<size_t> global2compressed;
	};




	//DOF handler for a single DOF type
	template<
		gv::mesh::BasicMeshType Mesh_t,
		typename DOF_t,
		typename Coef_t = double,
		size_t MAX_CHILDREN = 8 //maximum number of children that a basis function can have
		>
	class CharmsDOFhandler
	{
	private:
		void distribute_nodal();

	protected:
		const Mesh_t& mesh;
		std::vector<DOF_t>  dofs;
		std::vector<std::array<size_t, MAX_CHILDREN>> dof_children;
		std::vector<size_t> dof_parents;
		std::vector<std::set<size_t>> element_basis_s; //track basis functions per-element on the same level of refinement
		std::vector<std::set<size_t>> element_basis_a; //track basis functions per-element on coarser levels of refinement

		std::vector<Coef_t> coefs; //best tracked in a problem class and only write here for fileio
		std::vector<size_t> boundary_dofs;

		std::vector<size_t> refine_dof_list;
		
		DOFMap dof_map;

	public:
		inline size_t n_dofs() const {return dofs.size();}
		inline const DOF_t& dof(const size_t idx) const {assert(idx<dofs.size()); return dofs[idx];}
		inline const Coef_t& coef(const size_t idx) const {assert(idx<coefs.size()); return coefs[idx];}
		inline Coef_t& coef(const size_t idx) {assert(idx<coefs.size()); return coefs[idx];}
		
		void reserve(const size_t length) {
			dofs.reserve(length);
			coefs.reserve(length);
			dof_children.reserve(length);
			dof_parents.reserve(length);
		}

		void resize(const size_t length) {
			dofs.resize(length);
			coefs.resize(length);
			dof_children.resize(length);
			dof_parents.resize(length);
		}

		void clear() {
			dofs.clear(); 
			coefs.clear(); 
			boundary_dofs.clear();
			dof_children.clear();
			dof_parents.clear();
			refine_dof_list.clear();
		}

		//construct handler and link to the mesh
		CharmsDOFhandler(const Mesh_t& mesh) : mesh(mesh) {}

		//create the dofs for a 
		void distribute();

		//interpolate the coefs to evaluate the field at the given point
		// Coef_t interpolate(const )

		//save the mesh and nodal evaluations to a file
		void save_as(const std::string filename, const bool use_ascii=false) const;

		//active/deactivate basis functions
		void activate(const size_t idx) {
			assert(idx<this->size());
			if (dofs[idx].active) {return;}
			dofs[idx].active = true;

			//loop through support elements
			for (int i=0; i<DOF_t::max_support; ++i) {
				const size_t el_idx = dofs[idx].support_idx[i];
				if (el_idx != (size_t) -1) {
					//ensure the support elements have this basis function
					element_basis_s[el_idx].insert(idx);

					//if the support element is not active, activate it and all ancestor elements
					const auto& ELEM = mesh.getElement(el_idx);
					assert(ELEM.active);

					//TODO: implement what to do when we activate an element.
					//This needs to be done in the mesh, which we only have a const reference to.

					//add this basis function as an ancestor basis function to any descendent elements
					const std::vector<size_t> descendents;
					mesh.getElementDescendents_Unlocked(el_idx, descendents, false);
					for (size_t descendent : descendents) {
						element_basis_a[descendent].insert(idx);
					}
				}
			}
		}

		void deactivate(const size_t idx) {
			assert(idx<this->size());
			if (!dofs[idx].active) {return;}
			dofs[idx].active = false;

			//loop through support elements
			for (int i=0; i<DOF_t::max_support; ++i) {
				const size_t el_idx = dofs[idx].support_idx[i];

				//TODO: implement de-activation of an element if it has no active basis functions
			}

		}

		//refine basis functions
		void mark_refine(const size_t idx) {
			assert(idx<this->size());

			//check if this function is refinable
			if (!dofs[idx].active) {return;}

			const size_t nElem = mesh.nElements(false); //need to work with all elements, not just the active ones

			//mark support elements to be split
			for (int i=0; i<DOF_t::max_support; ++i) {
				const size_t el_idx = dofs[idx].support_idx[i];
				if (el_idx < nElem) {
					//only marks element to be split. The elements must be split outside of this class.
					mesh.splitElement(el_idx);
				}
			}

			//add this basis function to the list to be refined
			refine_dof_list.push_back(idx);
		}


		//create compressed/global DOF map
		DOFMap make_dof_map() {
			DOFMap map;
			map.global2compressed.reserve(dofs.size());
			map.compressed2global.reserve(mesh.nVertices()); //estimate

			size_t compressed_idx = 0;
			for (size_t i=0; i<dofs.size(); ++i) {
				if (dofs[i].active) {
					map.global2compressed.push_back(compressed_idx);
					map.compressed2global.push_back(i);
					compressed_idx++;
				}
				else {
					map.global2compressed.push_back( (size_t) -1);
				}
			}

			return map;
		}

		//create CSR matrix with the correct sparsity structure
		//outer_index[i] and outer_index[i+1] bookend the indices of inner_index that correpsond to row i
		//inner_index[outer_index[i]] through inner_index[outer_index[i+1]-1] store the columns with non-zero entries in the matrix
		//later the values array needs to be constructed so that matrix(i,j) = values[inner_index]
		void get_csr_structure(std::vector<size_t>& outer_index, std::vector<size_t>& inner_index) {
			
		}
	};


	///dispatch the distribute to the correct type
	template<gv::mesh::BasicMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>::distribute()
	{
		clear();

		if constexpr (std::is_same_v<DOF_t,VoxelQ1>) {distribute_nodal();}
		else if constexpr (std::is_same_v<DOF_t,HexQ1>) {distribute_nodal();}
		else {throw std::logic_error("this DOF_t is not supported yet");}
	}


	///distribute lagrange nodal dofs (it is assumed that the mesh is in a conformal state)
	template<gv::mesh::BasicMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>::distribute_nodal()
	{
		auto nvert = mesh.nVertices();
		this->resize(nvert);

		#pragma omp parallel for
		for (size_t n=0; n<mesh.nVertices(); ++n) {
			const auto& VERTEX = mesh.getVertex(n);
			std::array<size_t,DOF_t::max_support> support;
			std::array<size_t,DOF_t::max_support> local_idx;

			int i;
			for (i=0; i<VERTEX.elems.size(); ++i) {
				support[i] = VERTEX.elems[i];
				const auto& ELEM = mesh.getElement(support[i]);

				//ensure that the mesh and DOF are compatible
				if constexpr (std::is_same_v<DOF_t,VoxelQ1>) {
					if (ELEM.vtkID != VOXEL_VTK_ID) {
						throw std::runtime_error("attempting to distribute VoxelQ1 DOF to a non-voxel element");
					}
				}
				else if constexpr (std::is_same_v<DOF_t,HexQ1>) {
					if (ELEM.vtkID != HEXAHEDRON_VTK_ID) {
						throw std::runtime_error("attempting to distribute HexQ1 DOF to a non-hex element");
					}
				}

				//get local index within the support element
				for (size_t m=0; m<ELEM.vertices.size(); ++m) {
					if (ELEM.vertices[m]==n) {
						local_idx[i] = m;
						break;
					}
				}
				assert(ELEM.vertices[local_idx[i]] == n);
			}

			//mark unused support elements if the node was on the boundary
			for (;i<DOF_t::max_support; ++i) {
				support[i]   = (size_t) -1;
				local_idx[i] = (size_t) -1;
			}

			//create the DOF
			dofs[n] = DOF_t(n, support, local_idx);

			//populate parent/child relationships with null values
			dof_parents[n] = (size_t) -1;
			for (i=0; i<DOF_t::max_support; ++i) {dof_children[n][i] = (size_t) -1;}

			//mark DOF as part of the boundary
			if (VERTEX.boundary_faces.size()>0) {
				#pragma omp critical
				{
					boundary_dofs.push_back(n);
				}
			}
		}
	}
}


