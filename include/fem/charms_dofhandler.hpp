#pragma once

#include "fem/dofs.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "gutil.hpp"

#include <type_traits>
#include <vector>
#include <utility>
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
		size_t ndof() const {return compressed2global.size();}
		void clear() {compressed2global.clear(); global2compressed.clear();}
	};




	//DOF handler for a single DOF type
	template<
		gv::mesh::HierarchicalColorableMeshType Mesh_t,
		typename DOF_t,
		typename Coef_t = double,
		size_t MAX_CHILDREN = 8 //maximum number of children that a basis function can have
		>
	class CharmsDOFhandler
	{
	private:
		void distribute_nodal();

	protected:
		std::vector<DOF_t>  dofs;
		std::vector<std::array<size_t, MAX_CHILDREN>> dof_children;
		std::vector<size_t> dof_parents;
		std::vector<std::set<size_t>> element_basis_s; //track basis functions per-element on the same level of refinement
		std::vector<std::set<size_t>> element_basis_a; //track basis functions per-element on coarser levels of refinement

		std::vector<Coef_t> coefs; //best tracked in a problem class and only write here for fileio
		std::vector<size_t> boundary_dofs;

		std::set<size_t> refine_dof_list;
		
		DOFMap dof_map;

	public:
		const Mesh_t& mesh; //just a const ref. can be public.

		inline size_t ndof() const {return dofs.size();}
		inline const DOF_t& dof(const size_t idx) const {assert(idx<dofs.size()); return dofs[idx];}
		inline const Coef_t& coef(const size_t idx) const {assert(idx<coefs.size()); return coefs[idx];}
		inline Coef_t& coef(const size_t idx) {assert(idx<coefs.size()); return coefs[idx];}
		inline const DOFMap& get_dof_map() const {return dof_map;}

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
			assert(idx<this->ndof());
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
			assert(idx<this->ndof());
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
			assert(idx<this->ndof());

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
			refine_dof_list.insert(idx);
		}

		//mark all active basis functions whose support intersects the box for refinement
		void mark_refine(const typename Mesh_t::DomainBox_t& box) {
			#pragma omp parallel for
			for (size_t d_idx=0; d_idx<dofs.size(); ++d_idx) {
				const auto& DOF = dofs[d_idx];
				if (!DOF.active) {continue;}

				//loop through support elements
				for (size_t el_idx : DOF.support_idx) {
					if (el_idx == (size_t) -1) {continue;}
					bool added = false;

					const auto& ELEM = mesh.getElement(el_idx);
					for (size_t v_idx : ELEM.vertices) {
						const auto& VERTEX = mesh.getVertex(v_idx);
						if (box.contains(VERTEX.coord)) {
							#pragma omp critical
							{
								mark_refine(d_idx);
							}
							added=true;
							break;
						}
					}

					if (added) {
						#pragma omp critical
						{
							std::cout << "marked element " << el_idx << " for splitting\n";
						}
						break;
					}
				}
			}
		}


		//process the refinement of the marked dofs
		//mesh.processSplit() must be called between mark_refine() and process_refine()
		void process_refine<bool HIERARCHICAL=true>() {
			this->reserve(dofs.size()+MAX_CHILDREN*refine_dof_list.size());

			//add all new DOFs
			for (size_t d_idx : refine_dof_list) {
				DOF_t& PARENT = dofs[d_idx];

				//create children, add to list of dofs, populate relations
				std::vector<DOF_t> children = PARENT.make_children(mesh);
				size_t start = dofs.size();
				size_t end   = start + children.size();
				
				//move children dofs
				dofs.insert(
					dofs.end(),
					std::make_move_iterator(children.begin()),
					std::make_move_iterator(children.end())
				);

				//populate relations and activate children
				size_t i=0;
				for (size_t new_dof_idx=start; new_dof_idx<end; ++new_dof_idx) {
					/
					assert(dof_children[d_idx][i] == (size_t) -1);
					dof_children[d_idx][i++] = new_dof_idx;
					dof_parents[new_dof_idx] = d_idx;




				}

				//activate children
				for (size_t)
				if constexpr (HIERARCHICAL) {

				}
			}
		}


		//create compressed/global DOF map
		void make_dof_map() {
			dof_map.clear();
			dof_map.global2compressed.reserve(dofs.size());
			dof_map.compressed2global.reserve(mesh.nVertices()); //estimate

			size_t compressed_idx = 0;
			for (size_t i=0; i<dofs.size(); ++i) {
				if (dofs[i].active) {
					dof_map.global2compressed.push_back(compressed_idx);
					dof_map.compressed2global.push_back(i);
					compressed_idx++;
				}
				else {
					dof_map.global2compressed.push_back( (size_t) -1);
				}
			}
		}

		//create CSR/CSC matrix with the correct sparsity structure
		//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
		//RowMajor format is better for settng boundary conditions, but ColMajor might be better for Eigen routines
		template<int Format=Eigen::RowMajor, typename T=double>
		Eigen::SparseMatrix<T,Format> init_matrix() const;

	};


	///dispatch the distribute to the correct type
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>::distribute()
	{
		clear();

		if constexpr (std::is_same_v<DOF_t,VoxelQ1>) {distribute_nodal();}
		else if constexpr (std::is_same_v<DOF_t,HexQ1>) {distribute_nodal();}
		else {throw std::logic_error("this DOF_t is not supported yet");}
	}


	///distribute lagrange nodal dofs (it is assumed that the mesh is in a conformal state)
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>::distribute_nodal()
	{
		auto nvert = mesh.nVertices();
		this->resize(nvert);

		#pragma omp parallel for
		for (size_t n=0; n<mesh.nVertices(); ++n) {
			const auto& VERTEX = mesh.getVertex(n);
			std::array<size_t,DOF_t::max_support> support;
			std::array<size_t,DOF_t::max_support> local_idx;

			size_t i;
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
			dofs[n].active = true;

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

	//create CSR/CSC matrix with the correct sparsity structure
	//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	template<int Format, typename T>
	Eigen::SparseMatrix<T,Format> CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>::init_matrix() const {
		using Triplet = Eigen::Triplet<T>;
		std::vector<std::vector<Triplet>> color_coo_idx;

		size_t ncolors = mesh.nColors();
		color_coo_idx.resize(ncolors);

		#pragma omp parallel for
		for (size_t c=0; c<ncolors; ++c) {
			auto& coo_idx = color_coo_idx[c];
			coo_idx.reserve(64*mesh.colorCount(c)); //approximate. in CHARMS, it is difficult to know which basis functions interact
			for (size_t e_idx=0; e_idx<mesh.nElements(true); ++e_idx) {
				const auto& ELEM = mesh.getElement(e_idx);
				if (ELEM.color != c) {continue;}

				//basis_s-basis_s and basis_s-basis_a interactions
				for (size_t global_dof_1 : element_basis_s[ELEM.index]) {
					if (!dofs[global_dof_1].active) {continue;}

					size_t compressed_dof_1 = dof_map.global2compressed[global_dof_1];
					coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_1, T{0}));

					//basis_s-basis_s
					for (size_t global_dof_2 : element_basis_s[ELEM.index]) {
						if (global_dof_2 <= global_dof_1) {continue;}
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}

					//basis_s-basis_a
					for (size_t global_dof_2 : element_basis_a[ELEM.index]) {
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}
				}

				//basis_a-basis_a interactions
				for (size_t global_dof_1 : element_basis_a[ELEM.index]) {
					if (!dofs[global_dof_1].active) {continue;}

					size_t compressed_dof_1 = dof_map.global2compressed[global_dof_1];
					coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_1, T{0}));

					for (size_t global_dof_2 : element_basis_a[ELEM.index]) {
						if (global_dof_2 <= global_dof_1) {continue;}
						if (!dofs[global_dof_2].active) {continue;}

						size_t compressed_dof_2 = dof_map.global2compressed[global_dof_2];
						coo_idx.push_back(Triplet(compressed_dof_1, compressed_dof_2, T{0}));
						coo_idx.push_back(Triplet(compressed_dof_2, compressed_dof_1, T{0}));
					}
				}
			}
		}

		//join coo_idx from each color
		size_t n_coo_idx = 0;
		for (size_t c=0; c<mesh.nColors(); ++c) {n_coo_idx += color_coo_idx[c].size();}
		std::vector<Triplet> all_coo_idx;
		all_coo_idx.reserve(n_coo_idx);
		for (size_t c=0; c<mesh.nColors(); ++c) {
			all_coo_idx.insert(
				all_coo_idx.end(), 
				std::make_move_iterator(color_coo_idx[c].begin()), 
				std::make_move_iterator(color_coo_idx[c].end())
			);
		}

		//create matrix
		Eigen::SparseMatrix<T,Format> mat;
		mat.resize(dof_map.ndof(), dof_map.ndof());
		mat.setFromTriplets(all_coo_idx.begin(), all_coo_idx.end());
		return mat;
	}

	//print dof info
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, typename DOF_t, typename Coef_t, size_t MAX_CHILDREN>
	std::ostream& operator<<(std::ostream& os, const CharmsDOFhandler<Mesh_t,DOF_t,Coef_t,MAX_CHILDREN>& dofhandler) {
		os  << "\nCHARMS dofhandler:\n"
			<< "\tndofs (total)  : " << dofhandler.ndof() << "\n"
			<< "\tndofs (active) : " << dofhandler.get_dof_map().ndof() << "\n"
			<< "\tMesh at " << &dofhandler.mesh << " has " << dofhandler.mesh.nElements(true)
			<< " elements and " << dofhandler.mesh.nVertices() << " vertices\n";
		return os;
	}

}


