#pragma once

#include "fem/charms_dofs.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"
#include "mesh/vtk_elements.hpp"

#include "gutil.hpp"

#include <type_traits>
#include <vector>
#include <utility>
#include <unordered_set>
#include <array>
#include <string>
#include <cassert>

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
		IsCharmsDOF DOF_t,
		typename Coef_t = double>
	class CharmsDOFhandler
	{
	private:
		static constexpr size_t MAX_CHILDREN = static_cast<size_t>(DOF_t::nchildren);
		void distribute_nodal();

	protected:
		std::vector<DOF_t>  dofs;

		std::vector<std::unordered_set<size_t>> element_basis_s; //track basis functions per-element on the same level of refinement
		std::vector<std::unordered_set<size_t>> element_basis_a; //track basis functions per-element on coarser levels of refinement

		//map the index of the mesh feature to basis functions.
		//note that in CHARMS, more than one dof can live at a mesh feature (esp. nodal basis functions)
		std::vector<std::unordered_set<size_t>> mesh2dof; 

		std::vector<Coef_t> coefs; //best tracked in a problem class and only write here for fileio
		std::vector<size_t> boundary_dofs;

		std::unordered_set<size_t> refine_dof_list;
		
		DOFMap dof_map;

	public:
		const Mesh_t& mesh; //just a const ref. can be public.

		inline size_t ndof() const {return dofs.size();}
		inline const DOF_t& dof(const size_t idx) const {assert(idx<dofs.size()); return dofs[idx];}
		inline const Coef_t& coef(const size_t idx) const {assert(idx<coefs.size()); return coefs[idx];}
		inline Coef_t& coef(const size_t idx) {assert(idx<coefs.size()); return coefs[idx];}
		
		inline const DOFMap& get_dof_map() const {return dof_map;}
		inline const auto& get_dofs() const {return dofs;}
		inline const auto& get_element_basis_s() const {return element_basis_s;}
		inline const auto& get_element_basis_a() const {return element_basis_a;}


		void reserve(const size_t length) {
			dofs.reserve(length);
			coefs.reserve(length);
		}

		void resize(const size_t length) {
			dofs.resize(length);
			coefs.resize(length);
		}

		void clear() {
			dofs.clear(); 
			coefs.clear(); 
			boundary_dofs.clear();
			refine_dof_list.clear();
		}

		//construct handler and link to the mesh
		CharmsDOFhandler(const Mesh_t& mesh) : mesh(mesh) {}

		//create the dofs for a 
		void distribute();

		//interpolate the coefs to evaluate the field at the given point
		Coef_t eval_at_vertex(const size_t vertex_index) const;

		//save the mesh and nodal evaluations to a file
		void save_as(const std::string filename, const int coef_dim=1, const bool use_ascii=false) const;

		//get the children of a basis function
		std::vector<size_t> get_children(const size_t p_idx) const {
			assert(p_idx < dofs.size());
			std::vector<size_t> result;

			//a basis function C will be a child of the specified function P only if
			//the support of C is entirely contained in the support of P

			//aggregate the child elements that must be used for the support elements of child dofs
			std::unordered_set<size_t> child_elems;
			for (const size_t e_idx : dofs[p_idx].support_idx) {
				const auto& ELEM = mesh.getElement(e_idx);
				child_elems.insert(ELEM.children.begin(), ELEM.children.end());
			}

			//check the basis_s functions on the child elements
			for (const size_t c_elem_idx : child_elems) {
				for (const size_t c_idx : element_basis_s[c_elem_idx]) {
					bool is_child = true;
					const DOF_t& CHILD = dofs[c_idx];
					assert(CHILD.depth == dofs[p_idx].depth+1);
					for (const size_t e_idx : CHILD.support_idx) {
						if (!child_elems.contains(e_idx)) {
							is_child = false;
							break;
						}
					}

					if (is_child) {
						result.push_back(c_idx);
					}
				}
			}

			//ensure the list is of unique values
			std::sort(result.begin(), result.end());
			auto last = std::unique(result.begin(), result.end());
			result.erase(last, result.end());
			return result;
		}


		//active/deactivate basis functions
		void activate(const size_t idx) {
			assert(idx<this->ndof());
			assert(element_basis_a.size() == mesh.nElements(false));
			assert(element_basis_s.size() == mesh.nElements(false));

			if (dofs[idx].active) {return;}

			//a dof must have at least one active support element
			for (const size_t e_idx : dofs[idx].support_idx) {
				if (mesh.getElement(e_idx).is_active) {
					dofs[idx].active = true;
					break;
				}
			}

			//verify that the dof is included in the appropriate basis_a and basis_s sets
			#ifndef NDEBUG
			for (int i=0; i<DOF_t::max_support; ++i) {
				const size_t el_idx = dofs[idx].support_idx[i];
				if (el_idx != (size_t) -1) {
					//element activation is handled outside the dofhandler
					//populate basis_s and basis_a either way (in case the element is later activated)

					//ensure the correct elements have this in their basis_s and basis_a
					assert(element_basis_s[el_idx].contains(idx));

					std::vector<size_t> descendents;
					mesh.getElementDescendents_Unlocked(el_idx, descendents, false);
					for (size_t descendent : descendents) {
						assert(element_basis_a[descendent].contains(idx));
					}
				}
			}
			#endif
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

		bool is_refinable(const size_t idx) const {
			assert(idx<this->ndof());
			
			if (!dofs[idx].active) {return false;}

			//TODO: check if its parents are refined
			return true;
		}

		//refine basis functions
		void mark_refine(const size_t idx) {
			assert(idx<this->ndof());

			//check if this function is refinable
			if (!is_refinable(idx)) {return;}

			//mark support elements to be split
			for (size_t el_idx : dofs[idx].support_idx) {
				if (el_idx != (size_t) -1) {
					#ifndef NDEBUG
						const auto& ELEM = mesh.getElement(el_idx);
				        assert(ELEM.depth == dofs[idx].depth);
			        #endif

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
				if (!is_refinable(d_idx)) {continue;}

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
				}
			}
		}


		//process the refinement of the marked dofs
		//mesh.processSplit() must be called between mark_refine() and process_refine()
		template<bool HIERARCHICAL=true>
		void process_refine() {
			//ensure basis_s and basis_a have the same length as the total number of elements
			element_basis_s.resize(mesh.nElements(false));
			element_basis_a.resize(mesh.nElements(false));

			if constexpr (DOF_t::feature_dim == 0) {
				mesh2dof.resize(mesh.nVertices());
			}
			else if constexpr (DOF_t::feature_dim == 1) {
				throw std::runtime_error("CharmsDOFhandler: edge dofs not implemented");
			}
			else if constexpr (DOF_t::feature_dim == 2 and Mesh_t::SPACE_DIM!=2) {
				throw std::runtime_error("CharmsDOFhandler: face dofs not implemented");
			}
			else if constexpr (DOF_t::feature_dim == Mesh_t::SPACE_DIM) {
				mesh2dof.resize(mesh.nElements(false));
			}
			else {
				throw std::runtime_error("CharmsDOFhandler: unkown DOF type");
			}

			//create any new DOFs
			this->reserve(dofs.size()+MAX_CHILDREN*refine_dof_list.size());
			for (size_t parent_idx : refine_dof_list) {
				const DOF_t& PARENT = dofs[parent_idx];

				//update basis_a for the newly created elements
				for (size_t e_idx : PARENT.support_idx) {
					if (e_idx == (size_t) -1) {continue;}
					const auto& ELEM = mesh.getElement(e_idx);
					assert(ELEM.children.size() > 0);

					for (size_t c_elem_idx : ELEM.children) {
						element_basis_a[c_elem_idx].insert(element_basis_s[e_idx].begin(), element_basis_s[e_idx].end());
						element_basis_a[c_elem_idx].insert(element_basis_a[e_idx].begin(), element_basis_a[e_idx].end());
					}
				}


				//create children, add to list of dofs, populate relations
				auto children = PARENT.make_children(mesh);
				
				//determine which children are new (different parents can refine to create the same children)
				std::array<size_t, MAX_CHILDREN> child_dof_idx;
				child_dof_idx.fill((size_t) -1);

				//add any new dofs to the list and record the index of all child dofs
				for (size_t c_idx=0; c_idx<MAX_CHILDREN; ++c_idx) {
					//if this child is not a new dof, then each support element will have this
					//child as a dof in basis_s
					const DOF_t& CHILD = children[c_idx];

					//check if the child exists (this should only ever fail on the boundary)
					if (CHILD.global_idx == (size_t) -1) {continue;}
					
					//check if the child previously existed as a DOF
					size_t existing_dof_idx = (size_t) -1;
					for (size_t sib_idx : mesh2dof[CHILD.global_idx]) {
						if (CHILD == dofs[sib_idx]) {
							existing_dof_idx = sib_idx;
							break;
						}
					}

					//get the index of the child DOF and add to list if needed
					//note the contents of children[] is invalid
					if (existing_dof_idx == (size_t) -1) {
						//add new dof
						const size_t new_dof_idx = dofs.size();
						child_dof_idx[c_idx] = new_dof_idx;
						dofs.push_back(children[c_idx]);
						
						//update dof references
						const DOF_t& CHILD = dofs[new_dof_idx];
						mesh2dof[CHILD.global_idx].insert(new_dof_idx);

						for (size_t c_elem_idx : CHILD.support_idx) {
							if (c_elem_idx == (size_t) -1) {continue;}
							element_basis_s[c_elem_idx].insert(new_dof_idx);

							//if the support elements have been previously refined
							//then this new dof needs to be an ancestor basis function on
							//the child elements.
							const auto& ELEM = mesh.getElement(c_elem_idx);
							std::vector<size_t> desendent_elements;
							mesh.getElementDescendents_Unlocked(c_elem_idx, desendent_elements, false);
							for (size_t e : desendent_elements) {
								element_basis_a[e].insert(new_dof_idx);
							}
						}

						coefs.emplace_back(0);
					} else {
						child_dof_idx[c_idx] = existing_dof_idx;
						for (size_t c_elem_idx : CHILD.support_idx) {
							if (c_elem_idx == (size_t) -1) {continue;}
							element_basis_s[c_elem_idx].insert(existing_dof_idx);
						}
					}
				}

				//activate functions and update coefficients
				if constexpr (HIERARCHICAL) {
					for (size_t c_idx=0; c_idx<MAX_CHILDREN; ++c_idx) {
						const size_t child_idx = child_dof_idx[c_idx];
						if (child_idx == (size_t) -1) {continue;}

						const DOF_t& CHILD = dofs[child_idx];
						if (CHILD.global_idx != PARENT.global_idx) {
							activate(child_idx);
						}
						
						coefs[child_idx] = Coef_t{0};
					}
				}
				else {
					deactivate(parent_idx);
					for (size_t c_idx=0; c_idx<children.size(); ++c_idx) {
						const size_t child_idx = child_dof_idx[c_idx];
						if (child_idx == (size_t) -1) {continue;}

						activate(child_idx);

						//represent the parent function as a linear combination of the children functions
						coefs[child_idx] += coefs[parent_idx]*PARENT.get_child_coef(c_idx);
					}
				}
			}

			refine_dof_list.clear();
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
	};


	///dispatch the distribute to the correct type
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::distribute()
	{
		clear();

		if constexpr (DOF_t::feature_dim == 0) {distribute_nodal();}
		else {throw std::logic_error("this DOF_t is not supported yet");}
	}


	///distribute lagrange nodal dofs (it is assumed that the mesh is in a conformal state)
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::distribute_nodal()
	{
		auto nvert = mesh.nVertices();
		this->resize(nvert);
		element_basis_s.resize(mesh.nElements(false));
		element_basis_a.resize(mesh.nElements(false));
		mesh2dof.resize(mesh.nVertices());

		for (size_t n=0; n<mesh.nVertices(); ++n) {
			const auto& VERTEX = mesh.getVertex(n);
			std::array<size_t,DOF_t::max_support> support;
			std::array<size_t,DOF_t::max_support> local_idx;

			size_t i;
			for (i=0; i<VERTEX.elems.size(); ++i) {
				support[i] = VERTEX.elems[i];
				const auto& ELEM = mesh.getElement(support[i]);

				//get local index within the support element
				for (size_t m=0; m<ELEM.vertices.size(); ++m) {
					if (ELEM.vertices[m]==n) {
						local_idx[i] = m;
						break;
					}
				}
				assert(ELEM.vertices[local_idx[i]] == n);

				//add this dof to basis_s of the support elements
				element_basis_s[support[i]].insert(n);
			}

			//mark unused support elements if the node was on the boundary
			for (;i<DOF_t::max_support; ++i) {
				support[i]   = (size_t) -1;
				local_idx[i] = (size_t) -1;
			}

			//create the DOF
			dofs[n] = DOF_t(n, support, local_idx);
			dofs[n].active = true;
			dofs[n].depth = 0;
			mesh2dof[n].insert(n);

			//mark DOF as part of the boundary
			if (VERTEX.boundary_faces.size()>0) {
				boundary_dofs.push_back(n);
			}
		}
	}


	//interpolate the coefs
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	Coef_t CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::eval_at_vertex(const size_t vertex_index) const
	{
		//get closest vertex in the mesh
		const auto& VERTEX = mesh.getVertex(vertex_index);

		//assemble element ancestors
		std::vector<size_t> elements = VERTEX.elems;
		for (size_t e_idx : VERTEX.elems) {
			mesh.getElementAncestors_Unlocked(e_idx, elements, false);
		}
		std::sort(elements.begin(), elements.end());
		auto last = std::unique(elements.begin(), elements.end());
		elements.erase(last, elements.end());

		//assemble active basis functions
		std::vector<size_t> dofs_to_eval;
		for (size_t e_idx : elements) {
			const auto& ELEM = mesh.getElement(e_idx);

			for (size_t d_idx : element_basis_s[e_idx]) {
				if (dofs[d_idx].active) {
					dofs_to_eval.push_back(d_idx);
				}
			}

			for (size_t d_idx : element_basis_a[e_idx]) {
				if (dofs[d_idx].active) {
					dofs_to_eval.push_back(d_idx);
				}
			}
		}
		
		std::sort(dofs_to_eval.begin(), dofs_to_eval.end());
		last = std::unique(dofs_to_eval.begin(), dofs_to_eval.end());
		dofs_to_eval.erase(last, dofs_to_eval.end());


		//evaluate the basis functions
		Coef_t result{0};
		for (size_t d_idx : dofs_to_eval) {
			result += coefs[d_idx] * dofs[d_idx].eval_at(VERTEX.coord, mesh);
		}

		return result;
	}


	//save the mesh and solution
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	void CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::save_as(const std::string filename, const int coef_dim, const bool use_ascii) const {
		//save the mesh topology
		mesh.save_as(filename, false, use_ascii);

		//re-open the file in append mode
		std::ofstream file(filename, std::ios::app);
		if (not file.is_open()){
			throw std::runtime_error("Couldn't open " + filename + " in append mode");
			return;
		}

		//map active dofs to vertices (helpful for debugging)
		const size_t nVertices = mesh.nVertices();
		const size_t nElements = mesh.nElements();

		//write the value of the field at the mesh vertices
		std::stringstream buffer;
		buffer << "POINT_DATA " << nVertices << "\n"
		       << "FIELD vertex_debug 4\n";

		//vertex values
		buffer << "values " << coef_dim << " " << nVertices << " float\n";
		std::vector<Coef_t> vertex_values(nVertices,0);
		#pragma omp parallel for
		for (size_t i=0; i<nVertices; ++i) {
			vertex_values[i] = this->eval_at_vertex(i);
		}
		for (size_t i=0; i<nVertices; ++i) {
			buffer << vertex_values[i] << " ";
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

		//write the index of the active basis function
		buffer << "active_basis_index 1 " << nVertices << " integer\n";
		for (size_t v_idx=0; v_idx<mesh.nVertices(); ++v_idx) {
			//TODO: update when we have non-nodal DOFs
			size_t dof_idx = (size_t) -1;
			for (size_t d_idx : mesh2dof[v_idx]) {
				if (dofs[d_idx].active) {
					dof_idx = d_idx;
					break;
				}
			}
			const int c = dof_idx == (size_t) -1 ? -1 : (int) dof_idx;
			buffer << c << " ";
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

		//write depth of the active basis function
		buffer << "depth 1 " << nVertices << " integer\n";
		for (size_t v_idx=0; v_idx<mesh.nVertices(); ++v_idx) {
			size_t dof_idx = (size_t) -1;
			for (size_t d_idx : mesh2dof[v_idx]) {
				if (dofs[d_idx].active) {
					dof_idx = d_idx;
					break;
				}
			}
			const int depth = dof_idx == (size_t) -1 ? -1 : dofs[dof_idx].depth;
			buffer << depth << " ";
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

		//write the coefficeint of the active basis function
		buffer << "coef 1 " << nVertices << " float\n";
		for (size_t v_idx=0; v_idx<mesh.nVertices(); ++v_idx) {
			size_t dof_idx = (size_t) -1;
			for (size_t d_idx : mesh2dof[v_idx]) {
				if (dofs[d_idx].active) {
					dof_idx = d_idx;
					break;
				}
			}
			const double c = dof_idx == (size_t) -1 ? 0 : coefs[dof_idx];
			buffer << c << " ";
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");



		//write element debug information
		buffer << "CELL_DATA " << nElements << "\n"
			   << "FIELD element_debug 2\n";

		size_t max_basis_s=0;
		size_t max_basis_a=0;
		for (size_t e_idx=0; e_idx<nElements; ++e_idx) {
			max_basis_s = std::max(max_basis_s, element_basis_s[e_idx].size());
			max_basis_a = std::max(max_basis_a, element_basis_a[e_idx].size());
		}

		buffer << "basis_s " << max_basis_s << " " << nElements << " integer\n";
		for (const auto& ELEM : mesh) {
			int i=0;
			for (size_t d_idx : element_basis_s[ELEM.index]) {
				buffer << d_idx << " ";
				i++;
			}
			for (;i<max_basis_s; ++i) {
				buffer << "-1 ";
			}
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

		buffer << "basis_a " << max_basis_a << " " << nElements << " integer\n";
		for (const auto& ELEM : mesh) {
			int i=0;
			for (size_t d_idx : element_basis_a[ELEM.index]) {
				buffer << d_idx << " ";
				i++;
			}
			for (;i<max_basis_a; ++i) {
				buffer << "-1 ";
			}
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

	}



	





	//print dof info
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	std::ostream& operator<<(std::ostream& os, const CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>& dofhandler) {
		os  << "\nCHARMS dofhandler:\n"
			<< "\tndofs (total)  : " << dofhandler.ndof() << "\n"
			<< "\tndofs (active) : " << dofhandler.get_dof_map().ndof() << "\n"
			<< "\tMesh at " << &dofhandler.mesh << " has " << dofhandler.mesh.nElements(true)
			<< " elements and " << dofhandler.mesh.nVertices() << " vertices\n";
		return os;
	}

}


