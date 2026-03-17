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
		IsCharmsDOF DOF_t,
		typename Coef_t = double>
	class CharmsDOFhandler
	{
	private:
		static constexpr size_t MAX_CHILDREN = static_cast<size_t>(DOF_t::nchildren);
		void distribute_nodal();

	protected:
		std::vector<DOF_t>  dofs;
		std::vector<std::array<size_t,MAX_CHILDREN>> dof_children;
		std::vector<std::array<size_t,MAX_CHILDREN>> dof_parents;
		std::vector<std::unordered_set<size_t>> element_basis_s; //track basis functions per-element on the same level of refinement
		std::vector<std::unordered_set<size_t>> element_basis_a; //track basis functions per-element on coarser levels of refinement

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
		template<typename CoefContainer_t=std::vector<Coef_t>>
		Coef_t interpolate(const typename Mesh_t::Point_t& coord, const CoefContainer_t& container) const;

		//save the mesh and nodal evaluations to a file
		void save_as(const std::string filename, const int coef_dim=1, const bool use_ascii=false) const;

		//active/deactivate basis functions
		void activate(const size_t idx) {
			assert(idx<this->ndof());
			assert(element_basis_a.size() == mesh.nElements(false));
			assert(element_basis_s.size() == mesh.nElements(false));

			if (dofs[idx].active) {return;}
			dofs[idx].active = true;

			//loop through support elements
			for (int i=0; i<DOF_t::max_support; ++i) {
				const size_t el_idx = dofs[idx].support_idx[i];
				if (el_idx != (size_t) -1) {
					//element activation is handled outside the dofhandler
					//populate basis_s and basis_a either way (in case the element is later activated)

					//ensure the support elements have this basis function
					element_basis_s[el_idx].insert(idx);

					//add this basis function as an ancestor basis function to any descendent elements
					std::vector<size_t> descendents;
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

		bool is_refinable(const size_t idx) const {
			assert(idx<this->ndof());
			
			if (!dofs[idx].active) {return false;}

			for (size_t c_idx : dof_children[idx]) {
				if (c_idx != (size_t) -1) {return false;}
			}

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
				        assert(ELEM.is_active);
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
			this->reserve(dofs.size()+MAX_CHILDREN*refine_dof_list.size());

			//ensure basis_s and basis_a have the same length as the total number of elements
			element_basis_s.resize(mesh.nElements(false));
			element_basis_a.resize(mesh.nElements(false));

			//add all new DOFs
			for (size_t parent_idx : refine_dof_list) {
				DOF_t& PARENT = dofs[parent_idx];

				//update basis_a for the newly created elements
				for (size_t e_idx : PARENT.support_idx) {
					if (e_idx == (size_t) -1) {continue;}
					const auto& ELEM = mesh.getElement(e_idx);
					assert(ELEM.children.size() > 0);

					for (size_t c_idx : ELEM.children) {
						element_basis_a[c_idx].insert(element_basis_s[parent_idx].begin(), element_basis_s[parent_idx].end());
						element_basis_a[c_idx].insert(element_basis_a[parent_idx].begin(), element_basis_a[parent_idx].end());
					}
				}


				//create children, add to list of dofs, populate relations
				auto children = PARENT.make_children(mesh);
				
				//determine which children are new (different parents can refine to create the same children)
				std::array<size_t, MAX_CHILDREN> child_dof_idx;
				child_dof_idx.fill((size_t) -1);

				//add any new dofs to the list and record the index of all child dofs
				for (size_t c_idx=0; c_idx<MAX_CHILDREN; ++c_idx) {
					//if this child is not a new dof, then the first support element will have this
					//child as a dof in basis_s
					const DOF_t& CHILD = children[c_idx];

					//check if the child was actually created (this should only ever fail on the boundary)
					if (CHILD.global_idx == (size_t) -1) {continue;}
					
					//check if the child previously existed as a DOF
					size_t existing_dof_idx = (size_t) -1;
					for (size_t sib_idx : element_basis_s[CHILD.support_idx[0]]) {
						if (CHILD == dofs[sib_idx]) {
							existing_dof_idx = sib_idx;
							break;
						}
					}

					//get the index of the child DOF and add to list if needed
					//note the contents of children[] is invalid
					if (existing_dof_idx == (size_t) -1) {
						const size_t new_dof_idx = dofs.size();
						child_dof_idx[c_idx] = new_dof_idx;
						dofs.push_back(children[c_idx]);
						
						dof_parents.emplace_back();
						dof_parents[new_dof_idx].fill((size_t) -1);
						
						dof_children.emplace_back();
						dof_children[new_dof_idx].fill((size_t) -1);

						coefs.emplace_back(0);
					} else {
						child_dof_idx[c_idx] = existing_dof_idx;
					}
				}

				//populate relations
				for (size_t c_idx=0; c_idx<MAX_CHILDREN; ++c_idx) {
					const size_t child_idx = child_dof_idx[c_idx];
					if (child_idx == (size_t) -1) {continue;}

					dof_children[parent_idx][c_idx] = child_idx;
					dof_parents[child_idx][c_idx] = parent_idx;

					//basis_s and basis_a are handled when the basis function is activated
				}

				//activate functions and update coefficients
				if constexpr (HIERARCHICAL) {
					for (size_t c_idx=0; c_idx<MAX_CHILDREN; ++c_idx) {
						const size_t child_idx = child_dof_idx[c_idx];
						if (child_idx == (size_t) -1) {continue;}

						const DOF_t& CHILD = dofs[child_idx];
						activate(child_idx); //always activate to get basis_s populated correctly
						if (CHILD.global_idx == PARENT.global_idx) {deactivate(child_idx);}
						
						coefs[child_idx] = Coef_t{0};
					}
				}
				else {
					deactivate(parent_idx);
					for (size_t c_idx=0; c_idx<children.size(); ++c_idx) {
						const size_t child_idx = child_dof_idx[c_idx];
						if (child_idx == (size_t) -1) {continue;}
						
						const DOF_t& CHILD = dofs[child_idx];
						activate(child_idx);

						//represent the parent function as a linear combination of the children functions
						coefs[child_idx] = coefs[parent_idx]*PARENT.get_child_coef(c_idx);
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

		//create CSR/CSC matrix with the correct sparsity structure
		//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
		//RowMajor format is better for settng boundary conditions, but ColMajor might be better for Eigen routines
		template<int Format=Eigen::RowMajor, typename T=double>
		Eigen::SparseMatrix<T,Format> init_matrix() const;
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

			//populate parent/child relationships with null values
			dof_parents[n].fill((size_t) -1);
			dof_children[n].fill((size_t) -1);

			//mark DOF as part of the boundary
			if (VERTEX.boundary_faces.size()>0) {
				boundary_dofs.push_back(n);
			}
		}
	}


	//interpolate the coefs
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	template<typename CoefContainer_t>
	Coef_t CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::interpolate(const typename Mesh_t::Point_t& coord, const CoefContainer_t& container) const {
		//get closest vertex in the mesh
		const size_t vtx_idx = mesh.closestVertex(coord);
		const auto& VERTEX = mesh.getVertex(vtx_idx);

		//determine if the container provided compressid (active only) coefficents or all coefficients
		bool all_dofs = container.size() == dofs.size();
		if (!all_dofs) {assert(container.size() == dof_map.ndof());}

		//collect all ancestor elements of contain this coordinate.
		//this can be done by getting the ancestors of each of the active elements this coordinate belongs to
		//(the coordinate may be on the boundary between two active elements that have different parents)
		std::vector<size_t> support_elements;
		for (size_t	e_idx : VERTEX.elems) {
			const auto& ELEM = mesh.getElement(e_idx);
			if (!ELEM.is_active) {continue;}

			//construct vtk_element to check for containment
			auto* vtk_elem = gv::mesh::_VTK_ELEMENT_FACTORY<Mesh_t::SPACE_DIM, Mesh_t::REF_DIM, typename Mesh_t::Vertex_t::Scalar_t, double>(ELEM);
			std::vector<typename Mesh_t::Point_t> elem_vertices;
			elem_vertices.reserve(ELEM.vertices.size());
			for (size_t v_idx : ELEM.vertices) {elem_vertices.push_back(mesh.getVertex(v_idx).coord);}

			if (vtk_elem->contains(elem_vertices, coord)) {
				mesh.getElementAncestors_Unlocked(e_idx, support_elements, false);
			}

			delete vtk_elem;
		}

		//ensure we only look at each element once
		std::sort(support_elements.begin(), support_elements.end());
		auto last = std::unique(support_elements.begin(), support_elements.end());
		support_elements.erase(last, support_elements.end());





		//loop through elements that may contain the coordinate
		//TODO: improve the mesh elements
		Coef_t result{0};



		for (size_t e_idx : VERTEX.elems) {
			const auto& ELEM = mesh.getElement(e_idx);
			
			auto* vtk_elem = gv::mesh::_VTK_ELEMENT_FACTORY<Mesh_t::SPACE_DIM, Mesh_t::REF_DIM, typename Mesh_t::Vertex_t::Scalar_t>(ELEM);
			std::vector<typename Mesh_t::Point_t> elem_vertices;
			elem_vertices.reserve(ELEM.vertices.size());
			for (size_t v_idx : ELEM.vertices) {elem_vertices.push_back(mesh.getVertex(v_idx).coord);}

			if (vtk_elem->contains(elem_vertices, coord)) {
				//get reference coordinate
				const auto ref_coord = static_cast<typename DOF_t::RefPoint_t>(vtk_elem->geometric_to_reference(elem_vertices, coord));
				
				std::cout << "\nQuerry point: " << coord << " reference point: " << ref_coord << " (element " << e_idx << ")\n";

				//add active basis_s
				for (size_t d_idx : element_basis_s[e_idx]) {
					if (computed_basis.contains(d_idx)) {continue;}

					if (dofs[d_idx].active) {
						const DOF_t& DOF   = dofs[d_idx];
						const Coef_t& COEF = all_dofs ? container[d_idx] : container[dof_map.global2compressed[d_idx]];

						//get the support index for this element in this basis function
						int i;
						bool found=false;
						for (i=0; i<DOF.support_idx.size(); i++) {
							if (DOF.support_idx[i] == e_idx) {
								found=true;
								break;
							}
						}
						
						//increment the result
						if (found) {
							assert(DOF.support_idx[i]==e_idx);
							std::cout << "basis_s: " << d_idx << " (val= " << DOF.eval(ref_coord,i) << " coef= " << COEF << ")\n";
							computed_basis.insert(d_idx);
							result += COEF * DOF.eval(ref_coord, i);
						}
					}
				}

				//add active basis_a
				for (size_t d_idx : element_basis_a[e_idx]) {
					if (dofs[d_idx].active) {
						const DOF_t& DOF   = dofs[d_idx];
						const Coef_t& COEF = all_dofs ? container[d_idx] : container[dof_map.global2compressed[d_idx]];

						//get the support index for this element in this basis function
						int i;
						bool found=false;
						for (i=0; i<DOF.support_idx.size(); i++) {
							if (DOF.support_idx[i] == e_idx) {
								found=true;
								break;
							}
						}

						//increment the result
						if (found) {
							assert(DOF.support_idx[i]==e_idx);
							std::cout << "basis_a: " << d_idx << " (val= " << DOF.eval(ref_coord,i) << " coef= " << COEF << ")\n";
							computed_basis.insert(d_idx);
							result += COEF * DOF.eval(ref_coord, i);
						}
						
					}
				}
			}

			//clean memory
			delete vtk_elem;
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

		//write the value of the field at the mesh vertices
		std::stringstream buffer;
		const size_t nVertices = mesh.nVertices();
		buffer << "POINT_DATA " << nVertices << "\n"
		       << "FIELD field 1 \n";

		buffer << "values " << coef_dim << " " << nVertices << " float\n";
		for (auto it=mesh.vertexBegin(); it!=mesh.vertexEnd(); ++it) {
			buffer << this->interpolate(it->coord, this->coefs) << " ";
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");
	}



	//create CSR/CSC matrix with the correct sparsity structure
	//the DOF Map must be current (i.e, no refines/coarsening since the last time the map was computed)
	template<gv::mesh::HierarchicalColorableMeshType Mesh_t, IsCharmsDOF DOF_t, typename Coef_t>
	template<int Format, typename T>
	Eigen::SparseMatrix<T,Format> CharmsDOFhandler<Mesh_t,DOF_t,Coef_t>::init_matrix() const {
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


