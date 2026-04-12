#pragma once

#include "voxel_mesh/mesh/voxel_mesh.hpp"
#include "voxel_mesh/fem/fem_kernel.hpp"

#include <type_traits>
#include <cstdint>

namespace gv::vmesh
{
	//note that dofs and basis functions are used interchangably
	template<VoxelMeshType Mesh_type, typename DOF_type>
	class DofHandler
	{
	public:
		using DOF_t  = DOF_type;
		using Key_t  = typename DOF_t::Key_t;
		using Mesh_t = Mesh_type;
		using Elem_t = typename Mesh_t::VoxelElement;
		using Vert_t = typename Mesh_t::VoxelVertex;
		using Face_t = typename Mesh_t::VoxelFace;

		static_assert(
			std::same_as<Key_t, typename Mesh_t::VoxelElement> ||
			std::same_as<Key_t, typename Mesh_t::VoxelFace>    ||
			std::same_as<Key_t, typename Mesh_t::VoxelVertex>,
			"DofHandler - the feature key for the DOF must match the corresponding feature key of the mesh.");
		static_assert(std::same_as<Elem_t, typename DOF_type::QuadElem_t>,
			"DofHandler - the DOF quadrature element and the mesh element must be of the same type");

		//we can iterate over the mesh and REQUEST mesh refinement
		//however, mesh refinement must be done outside of this class
		//because multiple DOF handlers can reference the same mesh
		static constexpr uint64_t MAX_DEPTH = Mesh_t::MAX_DEPTH;
		const Mesh_t& mesh;

		static constexpr uint64_t TOTAL_POSSIBLE_DOFS = total_possible<Key_t>(MAX_DEPTH);
		
		//constructor and destructor
		DofHandler(const Mesh_t& mesh) : mesh(mesh) {}
		virtual ~DofHandler() {
			delete active_dofs;
			delete stale_dofs;
		}

		//non-copyable
		DofHandler(const DofHandler&) = delete;
		DofHandler& operator=(const DofHandler&) = delete;

		//movable by construction
		DofHandler(DofHandler&& other) :
			mesh(other.mesh), 
			active_dofs(other.active_dofs), 
			stale_dofs(other.stale_dofs)
		{
			other.active_dofs = nullptr;
			other.stale_dofs = nullptr;
		}

		//can't move by assignment
		DofHandler& operator=(DofHandler&&) = delete;

		//maintain lists of dofs that have been newly activated or deactivated
		//these are public so that the problem class can incrementally update any matrices
		//any time a dof is activated or deactivated, it's corresponding stale_dof bit is set to true.
		//another class may do something with this information and should then set the bit to false.
		std::bitset<TOTAL_POSSIBLE_DOFS>* stale_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>(0);

		protected:
		//track which dofs are active based on a global numbering system of the voxel features.
		//instances of the DOF_t act as an iterator into this set.
		std::bitset<TOTAL_POSSIBLE_DOFS>* active_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>(0);

		//track a compressed list of active dofs
		//the bitfield is the "source of truth"
		//but this is used to convert the global dof number
		//into the dof number that the matrix solver will use
		//to help incrementally update scalar fields and integration matrices,
		std::vector<DOF_t> active_dof_list_prev;
		std::vector<DOF_t> active_dof_list_curr;

		public:
		//simple querries
		inline constexpr uint64_t n_dofs() const {return active_dofs->count();}

		inline constexpr bool is_active(const DOF_t dof) const {return active_dofs->test(dof.key.linear_index());}
		inline constexpr void set_active(const DOF_t dof, const bool b) {active_dofs->set(dof.key.linear_index(), b);}

		inline constexpr bool is_stale(const DOF_t dof) const {return stale_dofs->test(dof.key.linear_index());}
		inline constexpr void set_stale(const DOF_t dof, const bool b) {stale_dofs->set(dof.key.linear_index(), b);}

		//simple management operations
		inline void reset_active() {active_dofs->reset();}
		inline void reset_stale() {stale_dofs->reset();}
		void snapshot_dof_list() {
			active_dof_list_prev = std::move(active_dof_list_curr);
			active_dof_list_curr.clear();
		}

		//activate all dofs at a certain depth if they have an active support element
		//all other dofs are inactive. the stale bits are reset as it is assumed
		//that this method will only be called when the user intends to "set/reset" the problem.
		void set_depth(const uint64_t dd) {
			active_dofs->reset();
			stale_dofs->reset();

			auto action = [this](Key_t key) {
				const DOF_t dof{key};
				if (has_active_support(dof)) {
					active_dofs->set(key.linear_index());
				}
			};

			//TODO: call in parallel if needed
			mesh.template for_each_depth<Key_t>(dd,action);
		}

		//check if a dof has an active support element
		bool has_active_support(const DOF_t dof) const {
			for (Elem_t el : dof.support()) {
				if (el.exists() and mesh.is_active(el)) {
					return true;
				}
			}
			return false;
		}

		bool has_active_basis(const Elem_t el) const {
			for (DOF_t dof : DOF_t::dofs_on_elem(el)) {
				if (dof.exists() and is_active(dof)) {
					return true;
				}
			}
			return false;
		}

		//gather active basis sets
		std::vector<DOF_t> basis_s(const Elem_t el) const {
			std::vector<DOF_t> bs;
			if (!el.exists()) {return bs;}

			bs.reserve(DOF_t::N_DOF_PER_ELEM);
			for (DOF_t dof : DOF_t::dofs_on_elem(el)) {
				if (is_active(dof)) {
					bs.push_back(dof);
				}
			}
			return bs;
		}

		std::vector<DOF_t> basis_a(Elem_t el) const {
			std::vector<DOF_t> ba;
			if (!el.exists() or el.depth()==0) {return ba;}
			
			ba.reserve(DOF_t::N_DOF_PER_ELEM * (el.depth()-1));
			for (uint64_t dd=el.depth(); dd>0; --dd) {
				auto bs = basis_s(el.parent());
				ba.insert(ba.end(), bs.begin(), bs.end());
				el = el.parent();
			}
			return ba;
		}

		//atomic operations
		void activate(const DOF_t dof) {
			const uint64_t idx = dof.linear_index();
			assert(dof.key.is_valid());

			//request the mesh to activate the support elements
			for (Elem_t el : dof.support()) {
				if(el.exists()) {mesh.activate(el);}
			}
			active_dofs->set(idx,true);
			stale_dofs->set(idx,true);
		}

		void deactivate(const DOF_t dof) {
			const uint64_t idx = dof.linear_index();
			
			active_dofs->set(idx,false);
			stale_dofs->set(idx,true);
			for (Elem_t el : dof.support()) {
				if (el.exists() and basis_s(el).empty()) {mesh.deactivate(el);}
			}
		}

		template<bool HIERARCHICAL=false>
		void refine(const DOF_t dof) {
			assert(dof.key.depth() < MAX_DEPTH);
			assert(is_active(dof));

			if constexpr (HIERARCHICAL) {
				for (DOF_t c : dof.children()) {
					if (c.exists() and !c.key.parent().exists()) {
						activate(c);
					}
				}
			}
			else {
				deactivate(dof);
				for (DOF_t c : dof.children()) {
					if (c.exists()) {
						activate(c);
					}
				}
			}
		}

		template<bool HIERARCHICAL=false>
		void unrefine(const DOF_t dof) {
			if constexpr (HIERARCHICAL) {
				assert(is_active(dof));

				for (DOF_t c : dof.children()) {
					if (c.exists() and !c.key.parent().exists()) {
						deactivate(c);
					}
				}
			}
			else {
				assert(!is_active(dof));
				for (DOF_t c : dof.children()) {
					if (c.exists()) {
						deactivate(c);
					}
				}
				activate(dof);
			}
		}

		template<bool HIERARCHICAL=false, typename Predicate=std::nullptr_t>
		void refine_depth(const uint64_t depth, Predicate&& pred=nullptr) {
			assert(depth < Mesh_t::MAX_DEPTH);

			auto action = [this](Key_t key) {
				const DOF_t dof{key};
				if (is_active(dof))	{refine<HIERARCHICAL>(DOF_t{key});}
			};

			mesh.template for_each_depth<Key_t>(depth, action, pred);
		}

		//tranfer computations between mesh refinements
		void compress_dof_numbers() {
			active_dof_list_curr.clear();
			active_dof_list_curr.reserve(active_dofs->count());

			auto action = [this](const Key_t key) {
				if (active_dofs->test(key.linear_index())) {
					active_dof_list_curr.emplace_back(key);
				}
			};

			mesh.template for_each<Key_t>(action);

			//TODO: allow a custom sorting comparator
			std::sort(active_dof_list_curr.begin(), active_dof_list_curr.end());
		}

		template<typename CoefContainer_t, typename EvalMethod>
		void init_coefs(CoefContainer_t& coefs, EvalMethod&& eval) const {
			assert(coefs.size() == active_dof_list_curr.size());
			for (uint64_t i=0; i<coefs.size(); ++i) {
				const DOF_t dof = active_dof_list_curr[i];
				assert(dof.exists());
				assert(dof.is_valid());
				assert(is_active(dof));
				coefs[i] = eval(dof);
			}
		}

		template<typename CoefContainer_t>
		void update_coefs(CoefContainer_t& old_coefs, CoefContainer_t& new_coefs) {
			//transfer each coefficient of old into new
			//or split its contribution into its children in new
			//or compress its contribution into its parent in new
			//the coef lists must always be sorted so that the lookup is fast
			assert(old_coefs.size() == active_dof_list_prev.size());
			assert(new_coefs.size() == active_dof_list_curr.size());

			//lamda to directly transfer a coefficient
			auto transfer = [&new_coefs, this](const double val, const DOF_t dof) {
				auto it = std::lower_bound(active_dof_list_curr.begin(), active_dof_list_curr.end(), dof);
				assert(it != active_dof_list_curr.end());
				uint64_t idx = std::distance(active_dof_list_curr.begin(), it);
				new_coefs[idx] = val;
				return true;
			};

			//lambda to increment a child or parent dof
			auto increment = [&new_coefs, this](const double val, const DOF_t dof) {
				auto it = std::lower_bound(active_dof_list_curr.begin(), active_dof_list_curr.end(), dof);
				assert(it != active_dof_list_curr.end());
				uint64_t idx = std::distance(active_dof_list_curr.begin(), it);
				new_coefs[idx] += val;
			};

			for (uint64_t i=0; i<old_coefs.size(); ++i) {
				const DOF_t old_dof = active_dof_list_prev[i];
				const double val    = old_coefs[i];

				//transfer, split, or compress
				if (is_active(old_dof)) {transfer(val, old_dof);}
				else {
					bool has_active_child = false;
					const auto child_coefs = old_dof.children_coef();
					const auto child_dofs  = old_dof.children();
					for (uint64_t c=0; c<old_dof.N_CHILDREN; ++c) {
						DOF_t child = child_dofs[c];
						if (child.exists() and is_active(child)) {
							has_active_child = true;
							increment(val*child_coefs[c], child);
						}
					}

					if (!has_active_child) {
						const auto parent_coefs = old_dof.parent_coefs();
						const auto parent_dofs = old_dof.parents();
						for (uint64_t p=0; p<old_dof.N_PARENTS; ++p) {
							DOF_t parent = parent_dofs[p];
							if (!parent.exists()) {break;} //parents are packed to the front, children are not
							if (is_active(parent)) {
								increment(val*parent_coefs[p], parent);
							}
						}
					}
				}
			}
		}

		//append solution to a vtk file
		template<typename CoefContainer_t>
		std::vector<double> interpolate_to_vertices(const CoefContainer_t& coefs, uint64_t n_vertices) const {
			//increment the position value for every active dof
			std::vector<double> result(n_vertices, 0.0);
			for (uint64_t i=0; i<active_dof_list_curr.size(); ++i) {
				const DOF_t dof = active_dof_list_curr[i];
				const double c  = coefs[i];

				//iterate over the support and update the values
				for (const Elem_t el : dof.support()) {
					if (!el.exists() || !mesh.is_active(el)) {continue;}

					for (int v=0; v<8; ++v) {
						const Vert_t vtx = el.vertex(v);
						const uint64_t vidx = vtx.reduced_key().linear_index();
						assert(vidx<n_vertices);

						Elem_t spt = el;
						assert(spt.depth()==dof.key.depth());
						typename DOF_t::RefPoint_t xi {
							2.0*(static_cast<double>(vtx.i())-static_cast<double>(el.i())) - 1.0,
							2.0*(static_cast<double>(vtx.j())-static_cast<double>(el.j())) - 1.0,
							2.0*(static_cast<double>(vtx.k())-static_cast<double>(el.k())) - 1.0,
						};
						result[vidx] += c*dof.eval(spt, xi);
					}
				}
			}
			return result;
		}
	};
}