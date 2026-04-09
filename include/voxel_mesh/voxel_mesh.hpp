#pragma once

#include "gutil.hpp"
#include "voxel_mesh/voxel_key.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>
#include <map>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <fstream>

namespace gv::vmesh
{
	//This class is for a hierarchical voxel mesh.
	//This structure allows us to very efficiently store elements, vertices, faces, etc.
	//The base mesh is 1x1x1 so that every vertex is (in reference coordinates) a dyadic rational number
	//Elements, vertices, and faces all have special index/key structs for their storage and logical relations.
	//All information for every element, vertex, and face is compressed into a 64-bit unsigned integer
	//With the element/vertex/face keys, we must have a maximum depth of 15. If more is needed (unlikely) we can stitch together
	//multiples of these meshes.
	class HierarchicalVoxelMesh
	{
	public:
		using VoxelElement = VoxelElementKey<>;
		using VoxelVertex = VoxelVertexKey<>;

	protected:
		static constexpr uint64_t MAX_DEPTH = 16; //strict upper bound so that we can index via depth.
		std::array<int, MAX_DEPTH> element_vector_sorted;
		std::array<std::vector<VoxelElement>, MAX_DEPTH> elements;

		//storage for converting to an unstructured mesh
		std::map<VoxelVertex, size_t> u_vertex_map;
		std::vector<VoxelVertex> u_vert;
		std::vector<std::array<size_t,8>> u_elem_conn;
		std::vector<VoxelElement> u_elem;
		size_t insert_vertex(VoxelVertex k) {
			k = k.reduced_key();
			auto [it, inserted] = u_vertex_map.emplace(k, u_vert.size());
			if (inserted) {u_vert.push_back(k);}
			return it->second;
		}

		size_t get_vertex(VoxelVertex k) {
			auto it = u_vertex_map.find(k.reduced_key());
			if (it == u_vertex_map.end()) {return (size_t) -1;}
			else {return it->second;}
		}

		//iterator accessors
		std::vector<VoxelElement>::const_iterator get_iterator(const VoxelElement el) const {
			assert(el.depth() < MAX_DEPTH);
			assert(element_vector_sorted[el.depth()]);

			auto& lv = elements[el.depth()]; //hierarchy level
			return std::lower_bound(lv.begin(), lv.end(), el);
		}

		std::vector<VoxelElement>::iterator get_iterator(const VoxelElement el) {
			assert(el.depth() < MAX_DEPTH);
			assert(element_vector_sorted[el.depth()]);

			auto& lv = elements[el.depth()]; //hierarchy level
			return std::lower_bound(lv.begin(), lv.end(), el);
		}

	public:
		using GeoPoint_t = gutil::Point<3,double>; //points in space
		using GeoBox_t   = gutil::Box<3,double>; //axis-aligned boxes in space
		const GeoPoint_t low;
		const GeoPoint_t high;
		const GeoPoint_t diag;

		HierarchicalVoxelMesh(int initial_depth, const GeoPoint_t low_, const GeoPoint_t high_) :
			low{gutil::elmin(low_, high_)},
			high{gutil::elmax(low_, high_)},
			diag{high-low}
			{
				set_coarsest_level(initial_depth);
			}

		size_t nElements() const {
			size_t result = 0;
			for (const auto& lv : elements) {
				result += lv.size();
			}
			return result;
		}

		constexpr GeoPoint_t ref2geo(VoxelVertex v) const {
			return low + v.normalized_coordinate()*diag;
		}

		void set_coarsest_level(uint64_t initial_depth) {
			//ensure the specified depth is reasonable and has room for refinement
			initial_depth = initial_depth > MAX_DEPTH/2 ? MAX_DEPTH/2 : initial_depth;

			for (auto& lv : elements) {lv.clear();}
			element_vector_sorted.fill(1);

			const uint64_t n_elements = uint64_t{1} << (3*initial_depth);
			elements[initial_depth].reserve(n_elements);
			
			//use constructor from the linear index (i + N*(j + N*k))
			//TODO: ranges and views seem like they would be good here
			for (uint64_t e_idx=0; e_idx<n_elements; ++e_idx) {
				elements[initial_depth].emplace_back(initial_depth, e_idx);
				elements[initial_depth][e_idx].set_active(true);
			}
		}

		void insert_sorted(const VoxelElement el) {
			auto it = get_iterator(el); //iterator to where el is or should be
			auto& lv = elements[el.depth()];
			if (it!=lv.end() && *it == el) {return;}
			lv.insert(it, el);
		}

		void insert_unsorted(const VoxelElement el) {
			assert(el.depth() < MAX_DEPTH);
			elements[el.depth()].push_back(el);
			element_vector_sorted[el.depth()] = 0;
		}

		void sort_elements() {
			for (uint64_t i=0; i<MAX_DEPTH; ++i) {
				if (element_vector_sorted[i]) {continue;}

				auto& container = elements[i];
				std::sort(container.begin(), container.end());
				element_vector_sorted[i] = 1;
			}
		}

		bool contains(const VoxelElement el) const {
			auto it = get_iterator(el);
			const auto& lv = elements[el.depth()]; //hierarchy level
			return it!=lv.end() && *it==el;
		}

		void erase(const VoxelElement el) {
			auto it = get_iterator(el);
			auto& lv = elements[el.depth()]; //hierarchy level
			
			if (it!=lv.end() && *it==el) {
				lv.erase(it);
			}
		}

		//traversal methods
		//traverse over all descendants and do something
		//the action should act on a VoxelElement&
		//optionally, a predicate can be passed (const VoxelElement to bool or similar) to reduce unnecessary recursion
		//for example [](VoxelElement el) {return el.active();}
		template<typename ElementAction = std::nullptr_t, bool top_first=true, typename ChildPredicate = std::nullptr_t>
		void for_each_descendant(const VoxelElement el, ElementAction&& action = nullptr, ChildPredicate&& pred = nullptr) {
			if constexpr (std::is_same_v<std::decay_t<ElementAction>, std::nullptr_t>) {return;}

			const auto it = get_iterator(el);
			const auto& lv = elements[el.depth()];
			assert(it!=lv.end() && *it==el);

			if constexpr (top_first) {action(*it);}

			//loop over children
			if (el.depth()+1 < MAX_DEPTH) {
				const auto& clv = elements[el.depth()+1];
				for (int c=0; c<8; ++c) {
					const VoxelElement child = el.child(c);
					const auto cit = get_iterator(child);
					if (cit!=clv.end() && *cit==child) {
						if constexpr (!std::is_same_v<std::decay_t<ChildPredicate>, std::nullptr_t>) {
							if (!pred(std::as_const(*cit))) {continue;}
						}

						for_each_descendant<ElementAction,top_first,ChildPredicate>(
							child, std::forward<ElementAction>(action), std::forward<ChildPredicate>(pred));
					}
				}
			}

			//looped over children or had no children
			if constexpr (!top_first) {action(*it);}
		}

		//traverse over all descendants and do something
		//the action should act on a const VoxelElement&
		template<typename ElementAction = std::nullptr_t, bool top_first=true, typename ChildPredicate = std::nullptr_t>
		void for_each_descendant(const VoxelElement el, ElementAction&& action = nullptr, ChildPredicate&& pred = nullptr) const {
			if constexpr (std::is_same_v<std::decay_t<ElementAction>, std::nullptr_t>) {return;}

			const auto  it = get_iterator(el);
			const auto& lv = elements[el.depth()];
			assert(it!=lv.end() && *it==el);

			if constexpr (top_first) {action(*it);}

			//loop over children
			if (el.depth()+1 < MAX_DEPTH) {
				const auto& clv = elements[el.depth()+1];
				for (int c=0; c<8; ++c) {
					const VoxelElement child = el.child(c);
					
					const auto cit = get_iterator(child);
					if (cit!=clv.end() && *cit==child) {
						if constexpr (!std::is_same_v<std::decay_t<ChildPredicate>, std::nullptr_t>) {
							if (!pred(std::as_const(*cit))) {continue;}
						}

						for_each_descendant<ElementAction,top_first,ChildPredicate>(
							child, std::forward<ElementAction>(action), std::forward<ChildPredicate>(pred));
					}
				}
			}

			//looped over children or had no children
			if constexpr (!top_first) {action(*it);}
		}

		//traverse all elements (allow to modify)
		template<typename ElementAction = std::nullptr_t>
		void for_each_element(ElementAction&& action = nullptr, const uint64_t depth=MAX_DEPTH) {
			if constexpr (std::is_same_v<std::decay_t<ElementAction>, std::nullptr_t>) {return;}

			if (depth<MAX_DEPTH) {
				for (VoxelElement& el : elements[depth]) {
					action(el);
				}
			}
			else {
				for (auto& lv : elements) {
					for (VoxelElement& el : lv) {
						action(el);
					}
				}
			}
		}

		//traverse all elements (constant)
		template<typename ElementAction = std::nullptr_t>
		void for_each_element(ElementAction&& action = nullptr, const uint64_t depth=MAX_DEPTH) const {
			if constexpr (std::is_same_v<std::decay_t<ElementAction>, std::nullptr_t>) {return;}

			if (depth<MAX_DEPTH) {
				for (VoxelElement el : elements[depth]) {
					action(el);
				}
			}
			else {
				for (const auto& lv : elements) {
					for (VoxelElement el : lv) {
						action(el);
					}
				}
			}
		}

		//refine all elements at a certain depth
		void refine_depth(const uint64_t depth) {
			auto action = [this](VoxelElement& el){
				if (el.is_active()) {
					this->refine(el);
				}
			};

			for_each_element(action, depth);
		}

		//refine and coarsen methods
		//optionally pass a predicate to dictate when a child should be activated.
		//by default activate all children
		//this can be usefull when approximating geometry that is not a rectangle
		template<typename ChildPredicate = std::nullptr_t>
		void refine(VoxelElement& el, ChildPredicate&& pred = nullptr) {
			assert(el.depth()+1 < MAX_DEPTH);
			
			el.set_active(false); //deactivate the parent

			//create children
			elements[el.depth()+1].reserve(elements[el.depth()+1].size() + 8);
			for (int c=0; c<8; c++) {
				VoxelElement child{el.child(c)};
				if constexpr (std::is_same_v<std::decay_t<ChildPredicate>, std::nullptr_t>) {
					child.set_active(true);
				}
				else {
					child.set_active(pred(child)); //always set in case the default changes in VoxelElement
				}

				insert_sorted(child);
			}
		}


		//deactivate all descendents, activate the specified element if an active child was found
		void collapse(VoxelElement& el) {
			bool found_active_descendant = false;
			auto action = [&found_active_descendant](VoxelElement& el) {
				found_active_descendant = found_active_descendant || el.is_active();
				el.set_active(false);
			};

			for_each_descendant(el, action);
			el.set_active(found_active_descendant);
		}


		//convert to an unstructured mesh
		template<typename ElementPredicate = std::nullptr_t>
		void build_unstructured_mesh(ElementPredicate&& pred = nullptr) {
			u_vertex_map.clear();
			u_vert.clear();
			u_elem_conn.clear();
			u_elem.clear();

			//construct the hierarchy
			for (uint64_t d=0; d<MAX_DEPTH; ++d) {
				for (VoxelElement el : elements[d]) {
					if constexpr (!std::is_same_v<std::decay_t<ElementPredicate>, std::nullptr_t>) {
						if (!pred(el)) {continue;}
					}

					std::array<size_t,8> connectivity;
					for (int v_idx=0; v_idx<8; ++v_idx) {
						connectivity[v_idx] = insert_vertex(el.vertex(v_idx));
					}
					u_elem_conn.push_back(connectivity);
					u_elem.push_back(el);
				}
			}
		}


		//methods to write to a file in various vtk formats
		//these methods write to an already open file/output stream
		void write_depth_structured_vtk(std::ostream& os, const uint64_t depth) const {
			assert(depth < MAX_DEPTH);
			const uint32_t N = uint32_t{1} << depth; //number of elements per dimension

			os << "# vtk DataFile Version 2.0\n"
			   << "Depth " << depth << " structured mesh\n"
			   << "ASCII\n"
			   << "DATASET STRUCTURED_POINTS\n"
			   << "DIMENSIONS " << N+1 << " " << N+1 << " " << N+1 << "\n" //number of vertices in per axis
			   << "ORIGIN " << low << "\n"
			   << "SPACING " << (high-low)/GeoPoint_t{N,N,N} << "\n\n";
		}

		
		template<typename CellDataLookup = std::nullptr_t>
		void append_depth_structured_cell_data_vtk(
			std::ostream& os, 
			const uint64_t depth, 
			const std::string& data_header, 
			CellDataLookup&& lookup
			) const {
			
			assert(depth < MAX_DEPTH);
			std::ostringstream buffer;
			buffer << data_header << "\n";

			if (!element_vector_sorted[depth]) {
				throw std::runtime_error("HierarchicalVoxelMesh::append_depth_structured_cell_data_vtk - elements must be sorted");
			}

			uint64_t gap_start{0};
			for (VoxelElement el : elements[depth]) {
				//check if we skipped any indices
				for (; gap_start<el.depth_linear_index(); ++gap_start) {
					VoxelElement blank_element(depth, gap_start);
					blank_element.set_active(false);
					buffer << lookup(blank_element) << "\n"; //newline is probably safer for most return types
				}

				buffer << lookup(el) << "\n";
				gap_start = el.depth_linear_index()+1;
			}

			//write any final skipped indices at the end
			const uint64_t n_elements = uint64_t{1} << (3*depth);
			for (; gap_start<n_elements; ++gap_start) {
				VoxelElement blank_element(depth, gap_start);
				blank_element.set_active(false);
				buffer << lookup(blank_element) << "\n";
			}

			os << buffer.str();
		}

		void write_unstructured_vtk(std::ostream& os) const {
			const size_t n_elements = u_elem.size();
			const size_t n_vertices = u_vert.size();

			std::ostringstream buffer;
			buffer << "# vtk DataFile Version 2.0\n"
				   << "Hierarchical mesh as unstructured mesh\n"
				   << "ASCII\n\n"
				   << "DATASET UNSTRUCTURED_GRID\n";

			//Vertex locations
			buffer << "POINTS " << n_vertices << " float\n";
			for (VoxelVertex vtx : u_vert) {
				buffer << ref2geo(vtx) << "\n";
			}
			buffer << "\n";
			os << buffer.str();
			buffer.str("");

			//Element connectivity
			buffer << "CELLS " << n_elements << " " << 9*n_elements << "\n";
			for (const auto& conn : u_elem_conn) {
				buffer << "8 ";
				for (size_t i : conn) {buffer << i << " ";}
				buffer << "\n";
			}
			buffer << "\n";
			os << buffer.str();
			buffer.str("");

			//Element type
			buffer << "CELL_TYPES " << n_elements << "\n";
			for (size_t i=0; i<n_elements; ++i) {buffer << "11 ";}
			buffer << "\n\n";
			os << buffer.str();
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_cell_data_vtk(
			std::ostream& os,
			const std::string& data_header, 
			CellDataLookup&& lookup
			) const {
			
			std::ostringstream buffer;
			buffer << data_header << "\n";

			for (VoxelElement el : u_elem) {
				buffer << lookup(el) << "\n";
			}
			buffer << "\n";
			os << buffer.str();
		}

		void save_hierarchy(const std::string prefix = "voxel_mesh_hierarchy_") const {
			uint64_t max_depth=0;
			for (uint64_t i=0; i<MAX_DEPTH; ++i) {
				if (!elements[i].empty()) {max_depth=i;}
			}

			for (uint64_t depth=0; depth<=max_depth; ++depth) {
				const std::string filename = prefix + std::to_string(depth) + ".vtk";
				std::ofstream file(filename);
				if (!file.is_open()) {
					throw std::runtime_error("HierarchicalVoxelMesh::save_hierarchy - could not open file: " + filename);
				}

				write_depth_structured_vtk(file, depth);

				const uint64_t n_elements = uint64_t{1} << (3*depth);
				file << "CELL_DATA " << n_elements << "\n";
				auto lookup = [](VoxelElement el){return el.is_active();};
				append_depth_structured_cell_data_vtk(file, depth, "SCALARS is_active int 1\nLOOKUP_TABLE default", lookup);
			}
		}

		void save_unstructured_mesh(const std::string filename = "voxel_mesh_unstructured.vtk") const {
			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("HierarchicalVoxelMesh::save_unstructured_mesh - could not open file: " + filename);
			}

			write_unstructured_vtk(file);
			file << "CELL_DATA " << u_elem.size() << "\n";

			auto lookup_active = [](VoxelElement el) {return el.is_active();};
			auto lookup_depth  = [](VoxelElement el) {return el.depth();};
			auto lookup_ijk    = [](VoxelElement el) {return gutil::Point<3,uint64_t>{el.i(), el.j(), el.k()};};
			auto lookup_linear = [](VoxelElement el) {return el.depth_linear_index();};

			append_unstructured_cell_data_vtk(file, "SCALARS is_active int 1\nLOOKUP_TABLE default", lookup_active);
			append_unstructured_cell_data_vtk(file, "SCALARS depth int 1\nLOOKUP_TABLE default", lookup_depth);
			append_unstructured_cell_data_vtk(file, "SCALARS ijk int 3\nLOOKUP_TABLE default", lookup_ijk);
			append_unstructured_cell_data_vtk(file, "SCALARS linear int 1\nLOOKUP_TABLE default", lookup_linear);
		}
	};


}