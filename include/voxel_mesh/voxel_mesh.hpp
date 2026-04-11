#pragma once

#include "gutil.hpp"
#include "voxel_mesh/keys/voxel_key.hpp"

#include <cstdint>
#include <type_traits>
#include <bitset>
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

	template<int MAX_DEPTH_=10, bool MORTON_ORDER=false> requires (MAX_DEPTH_>=0)
	class HierarchicalVoxelMesh
	{
	public:
		using VoxelElement = VoxelElementKey<MAX_DEPTH_+1,MORTON_ORDER>;
		using VoxelVertex = VoxelVertexKey<MAX_DEPTH_+1,MORTON_ORDER>;
		using VoxelFace = VoxelFaceKey<MAX_DEPTH_+1,MORTON_ORDER>;

		static constexpr uint64_t MAX_DEPTH = MAX_DEPTH_;
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = ((uint64_t{1} << 3*(MAX_DEPTH+1)) - 1) / 7;
	protected:
		std::bitset<TOTAL_POSSIBLE_ELEMENTS>* active_elem = new std::bitset<TOTAL_POSSIBLE_ELEMENTS>(0);

	public:
		using GeoPoint_t = gutil::Point<3,double>; //points in space
		using GeoBox_t   = gutil::Box<3,double>; //axis-aligned boxes in space
		const GeoPoint_t low;
		const GeoPoint_t high;
		const GeoPoint_t diag;

		HierarchicalVoxelMesh(const GeoPoint_t low_, const GeoPoint_t high_) :
			low{gutil::elmin(low_, high_)},
			high{gutil::elmax(low_, high_)},
			diag{high-low} {}

		virtual ~HierarchicalVoxelMesh() {delete active_elem;}

		inline size_t nElements() const {return active_elem->count();}

		static constexpr uint64_t start_depth(const uint64_t depth) {
			assert(depth<=MAX_DEPTH);
			return ((uint64_t{1} << (3*depth)) -1 )/7;
		}

		//indexing global indexing
		static constexpr uint64_t index(const VoxelElement e) {
			return e.linear_index();
		}

		static constexpr uint64_t index(const VoxelFace f) {
			return f.linear_index();
		}

		static constexpr uint64_t index(const VoxelVertex v) {
			return v.linear_index();
		}


		//layer operations
		template<typename Predicate = std::nullptr_t>
		void set_depth(const uint64_t depth, Predicate&& pred = nullptr) {
			//set to true by default. use the predicate if it is passed
			const uint64_t start = start_depth(depth);
			const uint64_t end = start_depth(depth+1);
			for (uint64_t idx = start; idx<end; ++idx) {
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					active_elem->set(idx, pred(VoxelElement(depth, idx-start)));
				}
				else {
					active_elem->set(idx);
				}
			}
		}

		inline void reset() {active_elem->reset();}

		template<typename Action, typename Predicate = std::nullptr_t>
		void for_each_element_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) {
			const uint64_t start = start_depth(depth);
			const uint64_t end   = start_depth(depth+1);

			for (uint64_t idx=start; idx<end; ++idx) {
				const VoxelElement el(depth,idx-start);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(el)) {continue;}
				}
				action(el);
			}
		}

		template<typename Action, typename Predicate = std::nullptr_t>
		void for_each_element(Action&& action, Predicate&& pred = nullptr) {
			for (uint64_t depth=0; depth<MAX_DEPTH; ++depth) {
				for_each_element_depth(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
			}
		}

		template<typename Action, typename Predicate = std::nullptr_t>
		void for_each_vertex_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) {
			const uint64_t start = VoxelVertex::depth_linear_start(depth);
			const uint64_t end   = VoxelVertex::depth_linear_start(depth+1);

			for (uint64_t idx=start; idx<end; ++idx) {
				const VoxelVertex vtx(depth,idx-start);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(vtx)) {continue;}
				}
				action(vtx);
			}
		}

		template<typename Action, typename Predicate = std::nullptr_t>
		void for_each_vertex(Action&& action, Predicate&& pred = nullptr) {
			for (uint64_t depth=0; depth<MAX_DEPTH; ++depth) {
				for_each_vertex_depth(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
			}
		}



		template<typename Action, typename Predicate = std::nullptr_t> requires (MORTON_ORDER)
		void for_each_color(const uint64_t depth, const uint64_t clr, Action&& action, Predicate&& pred = nullptr) {
			assert(clr<8);
			const uint64_t start = start_depth(depth) + clr;
			const uint64_t end   = start_depth(depth+1);

			for (uint64_t idx=start; idx<end; idx+=8) {
				const VoxelElement el(depth,idx);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(el)) {continue;}
				}
				action(el);
			}
		}

		template<typename Action, typename Predicate = std::nullptr_t> requires (!MORTON_ORDER)
		void for_each_color(const uint64_t depth, const uint64_t clr, Action&& action, Predicate&& pred = nullptr) {
			assert(clr<8);
			const uint64_t start = start_depth(depth);
			const uint64_t end   = start_depth(depth+1);

			for (uint64_t idx=start; idx<end; ++idx) {
				const VoxelElement el(depth,idx);
				if (el.color()!=clr) {continue;}

				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(el)) {continue;}
				}
				action(el);
			}
		}

		//atomic operations
		inline void activate(const VoxelElement el) {
			active_elem->set(index(el));
		}

		inline void deactivate(const VoxelElement el) {
			active_elem->reset(index(el));
		}

		inline bool is_active(const VoxelElement el) const {
			return active_elem->test(index(el));
		}

		inline void set(const VoxelElement el, const bool flag = true) {
			active_elem->set(index(el), flag);
		}

		template<typename Predicate = std::nullptr_t>
		void refine(const VoxelElement el, Predicate&& pred = nullptr) {
			deactivate(el);
			for (int c=0; c<8; ++c) {
				const VoxelElement child = el.child(c);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					set(child, pred(child));
				}
				else {
					activate(child);
				}
			}
		}

		template<typename Predicate = std::nullptr_t>
		void coarsen(const VoxelElement el, Predicate&& pred = nullptr) {
			if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
				set(el, pred(el));
			}
			else {
				activate(el);
			}

			for (int c=0; c<8; ++c) {deactivate(el.child(c));}
		}

		//tree operations
		template<typename Predicate = std::nullptr_t>
		void refine_to_depth(const VoxelElement el, const uint64_t depth, Predicate&& pred = nullptr) {
			if (el.depth() < depth) {
				refine(el, std::forward<Predicate>(pred));
			}
			for (int c=0; c<8; c++) {
				refine(el.child(c), depth, std::forward<Predicate>(pred));
			}
		}

		//geometry operations
		inline GeoPoint_t ref2geo(const VoxelVertex vtx) const {
			return low + diag*vtx.normalized_coordinate();
		}

		//methods to write to a file in various vtk formats
		//these methods write to an already open file/output stream
		void write_depth_structured_vtk(std::ostream& os, const uint64_t depth) const {
			assert(depth <= MAX_DEPTH);
			const uint32_t N = uint32_t{1} << depth; //number of elements per dimension

			os << "# vtk DataFile Version 2.0\n"
			   << "Depth " << depth << " structured mesh\n"
			   << "ASCII\n"
			   << "DATASET STRUCTURED_POINTS\n"
			   << "DIMENSIONS " << N+1 << " " << N+1 << " " << N+1 << "\n" //number of vertices in per axis
			   << "ORIGIN " << low << "\n"
			   << "SPACING " << (high-low)/GeoPoint_t{N,N,N} << "\n\n";
		}

		
		// template<typename CellDataLookup = std::nullptr_t>
		// void append_depth_structured_cell_data_vtk(
		// 	std::ostream& os, 
		// 	const uint64_t depth, 
		// 	const std::string& data_header, 
		// 	CellDataLookup&& lookup
		// 	) const {
			
		// 	assert(depth <= MAX_DEPTH);
		// 	std::ostringstream buffer;
		// 	buffer << data_header << "\n";

		// 	const uint64_t start = start_depth(depth);
		// 	const uint64_t end   = start_depth(depth+1);
		// 	for (uint64_t idx=start; idx<end; ++idx) {
		// 		const VoxelElement el(depth,idx-start);
		// 		buffer << lookup(el) << "\n";
		// 	}

		// 	os << buffer.str();
		// }

		void write_unstructured_vtk(std::ostream& os) {
			os << "# vtk DataFile Version 2.0\n"
			   << "Hierarchical mesh as unstructured mesh\n"
			   << "ASCII\n\n"
			   << "DATASET UNSTRUCTURED_GRID\n";

			//Loop through the active elements
			//write the element buffer and track the largest index
			//of the vertices to write
			//additionally, write the cell buffer
			std::ostringstream cell_buffer;
			const uint64_t n_elements = nElements();
			cell_buffer << "CELLS " << n_elements << " " << 9*n_elements << "\n";

			uint64_t max_vtx_index=0;
			auto action = [&](VoxelElement el) {
				if (is_active(el)) {
					cell_buffer << "8 ";
					for (int v=0; v<8; ++v) {
						// const VoxelVertex vtx = el.vertex(v).reduced_key();
						const VoxelVertex vtx = el.vertex(v);
						const uint64_t idx = vtx.linear_index();
						max_vtx_index = std::max(idx, max_vtx_index);
						cell_buffer << idx << " ";
					}
					cell_buffer << "\n";
				}
			};

			for_each_element(action);
			cell_buffer << "\n";

			const uint64_t n_vertices = max_vtx_index+1; //writing a few extra vertices is ok

			//write the vertex points up to the requested index
		   	std::ostringstream point_buffer;
			point_buffer << "POINTS " << n_vertices << " float\n";
			for (uint64_t depth=0; depth<=MAX_DEPTH; ++depth) {
				const uint64_t start = VoxelVertex::depth_linear_start(depth);
				const uint64_t end   = VoxelVertex::depth_linear_start(depth+1);
				
				for (uint64_t idx=start; idx < std::min(end, n_vertices); ++idx) {
					point_buffer << ref2geo(VoxelVertex(depth, idx-start)) << "\n";
				}
			}
			point_buffer << "\n";

			//Element type
			cell_buffer << "CELL_TYPES " << n_elements << "\n";
			for (size_t i=0; i<n_elements; ++i) {cell_buffer << "11 ";}
			cell_buffer << "\n\n";
			
			//write to file
			os << point_buffer.str();
			os << cell_buffer.str();
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_cell_data_vtk(
			std::ostream& os,
			const std::string& data_header, 
			CellDataLookup&& lookup
			) const {
			
			std::ostringstream buffer;
			buffer << data_header << "\n";

			auto pred   = [&](const VoxelElement el) {return this->is_active(el);};
			auto action = [&](const VoxelElement el) {buffer << lookup(el) << "\n";};
			for_each_element(action, pred);
			buffer << "\n";
			os << buffer.str();
		}

		// void save_hierarchy(const std::string prefix = "voxel_mesh_hierarchy_") const {
		// 	uint64_t max_depth=0;
		// 	for (uint64_t i=0; i<MAX_DEPTH; ++i) {
		// 		if (!elements[i].empty()) {max_depth=i;}
		// 	}

		// 	for (uint64_t depth=0; depth<=max_depth; ++depth) {
		// 		const std::string filename = prefix + std::to_string(depth) + ".vtk";
		// 		std::ofstream file(filename);
		// 		if (!file.is_open()) {
		// 			throw std::runtime_error("HierarchicalVoxelMesh::save_hierarchy - could not open file: " + filename);
		// 		}

		// 		write_depth_structured_vtk(file, depth);

		// 		const uint64_t n_elements = uint64_t{1} << (3*depth);
		// 		file << "CELL_DATA " << n_elements << "\n";
		// 		auto lookup = [](VoxelElement el){return el.is_active();};
		// 		append_depth_structured_cell_data_vtk(file, depth, "SCALARS is_active int 1\nLOOKUP_TABLE default", lookup);
		// 	}
		// }

		void save_unstructured_mesh(const std::string filename = "voxel_mesh_unstructured.vtk") {
			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("HierarchicalVoxelMesh::save_unstructured_mesh - could not open file: " + filename);
			}

			write_unstructured_vtk(file);
			// file << "CELL_DATA " << u_elem.size() << "\n";

			// auto lookup_active = [](VoxelElement el) {return el.is_active();};
			// auto lookup_depth  = [](VoxelElement el) {return el.depth();};
			// auto lookup_ijk    = [](VoxelElement el) {return gutil::Point<3,uint64_t>{el.i(), el.j(), el.k()};};
			// auto lookup_linear = [](VoxelElement el) {return el.depth_linear_index<MORTON_ORDER>();};

			// append_unstructured_cell_data_vtk(file, "SCALARS is_active int 1\nLOOKUP_TABLE default", lookup_active);
			// append_unstructured_cell_data_vtk(file, "SCALARS depth int 1\nLOOKUP_TABLE default", lookup_depth);
			// append_unstructured_cell_data_vtk(file, "SCALARS ijk int 3\nLOOKUP_TABLE default", lookup_ijk);
			// append_unstructured_cell_data_vtk(file, "SCALARS linear int 1\nLOOKUP_TABLE default", lookup_linear);
		}
	};


}