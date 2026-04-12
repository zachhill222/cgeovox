#pragma once

#include "gutil.hpp"
#include "voxel_mesh/mesh/keys/voxel_key.hpp"

#include <cstdint>
#include <type_traits>
#include <bitset>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <fstream>

#include <omp.h>

namespace gv::vmesh
{
	//concepts to constrain feature types to exactly match the/a
	//corresponding mesh feature
	template<typename F, typename M>
	concept MeshElementType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelElement>;

	template<typename F, typename M>
	concept MeshVertexType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelVertex>;

	template<typename F, typename M>
	concept MeshFaceType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelFace>;

	template<typename F, typename M>
	concept MeshFeatureType = MeshElementType<F,M> || MeshVertexType<F,M> || MeshFaceType<F,M>;

	template<typename M>
	concept VoxelMeshType = MeshElementType<typename M::VoxelElement, M> && 
							MeshVertexType<typename M::VoxelVertex, M> && 
							MeshFaceType<typename M::VoxelFace, M>;


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
		using VoxelVertex  = VoxelVertexKey<MAX_DEPTH_+1,MORTON_ORDER>;
		using VoxelFace    = VoxelFaceKey<MAX_DEPTH_+1,MORTON_ORDER>;
		using Mesh_t       = HierarchicalVoxelMesh<MAX_DEPTH_,MORTON_ORDER>; //this mesh type

		static constexpr uint64_t MAX_DEPTH = MAX_DEPTH_;
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = total_possible<VoxelElement>(MAX_DEPTH);

	protected:
		std::bitset<TOTAL_POSSIBLE_ELEMENTS>* active_elem = new std::bitset<TOTAL_POSSIBLE_ELEMENTS>(0);

		//alow other methods with a const ref to this class to request element to be activated
		//or deactivated. some other class will then process the request.
		//these requests are just the 'const' versions of activate(el) and deactivate(el).
		//they are thread safe in OpenMP.
		mutable std::vector<std::vector<VoxelElement>> request_active;
		mutable std::vector<std::vector<VoxelElement>> request_deactive;

	public:
		using GeoPoint_t = gutil::Point<3,double>; //points in space
		using GeoBox_t   = gutil::Box<3,double>; //axis-aligned boxes in space
		const GeoPoint_t low;
		const GeoPoint_t high;
		const GeoPoint_t diag;

		HierarchicalVoxelMesh(const GeoPoint_t low_, const GeoPoint_t high_) :
			low{gutil::elmin(low_, high_)},
			high{gutil::elmax(low_, high_)},
			diag{high-low}
		{
			#ifdef _OPENMP
				request_active.resize(omp_get_max_threads());
				request_deactive.resize(omp_get_max_threads());
			#else
				request_active.resize(1);
				request_deactive.resize(1);
			#endif
		}

		virtual ~HierarchicalVoxelMesh() {delete active_elem;}

		//simple querries and operations
		inline void reset() {active_elem->reset();}
		inline size_t nElements() const {return active_elem->count();}

		inline void activate(const VoxelElement el) {assert(el.is_valid()); active_elem->set(el.linear_index());}
		inline void activate(const VoxelElement el) const {
			assert(el.is_valid());
			#ifdef _OPENMP
				request_active[omp_get_thread_num()].push_back(el);
			#else
				request_active[0].push_back(el);
			#endif
		}

		inline void deactivate(const VoxelElement el) {assert(el.is_valid()); active_elem->reset(el.linear_index());}
		inline void deactivate(const VoxelElement el) const {
			assert(el.is_valid());
			#ifdef _OPENMP
				request_deactive[omp_get_thread_num()].push_back(el);
			#else
				request_deactive[0].push_back(el);
			#endif
		}

		inline bool is_active(const VoxelElement el) const {return active_elem->test(el.linear_index());}

		inline void set(const VoxelElement el, const bool flag = true) {active_elem->set(el.linear_index(), flag);}


		//process requested activations
		void process_request_active() {
			for (auto& request : request_active) {
				for (VoxelElement el : request) {
					activate(el);
					//check if all the children exist and are active
					//if so, deactivate the element
					bool all_children_active = true;
					for (VoxelElement c : el.children()) {
						if (c.exists() and !is_active(c)) {
							all_children_active = false;
							break;
						}
					}

					if (all_children_active) {
						deactivate(el);
					}
				}
				request.clear();
			}
		}

		//process requested deactivations
		void process_request_deactive() {
			for (auto& request : request_deactive) {
				for (VoxelElement el : request) {
					if (el.exists()) {
						std::cout << "deactivate el:\n" << el << std::endl;
						deactivate(el);}
				}
				request.clear();
			}
		}

		//layer operations
		template<typename Predicate = std::nullptr_t>
		void set_depth(const uint64_t depth, Predicate&& pred = nullptr) {
			//set to true by default. use the predicate if it is passed
			auto action = [&pred, this](VoxelElement el) {
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					active_elem->set(el.linear_index(), pred(el));
				}
				else {
					active_elem->set(el.linear_index());
				}
			};

			reset();
			//note that the predicate to activate an element is not the same
			//as the predicate to skip an element in the for_each loop
			for_each_depth<VoxelElement>(depth, action);
		}

		

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) {
			const uint64_t start = Key_t::depth_linear_start(depth);
			const uint64_t end   = Key_t::depth_linear_start(depth+1);
			Key_t key(depth,0); //element/vertex/face object (acts as an iterator)

			for (uint64_t idx=start; idx<end; ++idx, ++key) {
				assert(key.linear_index() == idx);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(key)) {
						continue;
					}
				}
				action(key);
			}
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each(Action&& action, Predicate&& pred = nullptr) {
			for (uint64_t depth=0; depth<MAX_DEPTH+1; ++depth) {
				for_each_depth<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
			}
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
			const uint64_t start = Key_t::depth_linear_start(depth);
			const uint64_t end   = Key_t::depth_linear_start(depth+1);
			Key_t key(depth,0); //element/vertex/face object (acts as an iterator)

			for (uint64_t idx=start; idx<end; ++idx, ++key) {
				assert(key.linear_index() == idx);
				if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
					if (!pred(key)) {
						continue;
					}
				}
				action(key);
			}
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each(Action&& action, Predicate&& pred = nullptr) const {
			for (uint64_t depth=0; depth<MAX_DEPTH+1; ++depth) {
				for_each_depth<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
			}
		}

		



		// template<typename Action, typename Predicate = std::nullptr_t> requires (MORTON_ORDER)
		// void for_each_color(const uint64_t depth, const uint64_t clr, Action&& action, Predicate&& pred = nullptr) {
		// 	assert(clr<8);
		// 	const uint64_t start = depth_linear_start(depth) + clr;
		// 	const uint64_t end   = depth_linear_start(depth+1);

		// 	for (uint64_t idx=start; idx<end; idx+=8) {
		// 		const VoxelElement el(depth,idx);
		// 		if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
		// 			if (!pred(el)) {continue;}
		// 		}
		// 		action(el);
		// 	}
		// }

		// template<typename Action, typename Predicate = std::nullptr_t> requires (!MORTON_ORDER)
		// void for_each_color(const uint64_t depth, const uint64_t clr, Action&& action, Predicate&& pred = nullptr) {
		// 	assert(clr<8);
		// 	const uint64_t start = depth_linear_start(depth);
		// 	const uint64_t end   = depth_linear_start(depth+1);

		// 	for (uint64_t idx=start; idx<end; ++idx) {
		// 		const VoxelElement el(depth,idx);
		// 		if (el.color()!=clr) {continue;}

		// 		if constexpr (!std::is_same_v<std::decay_t<Predicate>, std::nullptr_t>) {
		// 			if (!pred(el)) {continue;}
		// 		}
		// 		action(el);
		// 	}
		// }

		
		//hierarchy
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
			for (VoxelElement c : el.children()) {
				if (!c.exists()) {continue;}
				//it is possible that some children satisfy the predicate
				//even if the parent doesn't. this is especially likely if
				//the predicate is testing if the voxel is in some specified region (e.g., a true geometry)
				refine_to_depth(c, depth, std::forward<Predicate>(pred));
			}
		}

		//geometry operations
		inline GeoPoint_t ref2geo(const VoxelVertex vtx) const {
			return low + diag*vtx.normalized_coordinate();
		}

		uint64_t write_unstructured_vtk(std::ostream& os) const {
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
			auto action_c = [&](VoxelElement el) {
				if (is_active(el)) {
					cell_buffer << "8 ";
					for (int v=0; v<8; ++v) {
						// const VoxelVertex vtx = el.vertex(v).reduced_key();
						const VoxelVertex vtx = el.vertex(v);
						assert(vtx.is_valid());
						const uint64_t idx = vtx.linear_index();
						max_vtx_index = std::max(idx, max_vtx_index);
						cell_buffer << idx << " ";
					}
					cell_buffer << "\n";
				}
			};

			for_each<VoxelElement>(action_c);
			cell_buffer << "\n";

			const uint64_t n_vertices = max_vtx_index+1; //writing a few extra vertices is ok

			//write the vertex points up to the requested index
		   	std::ostringstream point_buffer;
			point_buffer << "POINTS " << n_vertices << " float\n";

			VoxelVertex vtx(0,0);
			for (uint64_t idx=0; idx<n_vertices; ++idx, ++vtx) {
				point_buffer << ref2geo(vtx) << "\n";
			}
			point_buffer << "\n";

			//Element type
			cell_buffer << "CELL_TYPES " << n_elements << "\n";
			for (size_t i=0; i<n_elements; ++i) {cell_buffer << "11 ";}
			cell_buffer << "\n\n";
			
			//write to file
			os << point_buffer.str();
			os << cell_buffer.str();

			//return the number of vertices for other methods to use
			return n_vertices;
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_cell_data_vtk(
			std::ostream& os,
			const std::string& data_header, 
			CellDataLookup&& lookup
			) const {
			
			std::ostringstream buffer;
			buffer << data_header << "\n";

			auto pred   = [this](const VoxelElement el) {return this->is_active(el);};
			auto action = [&buffer, &lookup](const VoxelElement el) {buffer << lookup(el) << "\n";};
			for_each<VoxelElement>(action, pred);
			buffer << "\n";
			os << buffer.str();
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_point_data_vtk(
			std::ostream& os,
			const std::string& data_header,
			const uint64_t n_vertices,
			CellDataLookup&& lookup
			) const {

			os << data_header << "\n";

			std::ostringstream point_buffer;
			VoxelVertex vtx(0,0);
			for (uint64_t idx=0; idx<n_vertices; ++idx, ++vtx) {
				point_buffer << lookup(vtx) << "\n";
			}
			point_buffer << "\n";

			os << point_buffer.str();
		}

		void save_unstructured_mesh(const std::string filename = "voxel_mesh_unstructured.vtk") const {
			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("HierarchicalVoxelMesh::save_unstructured_mesh - could not open file: " + filename);
			}

			write_unstructured_vtk(file);
			file << "CELL_DATA " << nElements() << "\n";

			auto lookup_dijk   = [](VoxelElement el) {return gutil::Point<4,uint64_t>{el.depth(), el.i(), el.j(), el.k()};};
			auto lookup_linear = [](VoxelElement el) {return el.depth_linear_index();};
			auto lookup_color  = [](VoxelElement el) {return el.color();};

			append_unstructured_cell_data_vtk(file, "SCALARS color int 1\nLOOKUP_TABLE default", lookup_color);
			append_unstructured_cell_data_vtk(file, "SCALARS dijk int 4\nLOOKUP_TABLE default",  lookup_dijk);
			append_unstructured_cell_data_vtk(file, "SCALARS linear int 1\nLOOKUP_TABLE default",lookup_linear);
		}
	};


}