#pragma once

#include <cassert>
#include <concepts>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree_util.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// BasicParallelOctree - Thread-safe spatial data structure
///
/// This octree supports concurrent insertions from multiple OpenMP threads via push_back_async().
/// Space must be pre-allocated via resize() before async insertions. Use shrink_to_fit() to reclaim
/// unused space afterwards.
///
/// Key features:
/// - Immediate data storage: push_back_async() stores data immediately and returns its index
/// - Deferred tree updates: A dedicated worker thread updates the octree structure asynchronously
/// - flush() waits until all pending tree updates complete
///
/// Thread safety:
/// - Multiple threads can call push_back_async() concurrently
/// - If the same new data is inserted by multiple threads, each gets a different index but only
///   one is correct. If data already exists in the tree, the correct index is safely returned.
////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gv::util {
	/////////////////////////////////////////////////
	/// BasicParallelOctree - Main container class
	///
	/// @tparam Data_t      Type of data to store
	/// @tparam SINGLE_DATA If true, data goes in first valid leaf only.
	///                     If false, data goes in all overlapping leaves.
	/// @tparam DIM         Spatial dimension (typically 3)
	/// @tparam N_DATA      Max data indices per leaf node
	/// @tparam T           Floating-point type for bounding boxes
	/////////////////////////////////////////////////
	template<typename Data_t, bool SINGLE_DATA, int DIM=3, int N_DATA=16, Float T=float>
	class BasicParallelOctree {
		static_assert(DIM > 0, "Dimension must be positive");
		static_assert(N_DATA > 0, "N_DATA must be positive");

	public:
		// Type aliases
		using Node_t  = OctreeParallelNode<DIM, N_DATA, T>;
		using Point_t = typename Node_t::Point_t;
		using Box_t   = typename Node_t::Box_t;
		
		static constexpr int N_CHILDREN = Node_t::N_CHILDREN;

	private:
		// Work item for async insertion
		struct DataBuffer {
			size_t  idx;
			Node_t* target_node;
		};

		// Tree structure
		Node_t* _root = nullptr;

		// Data storage
		std::vector<Data_t> _data;
		std::atomic<size_t> _next_data_idx{0};

		// Worker thread for tree updates
		std::thread _inserter_thread;
		std::condition_variable _inserter_cv;
		mutable std::mutex _inserter_mtx;
		std::atomic<bool> _running{true};
		
		// Pending work counter
		std::atomic<size_t> _total_pending{0};
		
		// Per-thread work queues
		std::vector<ThreadLocalQueue<DataBuffer>*> _all_queues;

	public:
		//============================================================
		// Construction and destruction
		//============================================================
		explicit BasicParallelOctree(const Box_t &bbox) 
			: _root(new Node_t(bbox)) 
		{
			resetDataIdx(_root);

			// Set up thread-local queues
			#ifdef _OPENMP
				int max_threads = omp_get_max_threads();
			#else
				int max_threads = 1;
			#endif
			
			_all_queues.resize(max_threads, nullptr);
			for (int i = 0; i < max_threads; i++) {
				_all_queues[i] = new ThreadLocalQueue<DataBuffer>();
				_all_queues[i]->queue_id = static_cast<size_t>(i);
			}

			// Start worker thread
			_inserter_thread = std::thread([this]() {
				_inserter_loop();
			});
		}

		BasicParallelOctree() : BasicParallelOctree(Box_t{}) {}

		virtual ~BasicParallelOctree() {
			// Signal worker thread to stop
			_running.store(false, std::memory_order_release);
			_inserter_cv.notify_one();

			// Wait for worker thread
			if (_inserter_thread.joinable()) {
				_inserter_thread.join();
			}

			// Clean up queues
			for (auto* queue : _all_queues) {
				delete queue;
			}

			// Clean up tree
			delete _root;
		}

		// Non-copyable, non-movable
		BasicParallelOctree(const BasicParallelOctree&) = delete;
		BasicParallelOctree& operator=(const BasicParallelOctree&) = delete;
		BasicParallelOctree(BasicParallelOctree&&) = delete;
		BasicParallelOctree& operator=(BasicParallelOctree&&) = delete;

		//============================================================
		// Container interface
		//============================================================
		
		void reserve(size_t length) {
			_data.reserve(length);
		}

		void shrink_to_fit() {
			_data.resize(_next_data_idx.load(std::memory_order_acquire));
			_data.shrink_to_fit();
		}

		bool empty() const {
			return _data.empty();
		}

		size_t size() const {
			return _next_data_idx.load(std::memory_order_acquire);
		}

		size_t capacity() const {
			return _data.capacity();
		}

		void resize(size_t length) {
			// Remove indices for data being removed
			if (_root != nullptr) {
				for (size_t i = length; i < _data.size(); i++) {
					_recursive_remove_idx(_root, i, _data[i]);
				}
			}
			_data.resize(length);
		}

		void clear() {
			_data.clear();
			_next_data_idx.store(0, std::memory_order_release);
			
			// Rebuild tree structure
			Node_t* new_root = new Node_t(_root->bbox);
			resetDataIdx(new_root);
			delete _root;
			_root = new_root;
		}

		//============================================================
		// Element access
		//============================================================
		
		const Data_t& operator[](size_t idx) const {
			assert(idx < size());
			return _data[idx];
		}

		Data_t& operator[](size_t idx) {
			assert(idx < size());
			return _data[idx];
		}

		//============================================================
		// Iterators
		//============================================================
		
		auto begin()        { return _data.begin(); }
		auto begin()  const { return _data.cbegin(); }
		auto cbegin() const { return _data.cbegin(); }
		auto end()          { return _data.end(); }
		auto end()    const { return _data.cend(); }
		auto cend()   const { return _data.cend(); }

		//============================================================
		// Spatial queries
		//============================================================
		
		const Box_t& bbox() const {
			return _root->bbox;
		}

		void set_bbox(const Box_t& new_bbox) {
			_recursive_expand_bbox(new_bbox);

			// Reinsert data into new nodes if needed
			if constexpr (!SINGLE_DATA) {
				for (size_t j = 0; j < _data.size(); j++) {
					_recursive_insert_data(_root, _data[j], j);
				}
			}
		}

		/// Find existing data in tree
		/// Returns index if found, (size_t)-1 if not found
		size_t find(const Data_t& val) const {
			return _recursive_find_index<true>(_root, val);
		}

		/// Reinsert data at given index (call only from single thread)
		void reinsert(size_t idx) {
			assert(idx < _data.size());
			
			_recursive_remove_idx(_root, idx, _data[idx]);

			if (!isValid(_root->bbox, _data[idx])) {
				_recursive_resize_to_fit_data(_data[idx], 8);
				
				if constexpr (!SINGLE_DATA) {
					for (size_t j = 0; j < _data.size(); j++) {
						_recursive_insert_data(_root, _data[j], j);
					}
				}
			} else {
				_recursive_insert_data(_root, _data[idx], idx);
			}
		}

		//============================================================
		// Insertion operations
		//============================================================
		
		/// Synchronous insertion
		size_t push_back(const Data_t &val) {
			Data_t copy(val);
			return push_back(std::move(copy));
		}

		/// Synchronous insertion (move version)
		size_t push_back(Data_t &&val) {
			// Try to find existing data
			Node_t* start_node = _recursive_find_best_node(_root, val);
			size_t idx = _recursive_find_index<true>(start_node, val);
			
			if (idx != (size_t)-1) {
				assert(_data[idx] == val);
				return idx;
			}

			// Insert new data
			idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
			if (_data.size() <= idx) {
				_data.resize(idx + 1);
			}
			_data[idx] = std::move(val);

			// Update tree immediately
			[[maybe_unused]] int flag = _recursive_insert_data(start_node, _data[idx], idx);
			assert(flag == 1);

			// Flush down the data if needed
			_recursive_push_data_down(start_node);
			return idx;
		}

		/// Asynchronous insertion (for parallel use with OpenMP)
		size_t push_back_async(Data_t &&val) {
			#ifndef _OPENMP
				return push_back(std::move(val));
			#endif

			// Try to find existing data
			Node_t* start_node = _recursive_find_best_node(_root, val);
			size_t idx = _recursive_find_index<false>(start_node, val);
			
			if (idx != (size_t)-1) {
				return idx;
			}

			// Reserve index and store data immediately
			idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
			_data[idx] = std::move(val);

			// Queue tree update for worker thread
			#ifdef _OPENMP
				int thread_no = omp_get_thread_num();
			#else
				int thread_no = 0;
			#endif
			
			ThreadLocalQueue<DataBuffer>* thread_queue = _all_queues[thread_no];
			thread_queue->push(DataBuffer{idx, start_node});
			// thread_queue->pending_count.fetch_add(1, std::memory_order_release);

			// Notify worker thread
			_total_pending.fetch_add(1, std::memory_order_release);
			_inserter_cv.notify_one();

			return idx;
		}

		/// Wait for all pending async insertions to complete
		void flush() {
			// Wake up worker thread
			_inserter_cv.notify_one();

			// Spin until all work is done
			while (_total_pending.load(std::memory_order_acquire) > 0) {
				std::this_thread::yield();
			}

			// Ensure all memory writes are visible
			std::atomic_thread_fence(std::memory_order_seq_cst);

			// Extra yield for safety
			_recursive_push_data_down(_root);
			
			std::this_thread::yield();
		}

		//============================================================
		// Data validation
		//============================================================
		
		/// Check if there are any duplicate values
		/// All data must be pushed down for this method to work.
		/// Data will be pushed down after flush() is called.
		/// Inserting data single-threaded will push the data to the leaf at the
		/// time of insertion.
		void duplicateCheck() const {_recursive_duplicate_data(_root);}

		//============================================================
		// Summary information
		//============================================================
		void treeSummary(size_t &n_nodes, size_t &n_idx, size_t &n_idx_cap, size_t &n_leafs, int &max_depth) const {
			_recursive_node_properties(_root, n_nodes, n_idx, n_idx_cap, n_leafs, max_depth);
		}

	private:
		//============================================================
		// Worker thread
		//============================================================
		
		void _inserter_loop() {
			while (_running.load(std::memory_order_acquire)) {
				// Wait for work
				{
					std::unique_lock<std::mutex> lock(_inserter_mtx);
					_inserter_cv.wait(lock);
				}

				// Process all queued work
				while (_total_pending.load(std::memory_order_acquire) > 0) {
					for (auto* thread_queue : _all_queues) {
						if (thread_queue==nullptr) continue;
						
						while (!thread_queue->empty()) {
							DataBuffer work;
							thread_queue->pop(work);

							// Try to insert into tree
							size_t existing_idx = _recursive_find_index<true>(work.target_node, _data[work.idx]);
							
							if (existing_idx == (size_t)-1) {
								// Not found, insert it
								[[maybe_unused]] int flag = _recursive_insert_data(
									work.target_node, _data[work.idx], work.idx);
								assert(flag == 1);
							} else if (existing_idx != work.idx) {
								// Found duplicate - this shouldn't happen with proper usage
								std::cout << "error" << std::endl;
								assert(false);
							}

							// Decrement counters AFTER work is complete
							// thread_queue->pending_count.fetch_sub(1, std::memory_order_release);
							_total_pending.fetch_sub(1, std::memory_order_release);
						}
					}
				}
			}
		}

		//============================================================
		// Abstract interface (must be overridden)
		//============================================================
		
		/// Determine if data belongs in the given bounding box
		virtual bool isValid(const Box_t &bbox, const Data_t &val) const = 0;

		//============================================================
		// Tree operations
		//============================================================
		
		/// Remove index from node and all descendants
		void _recursive_remove_idx(Node_t* node, size_t idx) {
			if (isLeaf(node)) {
				removeDataIdx(node, idx);
			} else {
				for (int c = 0; c < N_CHILDREN; c++) {
					_recursive_remove_idx(node->children[c], idx);
				}
			}
		}

		/// Remove index from node and descendants (optimized with validity check)
		void _recursive_remove_idx(Node_t* node, size_t idx, const Data_t &val) {
			if (isLeaf(node)) {
				removeDataIdx(node, idx);
			} else {
				for (int c = 0; c < N_CHILDREN; c++) {
					if (isValid(node->children[c]->bbox, val)) {
						_recursive_remove_idx(node->children[c], idx, val);
					}
				}
			}
		}

		/// Find best node to start insertion/search
		Node_t* _recursive_find_best_node(Node_t* node, const Data_t &val) {
			if (isLeaf(node)) {
				return (node->parent != nullptr) ? node->parent : node;
			}

			// Traverse to child containing data
			for (int c = 0; c < N_CHILDREN; c++) {
				if (isValid(node->children[c]->bbox, val)) {
					return _recursive_find_best_node(node->children[c], val);
				}
			}

			// Data not in any child
			return _root;
		}

		/// Find index of data in tree
		/// @tparam UNLOCKED If true, skip locking (for single-threaded use)
		template<bool UNLOCKED>
		size_t _recursive_find_index(const Node_t* node, const Data_t &val) const {
			if (node==nullptr) {return (size_t) -1;}

			//check this node's data
			//this will usually only run in the leaf nodes
			//if push_back_async is running, data will not be pushed down until flush() is called
			if (node->data_idx!=nullptr) {
				if constexpr (UNLOCKED) {
					for (int i = 0; i < node->cursor; i++) {
						size_t idx = node->data_idx[i];
						if (_data[idx] == val) {
							return idx;
						}
					}
				} else {
					std::shared_lock<std::shared_mutex> lock(node->_rw_mtx);
					for (int i = 0; i < node->cursor; i++) {
						size_t idx = node->data_idx[i];
						if (_data[idx] == val) {
							return idx;
						}
					}
				}
			} 

			//data has not been found, check the children
			for (int c = 0; c < N_CHILDREN; c++) {
				if (node->children[c]==nullptr) {continue;}

				if (isValid(node->children[c]->bbox, val)) {
					size_t idx = _recursive_find_index<UNLOCKED>(node->children[c], val);
					if (idx != (size_t)-1) {
						return idx;
					}
				}
			}

			return (size_t)-1;
		}

		/// Expand root bounding box to contain new region
		void _recursive_expand_bbox(const Box_t& new_bbox) {
			if (_root->bbox.contains(new_bbox)) {
				return;
			}

			// Double the bounding box and find best placement
			Box_t expanded_root_bbox = 2.0 * _root->bbox;
			int max_vertices = -1;
			int best_sibling_number = -1;

			for (int c = 0; c < N_CHILDREN; c++) {
				Point_t offset = _root->bbox.voxelvertex(c) - expanded_root_bbox.voxelvertex(c);
				Box_t test_box = expanded_root_bbox + offset;
				
				int n_verts = 0;
				for (int i = 0; i < N_CHILDREN; i++) {
					if (test_box.contains(new_bbox.voxelvertex(i))) {
						n_verts++;
					}
				}

				if (n_verts > max_vertices) {
					best_sibling_number = c;
					max_vertices = n_verts;
				}
			}

			// Create new root
			Point_t offset = _root->bbox.voxelvertex(best_sibling_number) 
			               - expanded_root_bbox.voxelvertex(best_sibling_number);
			Box_t new_root_bbox = expanded_root_bbox + offset;

			Node_t* old_root = _root;
			_root = new Node_t(new_root_bbox, old_root->depth - 1);
			_divide(_root);
			
			delete _root->children[best_sibling_number];
			_root->children[best_sibling_number] = old_root;
			old_root->parent = _root;

			assert(_root->bbox.voxelvertex(best_sibling_number) == 
			       old_root->bbox.voxelvertex(best_sibling_number));
			assert(Box_t(_root->bbox.center(), _root->bbox.voxelvertex(best_sibling_number)) == 
			       old_root->bbox);

			// Recursively expand if needed
			if (max_vertices < N_CHILDREN) {
				_recursive_expand_bbox(new_bbox);
			}
		}

		/// Expand bbox until data fits
		void _recursive_resize_to_fit_data(const Data_t &val, int iter) {
			if (iter < 0) {
				throw std::runtime_error("Maximum recursion depth in resize_to_fit_data");
			}

			if (!isValid(_root->bbox, val)) {
				set_bbox(2.0 * _root->bbox);
				_recursive_resize_to_fit_data(val, iter - 1);
			}
		}

		/// Check if there is duplicated data
		void _recursive_duplicate_data(const Node_t* node) const {
			if (node==nullptr) {return;}

			if (isLeaf(node)) {
				if (node->data_idx==nullptr) {return;}
				for (int i = 0; i < node->cursor; i++) {
					size_t ii = node->data_idx[i];
					for (int j = i+1; j < node->cursor; j++) {
						size_t jj = node->data_idx[j];
						if (_data[ii] == _data[jj]) {
							if (ii == jj) {
								std::cout << "node contains duplicate index: " << ii << std::endl;
							} else {
								std::cout << "node contains duplicate data at indices: " << ii << " and " << jj << std::endl;
							}
						}
					}
				}
			} else {
				assert(node->data_idx==nullptr);
				for (int c = 0; c < N_CHILDREN; c++) {
					_recursive_duplicate_data(node->children[c]);
				}
			}
		}

		/// Push data down the tree (recursively)
		void _recursive_push_data_down(Node_t* node) {
			if (node==nullptr) {return;}
			if (isLeaf(node)) {return;}

			// Distribute data to children
			if (node->data_idx != nullptr) {
				std::unique_lock<std::shared_mutex> lock(node->_rw_mtx);
				
				for (int i = 0; i < node->cursor; i++) {
					size_t idx = node->data_idx[i];
					const Data_t &data = _data[idx];
					
					for (int c = 0; c < N_CHILDREN; c++) {
						if (isValid(node->children[c]->bbox, data)) {
							//check if the child has room
							if (node->children[c]->cursor == N_DATA) {
								//if the child does not have room, push its data down
								if (isLeaf(node->children[c])) {
									_divide(node->children[c]);
								}
								_recursive_push_data_down(node->children[c]);
							}

							[[maybe_unused]] int flag = appendDataIdx(node->children[c], idx);
							assert(flag==1);
							
							if constexpr (SINGLE_DATA) {
								break;  // Only first valid child gets the data
							}
						}
					}
				}

				// Delete current data
				clearDataIdx(node);
			}

			// Recurse into children
			for (int c = 0; c < N_CHILDREN; c++) {
				_recursive_push_data_down(node->children[c]);
			}
		}


		/// Divide a leaf node into children
		void _divide(Node_t* node) {
			assert(isLeaf(node));

			// Create children
			for (int c = 0; c < N_CHILDREN; c++) {
				assert(node->children[c] == nullptr);
				node->children[c] = new Node_t(node, c);
			}

			// Clear parent node
			node->is_leaf.store(false, std::memory_order_release);
		}

		/// Insert data into tree
		int _recursive_insert_data(Node_t* node, const Data_t &val, size_t idx) {
			assert(node != nullptr);
			assert(isValid(node->bbox, val));

			// Handle leaf node
			if (isLeaf(node)) {
				std::unique_lock<std::shared_mutex> lock(node->_rw_mtx);
				
				int flag = appendDataIdx(node, idx);
				if (flag >= 0) {
					return flag;
				}
				
				// Node full, divide it
				_divide(node);
			}

			assert(!isLeaf(node));

			// Insert into appropriate child(ren)
			int flag = 99;
			for (int c = 0; c < N_CHILDREN; c++) {
				assert(node->children[c] != nullptr);
				
				if (isValid(node->children[c]->bbox, val)) {
					flag = _recursive_insert_data(node->children[c], val, idx);
					
					if constexpr (SINGLE_DATA) {
						return flag;
					}
				}
			}

			assert(flag != 99);
			return flag;
		}


		/////////////////////////////////////////////////
		/// Convenience and debud methods
		/////////////////////////////////////////////////
		void _recursive_node_properties(const Node_t* node, size_t &n_nodes, size_t &n_idx, size_t &n_idx_cap, size_t &n_leafs, int &max_depth) const {
			if (node == nullptr) {return;}
			n_nodes++;
			
			if (isLeaf(node)) {n_leafs++;}

			if (node->data_idx != nullptr) {
				n_idx += node->cursor;
				n_idx_cap += N_DATA;
			}

			max_depth = std::max(max_depth, node->depth);

			for (int c = 0; c < OctreeParallelNode<DIM,N_DATA,T>::N_CHILDREN; c++) {
				_recursive_node_properties(node->children[c], n_nodes, n_idx, n_idx_cap, n_leafs, max_depth);
			}
	}
	};
}