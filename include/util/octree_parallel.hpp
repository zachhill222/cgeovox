#pragma once

//debug
#include <cassert>
#include <concepts>

//octree structure
#include "util/point.hpp"
#include "util/box.hpp"

//storage
#include <vector>
#include <array>
#include <queue>

#include <omp.h>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

namespace gv::util {
	/////////////////////////////////////////////////
	/// Structure for OctreeParallelNodes
	/////////////////////////////////////////////////
	template<int DIM=3, int N_DATA=16, Float T=float>
	struct OctreeParallelNode {
		static constexpr int N_CHILDREN=std::pow(2,DIM);
		using Point_t = Point<DIM,T>;
		using Box_t   = Box<DIM,T>;
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;

		//tree structure
		Node_t* parent = nullptr;
		Node_t* children[N_CHILDREN] {nullptr};
		int sibling_number = -1; //parent->children[this->sibling_number] == this
		int depth = 0;
		const Box_t bbox;
		
		//data indices
		size_t* data_idx = nullptr;
		int cursor=0;

		//read write lock
		mutable std::shared_mutex _rw_mtx;
		std::atomic<bool> is_leaf{true};

		//create child of specified node and initialize storage
		OctreeParallelNode(Node_t* parent, int sibling_number) : 
			parent(parent),
			sibling_number(sibling_number),
			depth(parent->depth+1),
			bbox(parent->bbox.center(), parent->bbox.voxelvertex(sibling_number)) {}

		//create node with specified bounding box and do not initialize storage
		OctreeParallelNode(const Box_t& bbox, const int depth=0) : depth(depth), bbox(bbox) {}

		//destructor
		~OctreeParallelNode() {
			for (int c=0; c<N_CHILDREN; c++) {
				delete children[c];
				children[c] = nullptr;
			}
			delete[] data_idx;
		}
	};

	/////////////////////////////////////////////////
	/// Initialize data index storage in an OctreeParallelNode
	/////////////////////////////////////////////////
	template<int DIM, int N_DATA, Float T>
	void resetDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		delete[] node->data_idx;
		node->data_idx = new size_t[N_DATA];
		node->cursor = 0;
	}

	/////////////////////////////////////////////////
	/// Clear data index storage in an OctreeParallelNode
	/////////////////////////////////////////////////
	template<int DIM, int N_DATA, Float T>
	void clearDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		delete[] node->data_idx;
		node->data_idx = nullptr;
		node->cursor = 0;
	}

	/////////////////////////////////////////////////
	/// Append data to data_idx if it is not already contained
	///
	/// Return 1 if the data was successfully added.
	/// Return 0 if the data was already contained.
	/// Return -1 if there was no room.
	/////////////////////////////////////////////////
	template<int DIM, int N_DATA, Float T>
	int appendDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, const size_t idx) {
		if(node->data_idx==nullptr) {
			resetDataIdx(node);
		}

		for (int i=0; i<node->cursor; i++) {
			if (node->data_idx[i] == idx) {return 0;}
		}

		if (node->cursor >= N_DATA) {return -1;}
		node->data_idx[node->cursor] = idx;
		node->cursor++;
		return 1;
	}

	/////////////////////////////////////////////////
	/// Remove specified index from data_idx
	/////////////////////////////////////////////////
	template<int DIM, int N_DATA, Float T>
	void removeDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, const size_t idx) {
		if(node->data_idx == nullptr) {return;};

		for (int i=0; i<node->cursor; i++) {
			if (node->data_idx[i] == idx) {
				node->data_idx[i] = node->data_idx[node->cursor - 1];
				node->cursor--;
			}
		}
	}


	/////////////////////////////////////////////////
	/// Check if a node is a leaf
	/////////////////////////////////////////////////
	template<int DIM, int N_DATA, Float T>
	bool isLeaf(const OctreeParallelNode<DIM,N_DATA,T>* node) {
		return node->is_leaf.load(std::memory_order_acquire);
	}


	/////////////////////////////////////////////////
	/// BasicParallelOctree class to store most of the logic
	///
	/// @tparam Data_t      The type of data to be stored
	/// @tparam SINGLE_DATA A flag to set how the nodes store the data.
	///							When set to false, every leaf that the data overlaps will store an index to the data
	/// 						When set to true, only the first valid leaf will store an index to the data
	/// 						If Data_t encompases a region (say it represents a sphere), then this flag should most likely be false.
	/// 						If Data_t is located at a point, then this flag should most likely be true.
	/// @tparam DIM         The dimension of the octree (usually DIM=3)
	/// @tparam N_DATA      How many data points each leaf is allowed to store an index for. If operator==(Data_t A, Data_t B) is cheap, this can be larger.
	/// @tparam T           The precision for bounding boxes.
	/////////////////////////////////////////////////
	template<typename Data_t, bool SINGLE_DATA, int DIM=3, int N_DATA=16, Float T=float>
	class BasicParallelOctree {
		static_assert(DIM>0);
		static_assert(N_DATA>0);
	public:
		//aliases
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;
		using Point_t = typename Node_t::Point_t;
		using Box_t   = typename Node_t::Box_t;
		
		//common known values
		static constexpr int N_CHILDREN  = Node_t::N_CHILDREN;

	protected:
		//containers for parallel buffer/write
		struct DataBuffer {
			size_t  idx;
			Node_t* target_node;
		};

		struct ThreadLocalQueue {
			std::queue<DataBuffer> queue;
			std::atomic<size_t> pending_count{0};
			size_t queue_id;
		};


		//tree root
		Node_t* _root = nullptr;

		//data storage
		std::vector<Data_t> _data;
		std::atomic<size_t> _next_data_idx{0}; //track the index of the next data to store into _data
		
		//dedicated thread for updating the tree
		std::thread _inserter_thread;
		std::condition_variable _inserter_cv;
		mutable std::mutex _inserter_mtx;
		std::atomic<bool> _running{true};
		std::atomic<size_t> _total_pending{0};
		std::vector<ThreadLocalQueue*> _all_queues; //one buffer queue per thread

	public:
		BasicParallelOctree(const Box_t &bbox) : _root(new Node_t(bbox)) {
			resetDataIdx(_root);

			//set up thread buffering
			#ifdef _OPENMP
				int max_threads = omp_get_max_threads();
			#else
				int max_threads = 1;  // Single thread if no OpenMP
			#endif
			_all_queues.resize(max_threads, nullptr);
			
			for (int i=0; i<max_threads; i++) {
				_all_queues[i] = new ThreadLocalQueue();
				_all_queues[i]->queue_id = i;
			}			

			//start inserter thread
			_inserter_thread = std::thread([this]() {
				_inserter_loop();
			});
		}
		virtual ~BasicParallelOctree() {
			//stop threads
			_running.store(false);
			_inserter_cv.notify_one();

			if (_inserter_thread.joinable()) {
				_inserter_thread.join();
			}

			//delete queues
			for (auto* queue : _all_queues) {
				delete queue;
			}

			//delete the tree structure
			delete _root;
		}

		/////////////////////////////////////////////////
		/// Wrapper for std::vector
		/////////////////////////////////////////////////
		inline void reserve(const size_t length) {
			_data.reserve(length);
		}
		inline void shrink_to_fit() {
			_data.resize(_next_data_idx);
			_data.shrink_to_fit();
		}
		inline bool empty() const {
			return _data.empty();
		}
		inline size_t size() const {
			// return _data.size();
			return _next_data_idx.load(std::memory_order_acquire);
		}
		inline size_t capacity() const {
			return _data.capacity();
		}
		
		size_t push_back(const Data_t &val) {
			Data_t copy(val);
			return push_back(std::move(copy));
		}

		size_t push_back(Data_t &&val) {
			//serial push_back
			Node_t* start_node = nullptr;
			//try to find the data
			start_node = _recursive_find_best_node(_root, val);
			size_t idx = _recursive_find_index<true>(start_node, val);
			if (idx!=(size_t)-1) {
				assert(_data[idx]==val);
				return idx;
			}
			

			//insert the data
			idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
			if (_data.size()<=idx) {_data.resize(idx+1);}
			_data[idx] = std::move(val);

			[[maybe_unused]] int flag = _recursive_insert_data(start_node, _data[idx], idx);
			assert(flag==1);
			return idx;
		}

		size_t push_back_async(Data_t &&val) {
			#ifndef _OPENMP
				push_back(std::move(val));
			#endif

			//attempt to find the data
			Node_t* start_node = _recursive_find_best_node(_root, val);
			size_t  idx        = _recursive_find_index<false>(start_node, val);
			if (idx != (size_t) -1) {return idx;}

			//insert the data into storage
			//room must be allocated ahead of time
			idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
			_data[idx] = std::move(val);

			//mark the data for insertion into the tree
			#ifdef _OPENMP
				int thread_no = omp_get_thread_num();
			#else
				int thread_no = 0;
			#endif
			ThreadLocalQueue* thread_queue = _all_queues[thread_no];
			thread_queue->queue.push({idx, start_node});
			thread_queue->pending_count.fetch_add(1);

			//alert inserter thread
			_total_pending.fetch_add(1);
			_inserter_cv.notify_one();

			return idx;
		}

		void flush() {
			_inserter_cv.notify_one();
		    while (_total_pending.load(std::memory_order_acquire) > 0) {
		        // std::cout << "FLUSHING " << _total_pending << " elements" << std::endl;
		        std::this_thread::yield();
		    }
		}

		
		void clear() {
			//clear data and buffer
			_data.clear();
			//re-set tree structure
			Node_t* new_root = new Node_t(_root->bbox);
			resetDataIdx(new_root);
			delete _root;
			_root = new_root;
		}
		
		
		void resize(const size_t length) {
			//if the size is decreasing, remove any references from the octree
			if (_root!=nullptr) {
				for (size_t i=length; i<_data.size(); i++) {
					_recursive_remove_idx(_root, i, _data[i]);
				}
			}

			_data.resize(length);
		}

		inline std::vector<Data_t>::iterator       begin()        {return _data.begin();}
		inline std::vector<Data_t>::const_iterator begin()  const {return _data.cbegin();}
		inline std::vector<Data_t>::const_iterator cbegin() const {return _data.cbegin();}
		inline std::vector<Data_t>::iterator       end()          {return _data.end();}
		inline std::vector<Data_t>::const_iterator end()    const {return _data.cend();}
		inline std::vector<Data_t>::const_iterator cend()   const {return _data.cend();}


		/////////////////////////////////////////////////
		/// Common operations
		/////////////////////////////////////////////////
		const Box_t& bbox() const {return _root->bbox;}
		void set_bbox(const Box_t& bbox) {
			_recursive_expand_bbox(bbox);

			//if the data might be inserted into multiple nodes,
			//there may be new nodes that need the reference to existing data
			if constexpr (!SINGLE_DATA) {
				for (size_t j=0; j<_data.size(); j++) {
					_recursive_insert_data(_root, _data[j], j);
				}
			}
		}

		const Data_t& operator[](const size_t idx) const {
			assert(idx<size());
			return _data[idx];
		}

		Data_t& operator[](const size_t idx) {
			assert(idx<size());
			return _data[idx];
		}

		size_t find(const Data_t& val) const {
			return _recursive_find_index<true>(const_cast<const Node_t*>(_root), val);
		}

		void reinsert(const size_t idx) {
			//only call while a single thread is active
			assert(idx<_data.size());
			_recursive_remove_idx(_root, idx, _data[idx]);

			if (!isValid(_root->bbox, _data[idx])) {
				_recursive_resize_to_fit_data(_data[idx], 8);
				if constexpr (!SINGLE_DATA) {
					for (size_t j=0; j<_data.size(); j++) {
						_recursive_insert_data(_root, _data[j], j);
					}
				}
			} else {
				_recursive_insert_data(_root, _data[idx], idx);
			}
		}

		

	private:
		void _inserter_loop() {
			while (_running.load()) {
				{
					std::unique_lock<std::mutex> lock(_inserter_mtx);
					_inserter_cv.wait(lock);
				}

				while (_total_pending.load()>0) {
					for (auto* thread_queue : _all_queues) {
		                if (!thread_queue) continue;
		                
		                while (!thread_queue->queue.empty()) {
							DataBuffer work = thread_queue->queue.front();
							thread_queue->queue.pop();
							thread_queue->pending_count.fetch_sub(1);
							_total_pending.fetch_sub(1);

							size_t index = _recursive_find_index<true>(work.target_node, _data[work.idx]);
							if (index==(size_t) -1) {
								[[maybe_unused]] int flag = _recursive_insert_data(work.target_node, _data[work.idx], work.idx);
								assert(flag == 1);
							}
						}
					}
				}
			}
		}


		/////////////////////////////////////////////////
		/// Determine if a piece of data belongs to a box.
		/// Must be implemented for each Data_t
		/////////////////////////////////////////////////
		virtual bool isValid(const Box_t &bbox, const Data_t &val) const = 0;

		/////////////////////////////////////////////////
		/// Remove any reference to idx from the specified node and all descendents.
		///
		/// @param node The node to start the recursion from.
		///             The UNIQUE lock on this node should be engaged before calling.
		/// @param idx  The index to remove
		/////////////////////////////////////////////////
		void _recursive_remove_idx(Node_t* node, const size_t idx) {
			if (isLeaf(node)) {removeDataIdx(node, idx);}
			else {
				for (int c=0; c<N_CHILDREN; c++) {
					_recursive_remove_idx(node->children[c], idx);
				}
			}
		}


		/////////////////////////////////////////////////
		/// Remove any reference to idx from the specified node and all descendents.
		/// More efficient as the recursion will only go into children where the data is valid.
		///
		/// @param node The node to start the recursion from.
		///             The UNIQUE lock on this node should be engaged before calling.
		/// @param idx  The index to remove
		/// @param val  The data that correspond(ed) to _data[idx]
		/////////////////////////////////////////////////
		void _recursive_remove_idx(Node_t* node, const size_t idx, const Data_t &val) {
			if (isLeaf(node)) {
				removeDataIdx(node, idx);}
			else {
				for (int c=0; c<N_CHILDREN; c++) {
					if (isValid(node->children[c]->bbox, val)) {
						_recursive_remove_idx(node->children[c], idx, val);
					}
				}
			}
		}

		
		/////////////////////////////////////////////////
		/// Find the most likely node to contain the given data.
		/// If SINGLE_DATA is false, then this can be used to find data,
		/// but inserting data at the returned node is dangerous.
		/// 
		/// @param node The node to start the recursion from.
		/////////////////////////////////////////////////
		Node_t* _recursive_find_best_node(Node_t* node, const Data_t &val) {
			if (isLeaf(node)) {
				if (node->parent!=nullptr) {return node->parent;}
				else {return node;}
			} else {
				for (int c=0; c<N_CHILDREN; c++) {
					if (isValid(node->children[c]->bbox, val)) {
						return _recursive_find_best_node(node->children[c], val);
					}
				}
			}

			//this node has children, but the data is not valid in any of them
			return _root;
		}


		/////////////////////////////////////////////////
		/// Find the index of the specified data.
		///
		/// @param node The node to start the recursion from.
		/// @param val  The data to find.
		///
		/// @tparam UNLOCKED run without a mutex
		/////////////////////////////////////////////////
		template<bool UNLOCKED>
		size_t _recursive_find_index(const Node_t* node, const Data_t &val) const {
			if (isLeaf(node)) {
				if constexpr (UNLOCKED) {
					for (int i=0; i<node->cursor; i++) {
						const size_t IDX = node->data_idx[i];
						if (_data[IDX] == val) {return IDX;}
					}
				} else {
					std::shared_lock<std::shared_mutex> lock(node->_rw_mtx);
					for (int i=0; i<node->cursor; i++) {
						const size_t IDX = node->data_idx[i];
						if (_data[IDX] == val) {return IDX;}
					}
				}

				
			} else {
				assert(node->data_idx==nullptr);
				assert(node->cursor==0);
				for (int c=0; c<N_CHILDREN; c++) {
					if (isValid(node->children[c]->bbox, val)) {
						const size_t IDX = _recursive_find_index<UNLOCKED>(node->children[c], val);
						if (IDX != (size_t) -1) {return IDX;}
					}
				}
			}

			return (size_t) -1;
		}



		/////////////////////////////////////////////////
		/// Expand the bounding box for the octree until it encompases the desired region.
		/// If _root->bbox already encompases the desired region, then nothing happens.
		///
		/// This constructs a valid new_root node such that the current _root is a valid child
		/// (i.e., it could have been constructed as _root = new Node_t(new_root, c)).
		/// This is done recursively until new_root->bbox encompases the desired region.
		///
		/// It is assumed that no other threads will try to read from or write to the octree 
		/// until this method returns.
		///
		/// @param new_bbox The desired region to encompass.
		/////////////////////////////////////////////////
		void _recursive_expand_bbox(const Box_t& new_bbox) {
			if (_root->bbox.contains(new_bbox)) {return;}


			//double the bounding box and find the best shift
			Box_t expanded_root_bbox = 2*_root->bbox;
			int max_vertices=-1;
			int best_sibling_number=-1;
			for (int c=0; c<N_CHILDREN; c++) {
				Point_t offset = _root->bbox.voxelvertex(c) - expanded_root_bbox.voxelvertex(c);
				Box_t test_box = expanded_root_bbox + offset;
				int n_verts = 0;
				for (int i=0; i<N_CHILDREN; i++) {
					if (test_box.contains(new_bbox.voxelvertex(i))) {n_verts++;}
				}

				if (n_verts>max_vertices) {
					best_sibling_number = c;
					max_vertices = n_verts;
				}
			}

			Point_t offset = _root->bbox.voxelvertex(best_sibling_number) - expanded_root_bbox.voxelvertex(best_sibling_number);
			Box_t   new_root_bbox = expanded_root_bbox + offset;


			//make the new root node and update the tree structure/pointers
			Node_t* old_root = _root;
			_root            = new Node_t(new_root_bbox, old_root->depth-1);
			_divide(_root);
			delete _root->children[best_sibling_number];
			_root->children[best_sibling_number] = old_root;
			old_root->parent = _root;

			assert(_root->bbox.voxelvertex(best_sibling_number) == old_root->bbox.voxelvertex(best_sibling_number));
			assert(Box_t(_root->bbox.center(), _root->bbox.voxelvertex(best_sibling_number)) == old_root->bbox);


			//call again if we still need to expand
			if (max_vertices < N_CHILDREN) {
				_recursive_expand_bbox(new_bbox);
			}
		}


		/////////////////////////////////////////////////
		/// Expand the bounding box until the _root->bbox fits the data.
		///
		/// It is assumed that no other threads will try to read from or write to the octree 
		/// until this method returns.
		///
		/// @param val The desired data to encompass.
		/////////////////////////////////////////////////
		void _recursive_resize_to_fit_data(const Data_t &val, const int iter) {
			if (iter<0) {
				throw std::runtime_error("maximum recursion in resize_to_fit_data");
			}

			if (!isValid(_root->bbox, val)) {
				set_bbox(2*_root->bbox);
				_recursive_resize_to_fit_data(val, iter-1);
			}
		}


		/////////////////////////////////////////////////
		/// Divide a node. Its data will be pushed into the appropriate child node(s).
		/// All children are created, but only data index storage will be initialized
		/// on children where the data from node will be transfered to.
		///
		/// @param node The node to divide. The UNIQUE lock on the specified node must be engaged before 
		///                calling this method.
		/////////////////////////////////////////////////
		void _divide(Node_t* node) {
			assert(isLeaf(node));

			//create children
			for (int c=0; c<N_CHILDREN; c++) {
				assert(node->children[c]==nullptr);
				node->children[c] = new Node_t(node,c);
			}

			//pass data
			for (int i=0; i<node->cursor; i++) {
				const size_t IDX   = node->data_idx[i];
				const Data_t &DATA = _data[IDX];
				for (int c=0; c<N_CHILDREN; c++) {
					if (isValid(node->children[c]->bbox, DATA)) {
						[[maybe_unused]] int flag = appendDataIdx(node->children[c], IDX);
						assert(flag>=0);
						if constexpr (SINGLE_DATA) {break;} //only the first valid child gets the data
					}
				}
			}

			//clear the data from this node
			clearDataIdx(node);
			node->is_leaf.store(false, std::memory_order_release);
		}


		/////////////////////////////////////////////////
		/// Insert data into the octree starting at the specified node.
		/// This does not change _data. It only updates the tree structure.
		/// It is assumed that val==_data[idx] after this method returns.
		/// 
		///
		/// @param node The node to start the recursion from.
		///             The UNIQUE lock on this node should be engaged before calling.
		/// @param val  The data that is being inserted into the tree.
		/// @param idx  The index of the data that is being inserted into the tree.
		/////////////////////////////////////////////////
		int _recursive_insert_data(Node_t* node, const Data_t &val, const size_t idx) {
			assert(node!=nullptr);
			assert(isValid(node->bbox, val));

			//process if this is a leaf node
			if (isLeaf(node)) {
				std::unique_lock<std::shared_mutex> lock(node->_rw_mtx); //only one thread will alter the tree structure and write data
				// this keeps other threads from attempting to read whil this node is being altered
				//attempt to add data
				int flag = appendDataIdx(node, idx);
				if (flag>=0) {return flag;}
				_divide(node);
			}

			assert(node->data_idx==nullptr);
			assert(!isLeaf(node));

			int flag=99;
			for (int c=0; c<N_CHILDREN; c++) {
				assert(node->children[c]!=nullptr);
				if (isValid(node->children[c]->bbox, val)) {
					flag = _recursive_insert_data(node->children[c], val, idx);
					if constexpr (SINGLE_DATA) {return flag;}
				}
			}

			assert(flag!=99);
			return flag;
		}
	};


}
