#pragma once

#include <cassert>
#include <concepts>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#include "util/point.hpp"
#include "util/box.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// This file provides helper classes for the BasicParallelOctree class. Specifically,
/// it provides classes for the tree nodes and a thread-safe queue that multiple
/// threads can write to while the worker thread reads from it.
////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gv::util {

	/////////////////////////////////////////////////
	/// Octree node structure
	/////////////////////////////////////////////////
	template<int DIM=3, int N_DATA=16, Float T=float>
	struct OctreeParallelNode {
		static constexpr int N_CHILDREN = 1 << DIM;  // 2^DIM
		using Point_t = Point<DIM,T>;
		using Box_t   = Box<DIM,T>;
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;

		// Tree structure
		Node_t* parent = nullptr;
		Node_t* children[N_CHILDREN] {nullptr};
		int sibling_number = -1;  // parent->children[this->sibling_number] == this
		int depth = 0;
		const Box_t bbox;
		
		// Data indices stored in this node
		size_t* data_idx = nullptr;
		int cursor = 0;

		// Synchronization
		mutable std::shared_mutex _rw_mtx{};
		std::atomic<bool> is_leaf{true};

		// Create child node
		OctreeParallelNode(Node_t* parent, int sibling_number) 
			: parent(parent), 
			  sibling_number(sibling_number), 
			  depth(parent->depth + 1), 
			  bbox(parent->bbox.center(), parent->bbox.voxelvertex(sibling_number)) {}

		// Create root node with specified bounding box
		explicit OctreeParallelNode(const Box_t& bbox, int depth=0) 
			: depth(depth), bbox(bbox) {}

		// Destructor
		~OctreeParallelNode() {
			for (int c = 0; c < N_CHILDREN; c++) {
				delete children[c];
			}
			delete[] data_idx;
		}

		// Non-copyable, non-movable
		OctreeParallelNode(const OctreeParallelNode&) = delete;
		OctreeParallelNode& operator=(const OctreeParallelNode&) = delete;
		OctreeParallelNode(OctreeParallelNode&&) = delete;
		OctreeParallelNode& operator=(OctreeParallelNode&&) = delete;
	};

	/////////////////////////////////////////////////
	/// Node data management helpers
	/////////////////////////////////////////////////
	
	template<int DIM, int N_DATA, Float T>
	void resetDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		delete[] node->data_idx;
		node->data_idx = new size_t[N_DATA];
		node->cursor = 0;
	}

	template<int DIM, int N_DATA, Float T>
	void clearDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		delete[] node->data_idx;
		node->data_idx = nullptr;
		node->cursor = 0;
	}

	/// Append data index to node
	/// Returns: 1 if added, 0 if already present, -1 if no room
	template<int DIM, int N_DATA, Float T>
	int appendDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, size_t idx) {
		if (node->data_idx == nullptr) {
			resetDataIdx(node);
		}

		// Check if already present
		for (int i = 0; i < node->cursor; i++) {
			if (node->data_idx[i] == idx) {
				return 0;
			}
		}

		// Check capacity
		if (node->cursor >= N_DATA) {
			return -1;
		}

		// Add index
		node->data_idx[node->cursor] = idx;
		node->cursor++;
		return 1;
	}

	/// Remove data index from node
	template<int DIM, int N_DATA, Float T>
	void removeDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, size_t idx) {
		if (node->data_idx == nullptr) {
			return;
		}

		for (int i = 0; i < node->cursor; i++) {
			if (node->data_idx[i] == idx) {
				// Swap with last element and decrement cursor
				node->data_idx[i] = node->data_idx[node->cursor - 1];
				node->cursor--;
				return;
			}
		}
	}

	/// Check if node is a leaf (thread-safe)
	template<int DIM, int N_DATA, Float T>
	bool isLeaf(const OctreeParallelNode<DIM,N_DATA,T>* node) {
		return node->is_leaf.load(std::memory_order_acquire);
	}




	/////////////////////////////////////////////////
	/// Lock-free queue class for use in BasicParallelOctree
	/// Allows openmp threads to push while the worker thread is popping
	/////////////////////////////////////////////////
	template<typename DataBuffer>
	struct ThreadLocalQueue {
	    static constexpr size_t CAPACITY = 4096;
	    
	    DataBuffer queue[CAPACITY];
	    //thread A may only need head while thread B may only need tail.
	    //setting alignas(64) keeps one thread from loading both but only using one.
	    alignas(64) std::atomic<size_t> head{0};
	    alignas(64) std::atomic<size_t> tail{0}; 
	    std::atomic<size_t> pending_count{0};
	    size_t queue_id;
	    
	    bool push(DataBuffer&& item) {
	        size_t current_tail = tail.load(std::memory_order_relaxed);
	        size_t next_tail = (current_tail + 1) % CAPACITY;
	        
	        if (next_tail == head.load(std::memory_order_acquire)) {
	            return false;  // Queue full
	        }
	        
	        queue[current_tail] = std::move(item);
	        tail.store(next_tail, std::memory_order_release);
	        return true;
	    }
	    
	    bool pop(DataBuffer& item) {
	        size_t current_head = head.load(std::memory_order_relaxed);
	        
	        if (current_head == tail.load(std::memory_order_acquire)) {
	            return false;  // Queue empty
	        }
	        
	        item = queue[current_head];
	        head.store((current_head + 1) % CAPACITY, std::memory_order_release);
	        return true;
	    }
	    
	    bool empty() const {
	        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
	    }
	};
}