#pragma once

#include <cassert>
#include <concepts>
#include <mutex>
#include <shared_mutex>
#include <atomic>

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Single producer single consumer queue
////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gv::util {
	/////////////////////////////////////////////////
	/// A queue data structure for use in multithreaded applications
	/////////////////////////////////////////////////

	//TODO: CAPACITY is needed this large for high levels of refinement, but it's too much. Think about linked list or re-size and copy.
	//making more inserter threads could work.
	template<typename Data_t, size_t CAPACITY=4096>
	struct ThreadLocalQueue {
	    //thread A may only need head while thread B may only need tail.
	    //setting alignas(64) keeps one thread from loading both but only using one.
	    alignas(64) std::atomic<size_t> head{0};
	    alignas(64) std::atomic<size_t> tail{0};
	    std::atomic<size_t> count{0};
	    size_t queue_id;

	    Data_t queue[CAPACITY]; //store heap allocated so that it can be resized
	    std::atomic<size_t> buffer_bumps{0};

	    bool try_push(Data_t&& item) {
	        size_t current_tail = tail.load(std::memory_order_relaxed);
	        size_t next_tail = (current_tail + 1) % CAPACITY;
	        
	        if (next_tail == head.load()) {
	        	buffer_bumps++;
	            return false;  // Queue full
	        }
	        
	        queue[current_tail] = std::move(item);
	        tail.store(next_tail);
	        count++;
	        return true;
	    }
	    
	    bool try_pop(Data_t& item) {
	        size_t current_head = head.load(std::memory_order_relaxed);
	        
	        if (current_head == tail.load()) {
	            return false;  // Queue empty
	        }
	        
	        item = queue[current_head];
	        head.store((current_head + 1) % CAPACITY);
	        count--;
	        return true;
	    }
	    
	    bool empty() const {
	        return head.load() == tail.load();
	    }
	};
}