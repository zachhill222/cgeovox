#pragma once

//specializes the BasicOctree class into two classes.
//BasicOctree_Point is used to store data which is colocated at a single point
//BasicOctree_Vol is used to store data which encloses a volume
//each of these classes must be further specialized to the data type that is to be stored


#include "util/point.hpp"
#include "util/box.hpp"
// #include "compile_constants.hpp" //max octree depth
#include "concepts.hpp"
#include "basic_octree.hpp"

#include <cassert>
#include <vector>

#include <mutex>
#include <shared_mutex>

namespace gv::util
{
	//OCTREE FOR POINT DATA TYPES
	//store data in the first available node
	template<typename Data_t, int dim=3, int n_data=64, Float T=double>
	class BasicOctree_Point : public BasicOctree<Data_t,dim,n_data,T>
	{
	public:
		using Parent_t = BasicOctree<Data_t,dim,n_data,T>;
		using Point_t  = typename Parent_t::Point_t;
		using Box_t    = typename Parent_t::Box_t;
		using Node_t   = typename Parent_t::Node_t;

		//constructor if the bounding box and possibly capacity are unknown
		BasicOctree_Point(const size_t capacity=64) : BasicOctree<Data_t,dim,n_data,T>(capacity) {};

		//use this constructor if possible. the bounding box should be known ahead of time to avoid re-creating the octree structure.
		BasicOctree_Point(const Box_t &bbox, const size_t capacity=64) : BasicOctree<Data_t,dim,n_data,T>(bbox, capacity) {}

		//copy constructor
		BasicOctree_Point(const BasicOctree_Point &other) : BasicOctree<Data_t,dim,n_data,T>(other) {}

		virtual ~BasicOctree_Point() = default;

		protected:
			virtual bool is_data_valid(const Box_t &bbox, const Data_t &val) const = 0;  //check if data can be placed into a node with the specified bounding box
			// size_t recursive_find(Node_t* const node, const Data_t &val) const override; //primary method for finding data
			bool recursive_insert(Node_t* const node, const Data_t &val, const size_t idx) override; //primary method for inserting data into octree structure
			// const Node_t* recursive_get_node(const Node_t* const node, const Point_t &coord) const override; //primary method for accessing nodes associated with a coordinate
			// void recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const override; //primary method for getting leaf nodes intersecting a region
			void divide(Node_t* const node) override; //divide a node into its children nodes
	};

	template<typename Data_t, int dim, int n_data, Float T>
	bool BasicOctree_Point<Data_t,dim,n_data,T>::recursive_insert(Node_t* const node, const Data_t &val, const size_t idx) {
		//the data must be valid in this node now
		assert(is_data_valid(node->bbox,val));


		//in valid node. store index to data if there is room
		if (node->data_cursor < n_data)	{
			append_index(node,idx);
			return true;
		}

		//recurse into first valid child
		if (is_divided(node)) {
			for (int c_idx=0; c_idx<Parent_t::n_children; c_idx++) {
				Node_t* child = node->children[c_idx];

				//attempt to put data into first valid child
				if (is_data_valid(child->bbox, val)) {
					std::unique_lock<std::shared_mutex> lock(child->_rw_mutex);
					return recursive_insert(child, val, idx);
				}
			}

			//the insertion was not successful
			return false;
		}

		//in valid leaf node, but it must divide
		assert(node->depth - this->_root->depth<gv::constants::OCTREE_MAX_DEPTH);
		if (node->depth - this->_root->depth>=gv::constants::OCTREE_MAX_DEPTH) {return false;} //could not insert data

		divide(node); //moves data, frees some memory, creates children. checks for a maximum tree depth.
		return recursive_insert(node, val, idx); //re-run insertion here now that it is divided
		return false;
	}

	

	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree_Point<Data_t,dim,n_data,T>::divide(Node_t* const node) {
		// std::lock_guard<std::mutex> lock(node->_mtx);

		//it is assumed that the node is not divided already and that we will not violate the maximum tree depth
		assert(!is_divided(node));
		assert(node->depth - this->_root->depth<gv::constants::OCTREE_MAX_DEPTH);

		//create children (do not copy data)
		for (int c_idx=0; c_idx<Parent_t::n_children; c_idx++) {
			Node_t* child = new Node_t(node, c_idx); //create child
			node->children[c_idx] = child; //use child as a convenient reference locally
		}
	}






	//OCTREE FOR VOLUME DATA TYPES
	//store data in every valid leaf node, but not in any other nodes

	template<typename Data_t, int dim=3, int n_data=8, Float T=double>
	class BasicOctree_Vol : public BasicOctree<Data_t,dim,n_data,T>
	{
	public:
		using Parent_t = BasicOctree<Data_t,dim,n_data,T>;
		using Point_t  = typename Parent_t::Point_t;
		using Box_t    = typename Parent_t::Box_t;
		using Node_t   = typename Parent_t::Node_t;

		//constructor if the bounding box and possibly capacity are unknown
		BasicOctree_Vol(const size_t capacity=64) : BasicOctree<Data_t,dim,n_data,T>(capacity) {};

		//use this constructor if possible. the bounding box should be known ahead of time to avoid re-creating the octree structure.
		BasicOctree_Vol(const Box_t &bbox, const size_t capacity=64) : BasicOctree<Data_t,dim,n_data,T>(bbox, capacity) {}

		//copy constructor
		BasicOctree_Vol(const BasicOctree_Vol &other) : BasicOctree<Data_t,dim,n_data,T>(other) {}

		virtual ~BasicOctree_Vol() = default;

		protected:
			virtual bool is_data_valid(const Box_t &bbox, const Data_t &val) const = 0;  //check if data can be placed into a node with the specified bounding box
			// size_t recursive_find(Node_t* const node, const Data_t &val) const override; //primary method for finding data
			bool recursive_insert(Node_t* const node, const Data_t &val, const size_t idx) override; //primary method for inserting data into octree structure
			// const Node_t* recursive_get_node(const Node_t* const node, const Point_t &coord) const override; //primary method for accessing nodes associated with a coordinate
			// void recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const override; //primary method for getting leaf nodes intersecting a region
			void divide(Node_t* const node) override; //divide a node into its children nodes
	};

	

	template<typename Data_t, int dim, int n_data, Float T>
	bool BasicOctree_Vol<Data_t,dim,n_data,T>::recursive_insert(Node_t* const node, const Data_t &val, const size_t idx)
	{
		//insert data into ALL leaf nodes that are valid
		//it is assumed that is_data_valid(node,val) is true
		assert(is_data_valid(node->bbox,val));

		//recurse into all valid children
		if (is_divided(node)) {
			bool success = true;
			for (int c_idx=0; c_idx<Parent_t::n_children; c_idx++) {
				Node_t* child = node->children[c_idx];

				//attempt to put data into all valid children
				if (is_data_valid(child->bbox, val)) {
					success = success and recursive_insert(child, val, idx); //data must be successfully added to all leaves
				}
			}
			return success;
		}

		//in valid leaf node. store index to data if there is room
		if (node->data_cursor < n_data)	{
			append_index(node,idx); return true;
		}

		//in valid leaf node, but it must divide
		assert(node->depth - this->_root->depth < gv::constants::OCTREE_MAX_DEPTH);
		if (node->depth - this->_root->depth >= gv::constants::OCTREE_MAX_DEPTH) {return false;} //could not insert data
		
		divide(node); //moves data, frees some memory, creates children. checks for a maximum tree depth.
		return recursive_insert(node, val, idx); //re-run insertion here now that it is divided
	}

	
	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree_Vol<Data_t,dim,n_data,T>::divide(Node_t* const node) {
		//it is assumed that the node is not divided already and that we will not violate the maximum tree depth
		assert(!is_divided(node));
		assert(node->depth - this->_root->depth<gv::constants::OCTREE_MAX_DEPTH);

		//create children and copy valid data into children
		for (int c_idx=0; c_idx<Parent_t::n_children; c_idx++) {
			Node_t* child = new Node_t(node, c_idx); //create child
			node->children[c_idx] = child; //use child as a convenient reference locally
			for (int d_idx=0; d_idx<node->data_cursor; d_idx++) {
				//initialize data in child
				size_t idx = node->data_idx[d_idx];
				if (is_data_valid(child->bbox, Parent_t::_data[idx])) {
					append_index(child, idx);
				}
			}
		}

		//free memory at current node
		node->data_cursor = 0; //no memory is free, but the node will not "see" any data
		// if (node->data_idx!=nullptr)
		// {
		// 	delete[] node->data_idx;
		// 	node->data_idx=nullptr; //mark that this has been deleted to avoid double frees
		// 	node->data_cursor = 0;
		// }
	}
}