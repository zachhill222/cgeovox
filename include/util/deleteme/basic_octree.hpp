#pragma once

//standard octree implementation for organizing data in 2D and 3D space
//each data type (Data_t) will require an octree container that inherits from the BasicOctree class (e.g., <Data_t>Octree)
//each user implemented octree container will require a method with the signature: bool is_data_valid(const Box_t &bbox, const Data_t &val) for data insertion
//if Data_t encloses a volume (e.g. a spherical particle), then is_data_valid(bbox,val) should test for intersection of bbox and val.
//the data type must implement an equality comparison (e.g., bool operator==(const Data_t& other) const) only unique data (by this comparison) will be stored in the container
//to be able to effectively re-size the array, Data_t must have a move assignment operator (std::move() must be callable)

//TODO:
//	-implement data deletion
//	-implement data sorting

#include "util/point.hpp"
#include "util/box.hpp"

#include "compile_constants.hpp" //max octree depth
#include "concepts.hpp"

#include <cassert>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include <omp.h>

namespace gv::util
{
	//class for the nodes of the octree
	//this class stores indices to a primary vector where the data is stored
	//data indices are only stored in leaf nodes and a single index may be stored in multiple nodes
	//this is useful when Data_t encloses a region rather than being located at a single point and mostly irrelevant if the data is located at a point

	template<typename Data_t, int dim=3, int n_data=8, Float T=double>
	class OctreeNode {
	public:
		//common typedefs and constants
		static constexpr int _dim = dim;
		static constexpr int n_children = std::pow(2,dim);
		using Node_t  = OctreeNode<Data_t,dim,n_data,T>;
		using Point_t = gv::util::Point<dim,T>;
		using Box_t   = gv::util::Box<dim,T>;

		//constructor for initializing _root node
		OctreeNode(const Box_t& bbox, const int depth=0) :
			// data_idx(new size_t[n_data]), 
			data_cursor(0),
			bbox(bbox), 
			parent(nullptr), 
			sibling_number(-1),
			depth(depth) {}

		//constructor for initializing children nodes
		OctreeNode(Node_t* const parent, const int sibling_number) :
			// data_idx(new size_t[n_data]),
			data_cursor(0), 
			bbox(parent->bbox.voxelvertex(sibling_number), parent->bbox.center()), 
			parent(parent), 
			sibling_number(sibling_number),
			depth(parent->depth+1) {}

		//destructor
		~OctreeNode() {
			for (int c_idx=0; c_idx<n_children; c_idx++) {delete children[c_idx];}
		}

		//data storage
		// size_t* data_idx; //only store the pointer here so that it can be deleted when a node is divided
		size_t data_idx[n_data];
		int  data_cursor = 0; //used to track where the next data index should be written (e.g., data_idx[data_cursor])

		//geometry
		const Box_t bbox; //portion of space that this node is responsible for

		//tree information
		Node_t* parent;
		Node_t* children[n_children] {nullptr};
		int sibling_number; //parent->children[this->sibling_number] == this
		mutable std::shared_mutex _rw_mutex; //lock when inserting data from this node
		const int depth;
	};

	//helpful functions to call on nodes
	template<typename Data_t, int dim, int n_data, Float T>
	bool is_divided(const OctreeNode<Data_t,dim,n_data,T>* const node) {return node->children[0]!=nullptr;}

	template<typename Data_t, int dim, int n_data, Float T>
	void append_index(OctreeNode<Data_t,dim,n_data,T>* const node, const int idx) {
		//it is assumed that there is room
		assert(node->data_cursor<n_data);
		node->data_idx[node->data_cursor] = idx;
		node->data_cursor += 1;
	}

	// if this this node contains data_idx[k] == idx for some k, then that index is swapped with the last index and the cursor decreased by 1
	template<typename Data_t, int dim, int n_data, Float T>
	void remove_idx(OctreeNode<Data_t,dim,n_data,T>* const node, const size_t idx) {
		for (int i=0; i<node->data_cursor; i++) {
			if (node->data_idx[i] == idx) {
				node->data_cursor -= 1; //point data cursor to the last data written to this node
				// std::swap(node->data_idx[i], node->data_idx[node->data_cursor]); //swap the specified index with the last valid data
				const size_t last = node->data_idx[node->data_cursor];
				node->data_idx[i] = last;
				//the next index to be written on this node will overwrite idx, which is now at the next location of data_idx to be written to
				return;
			}
		}
	}

	

	//octree container. handles division of nodes, insertion and retrieval of data, storage of data, etc.
	template<typename Data_t, int dim=3, int n_data=8, Float T=double>
	class BasicOctree
	{
	public:
		//common typedefs and constants
		static constexpr int _dim = dim;
		static constexpr int n_children = std::pow(2,dim);
		using Point_t = gv::util::Point<dim,T>;
		using Box_t   = gv::util::Box<dim,T>;
		using Node_t  = OctreeNode<Data_t,dim,n_data,T>; //this should not be exposed to standard use, but it is useful for other containers
	
	protected:
		//data and tree structure
		Node_t*  _root;
		// Data_t* _data;
		std::vector<Data_t> _data;
		// size_t  _data_cursor;
		// size_t  _capacity;
		std::vector<size_t> _indices_to_insert;

		mutable std::shared_mutex _rw_mutex; //allow parallel find/contains but only if no push_back is active. push_back is always serial. most logic is in the nodes
		std::atomic<size_t> _data_count=0; //_data.size() but thread safe
	public:
		//constructor if the bounding box and capacity are unknown
		BasicOctree(const size_t capacity=64) :
			_root(new Node_t(Box_t(Point_t(0), Point_t(1)))),
			_indices_to_insert() {_data.reserve(capacity);}

		//use this constructor if possible. the bounding box should be known ahead of time to avoid re-creating the octree structure.
		//changing the bounding box as a copy constructor must be done in the octree for the specific data type so that the correct
		//is_data_valid() method is available
		BasicOctree(const Box_t &bbox, const size_t capacity=64) :
			_root(new Node_t(bbox)),
			_indices_to_insert()	{_data.reserve(capacity);}

		//copy constructor
		BasicOctree(const BasicOctree& other) :	_root(new Node_t(other.bbox())),
			_indices_to_insert()	{
			_data.reserve(other.size());
			for (size_t i=0; i<other.size(); i++) {
				[[maybe_unused]] int flag = push_back(other[i]);
				assert(flag==1);
			}
		}

		//destructor
		virtual ~BasicOctree() {
			if (_root!=nullptr) {delete _root;}
		}

		//these are why we have an octree
		size_t find(const Data_t &val) const; //find the index for some data. return (size_t) -1 if unsuccessful (most likely larger than _data.size())
		bool contains(const Data_t &val) const; //check if some data is currently in the tree. wrapper around find().
		bool is_processed() const {return _indices_to_insert.empty();}
		void reserve_buffer(const size_t length) {_indices_to_insert.reserve(length);}
		void free_buffer() {_indices_to_insert.resize(0);}

		//add data to the container
		int push_back(const Data_t &val); //attempt to insert data. return 1 on success, 0 if the data was already contained, and -1 on failure
		int push_back(const Data_t &val, size_t &idx); //same as push_back(const Data_t&) but puts the storage index into idx when flag=0 or 1
		int push_back(Data_t &&val); //attempt to insert data. return 1 on success, 0 if the data was already contained, and -1 on failure
		int push_back(Data_t &&val, size_t &idx); //same as push_back(const Data_t&) but puts the storage index into idx when flag=0 or 1

		int push_back_parallel(Data_t &&val, size_t &idx); //add data to the octree and stage it for updating the structure
		void process_insertion(); //process octree after parallel insertion

		//re-insert data into the container (for example if the data stored at idx changes location)
		void reinsert(const size_t idx) {
			resize_to_fit_data(_data[idx],8);
			recursive_remove_idx(_root, idx);
			recursive_insert(_root, _data[idx], idx);
		}


		//data memory information and control
		inline size_t capacity() const {return _data.capacity();} //maximum number of elements
		inline size_t size() const {return _data.size();} //current number of elements
		void reserve(const size_t new_capacity) {
			std::unique_lock<std::shared_mutex> lock(_rw_mutex);
			reserve_unlocked(new_capacity);
		}

		void clear(const bool delete_structure = true); //delete all data and tree structure
		Box_t bbox() const {return _root->bbox;} //get a copy of the bounding box
		void set_bbox(const Box_t& bbox); //re-size the bounding box. requires data a data move and re-creation of the entire tree structure.

		//data access by index
		const Data_t& operator[](const size_t idx) const {
			//return const data with no bound check
			assert(idx<size()); //bound check only if NDEBUG is not defined
			std::shared_lock<std::shared_mutex> lock(_rw_mutex);
			return _data[idx];
		}
		Data_t& operator[](const size_t idx) { //return data with no bound check (octree can be invalidated if data is improperly modified after insertion)
			assert(idx<size()); //bound check only if NDEBUG is not defined
			std::shared_lock<std::shared_mutex> lock(_rw_mutex);
			return _data[idx];
		} 
		const Data_t& at(const size_t idx) const {
			std::shared_lock<std::shared_mutex> lock(_rw_mutex);
			return _data.at(idx);
		}
		Data_t& at(const size_t idx) {
			std::shared_lock<std::shared_mutex> lock(_rw_mutex);
			return _data.at(idx);
		}

		//get data associated with a region
		std::vector<size_t> get_data_indices(const Box_t &box) const; //get all data indices from nodes that intersect the given bounding box.
		std::vector<size_t> get_data_indices(const Point_t &coord) const; //get all data indices from nodes that contain the given coordinate.

		//access nodes of octree
		std::vector<const Node_t*> _get_node(const Point_t &coord) const; //get the nodes with data that contain the given coordinate
		std::vector<const Node_t*> _get_node(const Box_t &box) const; //get all nodes with data that intersect the given region


		//iterators
		std::vector<Data_t>::iterator       begin() {return _data.begin();}
		std::vector<Data_t>::const_iterator begin() const {return _data.cbegin();}
		std::vector<Data_t>::const_iterator cbegin() const {return _data.cbegin();}
		std::vector<Data_t>::iterator       end() {return _data.end();}
		std::vector<Data_t>::const_iterator end() const {return _data.cend();}
		std::vector<Data_t>::const_iterator cend() const {return _data.cend();}


	protected:
		//OVERWRITE THIS IN THE DATA SPECIFIC OCTREE CONTAINER
		virtual bool is_data_valid(const Box_t &bbox, const Data_t &val) const = 0;  //check if data can be placed into a node with the specified bounding box

		void resize_to_fit_data(const Data_t &val, const int iter) {
			if (iter<0) {
				throw std::runtime_error("maximum recursion in resize_to_fit_data");
			}

			if (!is_data_valid(_root->bbox, val)) {
				set_bbox(2*_root->bbox);
				resize_to_fit_data(val, iter-1);
			}
		}
		size_t recursive_find(Node_t* const node, const Data_t &val) const; //primary method for finding data
		Node_t* recursive_find_best_node(Node_t* const node, const Data_t &val) const;
		virtual bool recursive_insert(Node_t* const node, const Data_t &val, const size_t idx) = 0; //primary method for inserting data into octree structure. only modifies the octree structure.
		void recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Point_t &coord) const; //primary method for accessing nodes associated with a coordinate
		void recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const; //primary method for getting leaf nodes intersecting a region
		virtual void divide(Node_t* const node) = 0; //divide a node into its children nodes
		void recursive_remove_idx(Node_t* const node, const size_t idx);

	private:
		//for parallel read-write
		size_t find_unlocked(const Data_t &val) const {
			if (!is_data_valid(_root->bbox, val)) {return (size_t) -1;} //data is invalid so it is not in the octree.
			return recursive_find(_root, val);
		}

		void reserve_unlocked(const size_t new_capacity);
	};


	//////////////////////// BASIC OCTREE IMPLEMENTATION ////////////////////////
	template<typename Data_t, int dim, int n_data, Float T>
	size_t BasicOctree<Data_t,dim,n_data,T>::find(const Data_t &val) const
	{	
		std::shared_lock<std::shared_mutex> lock(_rw_mutex);
		return find_unlocked(val);
	}

	template<typename Data_t, int dim, int n_data, Float T>
	bool BasicOctree<Data_t,dim,n_data,T>::contains(const Data_t &val) const {return find(val) != (size_t) -1;}

	//copy push_back
	template<typename Data_t, int dim, int n_data, Float T>
	int BasicOctree<Data_t,dim,n_data,T>::push_back(const Data_t &val)
	{
		size_t idx = (size_t) -1;
		Data_t copy(val);
		return push_back(std::move(copy), idx);
	}

	//copy push_back
	template<typename Data_t, int dim, int n_data, Float T>
	int BasicOctree<Data_t,dim,n_data,T>::push_back(const Data_t &val, size_t &idx)
	{
		Data_t copy(val);
		return push_back(std::move(copy), idx);
	}


	//move push_back
	template<typename Data_t, int dim, int n_data, Float T>
	int BasicOctree<Data_t,dim,n_data,T>::push_back(Data_t &&val) {
		size_t idx = (size_t) -1;
		return push_back(val, idx);
	}


	//move push_back
	template<typename Data_t, int dim, int n_data, Float T>
	int BasicOctree<Data_t,dim,n_data,T>::push_back(Data_t &&val, size_t &idx) {

		//check if the current box can contain the data
		if (!is_data_valid(_root->bbox, val)) {
			std::unique_lock<std::shared_mutex> lock(_rw_mutex);
			resize_to_fit_data(val,8);
		}


		//attempt to find the data. if the data does not exist, find the node most likely to contain it.
		Node_t* data_node = recursive_find_best_node(_root, val);
		idx = recursive_find(data_node, val);
		if (idx != (size_t) -1) {return 0;} //data was found and its index put into idx

		//the data must be inserted
		{
			std::unique_lock<std::shared_mutex> lock(data_node->_rw_mutex);
			const size_t new_stored_index = _data_count.fetch_add(1);
			bool success = recursive_insert(data_node, val, new_stored_index);
			if (success) {
				idx = new_stored_index; //pass back correct index of the inserted data
				_data.push_back(std::move(val)); //maybe race condition?
				return 1; //successful insertion
			} else {
				throw std::runtime_error("Insertion into Octree failed.");
				return -1; //insertion failed
			}
		}

		//we should never reach this point
		throw std::runtime_error("Octree failed");
		return -1;
	}



	//process _indices_to_process
	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::process_insertion() {
		#pragma omp parallel for
		for (size_t i=0; i<_indices_to_insert.size(); i++) {
			size_t idx = _indices_to_insert[i];

			Node_t* data_node = recursive_find_best_node(_root, _data[idx]);
			assert(data_node!=nullptr);

			std::unique_lock<std::shared_mutex> lock(data_node->_rw_mutex);
			bool success = recursive_insert(data_node, _data[idx], idx);
			assert(success);
		}

		_indices_to_insert.clear();
	}


	//move push_back_parallel
	template<typename Data_t, int dim, int n_data, Float T>
	int BasicOctree<Data_t,dim,n_data,T>::push_back_parallel(Data_t &&val, size_t &idx) {
		//the method that calls this must guarantee that identical data will not be attempted to be inserted
		if (!is_data_valid(_root->bbox, val)) {
			std::unique_lock<std::shared_mutex> lock(_rw_mutex);
			resize_to_fit_data(val, 8);
		}


		//attempt to find the data.
		idx = find_unlocked(val);
		if (idx != (size_t) -1) {return 0;} //data was found and its index put into idx
		
		std::unique_lock<std::shared_mutex> lock(_rw_mutex);
		idx = _data.size();
		_data.push_back(std::move(val));
		_indices_to_insert.push_back(idx);
		return 1;
	}

	
	





	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::recursive_remove_idx(Node_t* const node, const size_t idx) {
		assert(idx<_data.size());
		remove_idx(node, idx); //if the node contains the specified index, remove it
		for (int c_idx=0; c_idx<n_children; c_idx++) {
			if (is_divided(node) and is_data_valid(node->children[c_idx]->bbox, _data[idx])) {
				recursive_remove_idx(node->children[c_idx], idx);
			}
		}
	}

	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::reserve_unlocked(const size_t new_capacity) {
		assert(new_capacity>=size()); //make sure that the new capacity has enough room for the current data
		_data.reserve(new_capacity);
	}

	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::clear(const bool delete_structure)
	{
		//delete current data and tree structure
		if (delete_structure and _root!=nullptr)
		{
			Box_t bbox = _root->bbox; 
			delete _root; 
			_root = new Node_t(bbox);
		}

		_data.clear();
	}

	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::set_bbox(const Box_t& new_bbox)
	{
		if (_root->bbox.contains(new_bbox)) {return;}

		//find a new_root_bbox such that
		// a) new_root_bbox contains new_bbox
		// b) the current _root->bbox is the descendent of an OctreeNode with new_root_bbox


		//expand the root bounding box and shift to be as close to new_root_bbox as possible
		Box_t expanded_root_bbox = 2*_root->bbox;
		int max_vertices=-1;
		int best_sibling_number=-1;
		for (int i=0; i<n_children; i++) {
			Point_t offset = _root->bbox.voxelvertex(i) - expanded_root_bbox.voxelvertex(i);
			Box_t test_box = expanded_root_bbox + offset;
			int n_verts = 0;
			for (int j=0; j<n_children; j++) {
				if (test_box.contains(new_bbox.voxelvertex(j))) {n_verts++;}
			}

			if (n_verts>max_vertices) {
				best_sibling_number=i;
				max_vertices = n_verts;
			}
		}

		//make the new box
		Point_t offset     = _root->bbox.voxelvertex(best_sibling_number) - expanded_root_bbox.voxelvertex(best_sibling_number);
		Box_t new_root_box = expanded_root_bbox + offset;
		
		//make the new root and point it to the old root
		Node_t* old_root = _root;
		_root = new Node_t(new_root_box, old_root->depth - 1);
		divide(_root);
		delete _root->children[best_sibling_number];
		_root->children[best_sibling_number] = old_root;
		old_root->parent = _root;
		assert(_root->bbox.voxelvertex(best_sibling_number) == old_root->bbox.voxelvertex(best_sibling_number));
		assert(Box_t(_root->bbox.center(), _root->bbox.voxelvertex(best_sibling_number)) == old_root->bbox);

		//re-call in case we need to still expand
		if (max_vertices < n_children) {set_bbox(new_bbox);}
	}


	template<typename Data_t, int dim, int n_data, Float T>
	std::vector<size_t> BasicOctree<Data_t,dim,n_data,T>::get_data_indices(const Box_t &box) const
	{
		assert(box.intersects(_root->bbox));
		std::vector<size_t> result;

		std::vector<const Node_t*> nodes = _get_node(box);

		for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
		{
			const Node_t* const node = nodes[n_idx];
			for (int d_idx=0; d_idx<node->data_cursor; d_idx++)	{
				size_t idx = node->data_idx[d_idx];
				if (is_data_valid(box, _data[idx])) {result.push_back(idx);}
			}
		}

		//make result contain only unique values
		std::sort(result.begin(), result.end());
		auto delete_past = std::unique(result.begin(), result.end());
		result.erase(delete_past, result.end());

		return result;
	}

	template<typename Data_t, int dim, int n_data, Float T>
	std::vector<size_t> BasicOctree<Data_t,dim,n_data,T>::get_data_indices(const Point_t &coord) const
	{
		assert(_root->bbox.contains(coord));
		std::vector<size_t> result;

		std::vector<const Node_t*> nodes = _get_node(coord);

		for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
		{
			const Node_t* const node = nodes[n_idx];
			for (int d_idx=0; d_idx<node->data_cursor; d_idx++)
			{
				size_t idx = node->data_idx[d_idx];
				result.push_back(idx);
			}
		}

		//make result contain only unique values
		std::sort(result.begin(), result.end());
		auto delete_past = std::unique(result.begin(), result.end());
		result.erase(delete_past, result.end());

		return result;
	}

	template<typename Data_t, int dim, int n_data, Float T>
	std::vector<const typename BasicOctree<Data_t,dim,n_data,T>::Node_t*> BasicOctree<Data_t,dim,n_data,T>::_get_node(const Point_t &coord) const
	{
		std::vector<const Node_t*> result;
		recursive_get_node(_root, result, coord);
		return result;
	}

	template<typename Data_t, int dim, int n_data, Float T>
	std::vector<const typename BasicOctree<Data_t,dim,n_data,T>::Node_t*> BasicOctree<Data_t,dim,n_data,T>::_get_node(const Box_t &box) const
	{
		std::vector<const Node_t*> result;
		recursive_get_node(_root, result, box);
		return result;
	}


	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Point_t &coord) const
	{
		//it is assumed that we are in a valid node
		assert(node->bbox.contains(coord));

		//check if the node has data
		if (node->data_cursor > 0) {result.push_back(node);}

		//if divided, recurse into each valid child
		if (is_divided(node))
		{
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx];
				if (child->bbox.contains(coord)) {recursive_get_node(child, result, coord);}
			}
		}
	}

	template<typename Data_t, int dim, int n_data, Float T>
	void BasicOctree<Data_t,dim,n_data,T>::recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const
	{
		//it is assumed that the current node intersect the specified box
		assert(node->bbox.intersects(box));

		//check if the node has data
		if (node->data_cursor > 0) {result.push_back(node);}

		//if divided, recurse into each valid child
		if (is_divided(node))
		{
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx];
				if (child->bbox.intersects(box)) {recursive_get_node(child, result, box);}
			}
		}
	}


	template<typename Data_t, int dim, int n_data, Float T>
	size_t BasicOctree<Data_t,dim,n_data,T>::recursive_find(Node_t* const node, const Data_t &val) const {
		// std::shared_lock<std::shared_mutex> lock(node->_rw_mutex);
		//it is assumed that is_data_valid(node,val) is true
		// assert(is_data_valid(node->bbox, val));
		if (!is_data_valid(node->bbox, val)) {return (size_t) -1;}

		//search data in this node
		for (int d_idx=0; d_idx<node->data_cursor; d_idx++)	{
			size_t idx = node->data_idx[d_idx];
			if (val==_data[idx]) {return idx;}
		}

		//recurse into valid children
		if (is_divided(node)) {
			for (int c_idx=0; c_idx<n_children; c_idx++) {
				Node_t* child = node->children[c_idx]; 
				if (is_data_valid(child->bbox, val)) {
					//make sure that the child node is not being changed here
					size_t result = recursive_find(child, val); 
					if (result != (size_t) -1) {return result;}
				}
			}
		}

		//data could not be found
		return (size_t) -1;
	}


	template<typename Data_t, int dim, int n_data, Float T>
	typename BasicOctree<Data_t,dim,n_data,T>::Node_t* BasicOctree<Data_t,dim,n_data,T>::recursive_find_best_node(Node_t* const node, const Data_t &val) const {

		//if the data is not valid in the current node, move up the tree if possible
		// if (!is_data_valid(node->bbox, val)) {
		// 	if (node->parent != nullptr) { return recursive_find(node->parent, val, idx);}
		// 	else {
		// 		idx = (size_t) -1;
		// 		return nullptr;
		// 	}
		// }

		std::shared_lock<std::shared_mutex> lock(node->_rw_mutex);
		//it the data must be valid in this node now
		assert(is_data_valid(node->bbox, val));

		//search data in this node
		for (int d_idx=0; d_idx<node->data_cursor; d_idx++)	{
			size_t index = node->data_idx[d_idx];
			if (val==_data[index]) {
				return node;
			}
		}
		
		//recurse into valid children
		if (is_divided(node)) {
			for (int c_idx=0; c_idx<n_children; c_idx++) {
				Node_t* child = node->children[c_idx]; 
				if (is_data_valid(child->bbox, val)) {
					return recursive_find_best_node(child, val);
				}
			}
		}

		//data could not be found, but this node (or a sibling) is a likely candidate to store the data
		if (!is_divided(node)) {
			if (node->parent != nullptr) {return node->parent;}
		}
		return node;
	}
}