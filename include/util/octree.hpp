#pragma once

//standard octree implementation for organizing data in 2D and 3D space
//each data type (Data_t) will require an octree container that inherits from the BasicOctree class (e.g., <Data_t>Octree)
//each user implemented octree container will require a method with the signature: bool is_data_valid(const Box_t &bbox, const Data_t &val) for data insertion
//if Data_t encloses a volume (e.g. a spherical particle), then is_data_valid(bbox,val) should test for intersection of bbox and val.
//the data type must implement an equality comparison (e.g., bool operator==(const Data_t& other) const) only unique data (by this comparison) will be stored in the container
//to be able to effectively re-size the array, Data_t must have a move assignment operator (std::move() must be callable)

//TODO:
//  -implement push_back(Data_t&&)
//	-implement data deletion
//	-implement data sorting

#include "util/point.hpp"
#include "util/box.hpp"
#include "compile_constants.hpp" //max octree depth

#include <cassert>
#include <vector>

namespace gv::util
{
	//class for the nodes of the octree
	//this class stores indices to a primary vector where the data is stored
	//data indices are only stored in leaf nodes and a single index may be stored in multiple nodes
	//this is useful when Data_t encloses a region rather than being located at a single point and mostly irrelevant if the data is located at a point

	template<typename Data_t, int dim=3, int n_data=8>
	class OctreeNode
	{
	public:
		//common typedefs and constants
		static constexpr int _dim = dim;
		static constexpr int n_children = std::pow(2,dim);
		using Node_t  = OctreeNode<Data_t,dim,n_data>;
		using Point_t = gv::util::Point<dim,double>;
		using Box_t   = gv::util::Box<dim>;

		//constructor for initializing _root node
		OctreeNode(const Box_t& bbox) :
			data_idx(new size_t[n_data]), 
			data_cursor(0), 
			bbox(bbox), 
			parent(nullptr), 
			sibling_number(-1),
			n_descendents(0),
			depth(0) {}

		//constructor for initializing children nodes
		OctreeNode(Node_t* const parent, const int sibling_number) :
			data_idx(new size_t[n_data]), 
			data_cursor(0), 
			bbox(parent->bbox.voxelvertex(sibling_number), parent->bbox.center()), 
			parent(parent), 
			sibling_number(sibling_number),
			n_descendents(0),
			depth(parent->depth+1) {}

		//destructor
		~OctreeNode()  //REMARK: I don't think any class should inherit from this. otherwise this should be virtual.
		{
			if (data_idx!=nullptr) {delete[] data_idx;}
			if (children[0]!=nullptr) {for (int c_idx=0; c_idx<n_children; c_idx++) {delete children[c_idx];}}
		}

		//data storage
		size_t* data_idx; //only store the pointer here so that it can be deleted when a node is divided
		int  data_cursor = 0; //used to track where the next data index should be written (e.g., data_idx[data_cursor])

		//geometry
		const Box_t bbox; //portion of space that this node is responsible for

		//tree information
		Node_t* const parent;
		Node_t* children[n_children] {nullptr};
		int const sibling_number; //parent->children[this->sibling_number] == this
		size_t n_descendents; //not used for now
		int const depth;
	};

	//helpful functions to call on nodes
	template<typename Data_t, int dim, int n_data>
	bool is_divided(const OctreeNode<Data_t,dim,n_data>* const node) {return node->children[0]!=nullptr;}

	template<typename Data_t, int dim, int n_data>
	void append_index(OctreeNode<Data_t,dim,n_data>* const node, const int idx)
	{
		//it is assumed that there is room
		assert(node->data_cursor<n_data);
		node->data_idx[node->data_cursor] = idx;
		node->data_cursor += 1;
	}



	//octree container. handles division of nodes, insertion and retrieval of data, storage of data, etc.
	template<typename Data_t, int dim=3, int n_data=8>
	class BasicOctree
	{
	public:
		//common typedefs and constants
		static constexpr int _dim = dim;
		static constexpr int n_children = std::pow(2,dim);
		using Point_t = gv::util::Point<dim,double>;
		using Box_t   = gv::util::Box<dim>;
		using Node_t  = OctreeNode<Data_t,dim,n_data>; //this should not be exposed to standard use, but it is useful for other containers
	
	private:
		//data and tree structure
		Node_t*  _root;
		Data_t* _data;
		size_t  _data_cursor;
		size_t  _capacity;

	public:
		//constructor if the bounding box and capacity are unknown
		BasicOctree(const size_t capacity=64) :
			_root(new Node_t(Box_t(Point_t(0), Point_t(1)))),
			_data(new Data_t[capacity]),
			_data_cursor(0),
			_capacity(capacity) {}

		//use this constructor if possible. the bounding box should be known ahead of time to avoid re-creating the octree structure.
		//changing the bounding box as a copy constructor must be done in the octree for the specific data type so that the correct
		//is_data_valid() method is available
		BasicOctree(const Box_t &bbox, const size_t capacity=64) :
			_root(new Node_t(bbox)),
			_data(new Data_t[capacity]),
			_data_cursor(0),
			_capacity(capacity) {}

		//destructor
		virtual ~BasicOctree()
		{
			if (_root!=nullptr) {delete _root;}
			if (_data!=nullptr) {delete[] _data; _data_cursor=0;}
		}

		//these are why we have an octree
		size_t find(const Data_t &val) const; //find the index for some data. return (size_t) -1 if unsuccessful (most likely larger than _data.size())
		bool contains(const Data_t &val) const; //check if some data is currently in the tree. wrapper around find().
		
		//add data to the container
		int push_back(const Data_t &val); //attempt to insert data. return 1 on success, 0 if the data was already contained, and -1 on failure
		int push_back(const Data_t &val, size_t &idx); //same as push_back(const Data_t&) but puts the storage index into idx when flag=0 or 1

		//data memory information and control
		inline size_t capacity() const {return _capacity;} //maximum number of elements
		inline size_t size() const {return _data_cursor;} //current number of elements
		void reserve(const size_t new_capacity); //re-size reserved space
		void clear(); //delete all data and tree structure
		Box_t bbox() const {return _root->bbox;} //get a copy of the bounding box
		void set_bbox(const Box_t& bbox); //re-size the bounding box. requires data a data move and re-creation of the entire tree structure.

		//data access by index
		const Data_t& operator[](const size_t idx) const {return _data[idx];} //return const data with no bound check
		Data_t& operator[](const size_t idx) {return _data[idx];} //return data with no bound check (octree can be invalidated if data is improperly modified after insertion)
		const Data_t& at(const size_t idx) const {assert(idx<size()); return _data[idx];} //return const data with bound check
		Data_t& at(const size_t idx) {assert(idx<size()); return _data[idx];} //return data with bound check (octree can be invalidated if data is improperly modified after insertion)

		//get data associated with a region
		std::vector<size_t> get_data_indices(const Box_t &box) const; //get all data indices from nodes that intersect the given bounding box.
		std::vector<size_t> get_data_indices(const Point_t &coord) const; //get all data indices from nodes that contain the given coordinate.

		//access nodes of octree
		const Node_t* _get_node(const Point_t &coord) const; //get the first leaf node with the given coordinate
		std::vector<const Node_t*> _get_node(const Box_t &box) const; //get all leaf nodes that intersect the given region

	protected:
		//this is also why we have an octree. useful for getting all data associated with some key. implement in data specific octree container.
		// Node_t* _get_node(const Point_t &coord); //get first leaf node that is contains with the specified coordinate
		// std::vector<Node_t*> _get_node(const Box_t &box); //get all leaf nodes that intersect the specified region

		//OVERWRITE THIS IN THE DATA SPECIFIC OCTREE CONTAINER
		virtual bool is_data_valid(const Box_t &bbox, const Data_t &val) const {assert(false); return false;} //check if data can be placed into a node with the specified bounding box

	private:
		size_t recursive_find(Node_t* const node, const Data_t &val) const; //primary method for finding data
		bool recursive_insert(Node_t* const node, const Data_t &val, const size_t idx); //primary method for inserting data into octree structure
		const Node_t* recursive_get_node(const Node_t* const node, const Point_t &coord) const; //primary method for accessing nodes associated with a coordinate
		void recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const; //primary method for getting leaf nodes intersecting a region
		void divide(Node_t* const node); //divide a node into its children nodes
	};


	//////////////////////// BASIC OCTREE IMPLEMENTATION ////////////////////////
	template<typename Data_t, int dim, int n_data>
	size_t BasicOctree<Data_t,dim,n_data>::find(const Data_t &val) const
	{	
		if (!is_data_valid(_root->bbox, val)) {return (size_t) -1;} //data is invalid so it is not in the octree.
		return recursive_find(_root, val);
	}

	template<typename Data_t, int dim, int n_data>
	bool BasicOctree<Data_t,dim,n_data>::contains(const Data_t &val) const {return find(val)!= (size_t) -1;}

	template<typename Data_t, int dim, int n_data>
	int BasicOctree<Data_t,dim,n_data>::push_back(const Data_t &val)
	{
		size_t idx;
		return push_back(val, idx);
	}

	template<typename Data_t, int dim, int n_data>
	int BasicOctree<Data_t,dim,n_data>::push_back(const Data_t &val, size_t &idx)
	{
		if (size()>=capacity()) {reserve(2*_capacity);} //increase storage size if needed
		assert(size()<capacity());

		if (!is_data_valid(_root->bbox,val)) {return -1;} //data can't be added

		idx = find(val);
		if (idx == (size_t) -1)
		{
			bool success = recursive_insert(_root, val, _data_cursor);
			if (success)
			{
				idx = _data_cursor; //pass back correct index of the inserted data
				_data[_data_cursor] = val;
				_data_cursor += 1;
				return 1; //data was not contained and was successfully inserted
			}
			std::cout << "COULD NOT INSERT\n" << val << "\n" << std::endl;
			return -1; //data was not contained and could not be inserted
		}
		return 0; //data was already contained and did not need to be inserted
	}

	template<typename Data_t, int dim, int n_data>
	void BasicOctree<Data_t,dim,n_data>::reserve(const size_t new_capacity)
	{
		assert(new_capacity>=size()); //make sure that the new capacity has enough room for the current data
		
		//move old data to temporary location
		Data_t* old_data = _data;

		//prepare new data array
		_data = new Data_t[new_capacity];
		_capacity = new_capacity;

		//move data
		for (size_t idx=0; idx<size(); idx++) //copy old data into new array. the data is in the same order, so the octree structure does not change.
		{
			_data[idx] = std::move(old_data[idx]);
		}

		//free the old array and re-point it to the new array
		delete[] old_data;
	}

	template<typename Data_t, int dim, int n_data>
	void BasicOctree<Data_t,dim,n_data>::clear()
	{
		//delete current data and tree structure
		if (_root!=nullptr)
		{
			Box_t bbox = _root->bbox; 
			delete _root; 
			_root = new Node_t(bbox);
		}

		if (_data!=nullptr)
		{
			delete[] _data; 
			_data_cursor=0;
		}
	}

	template<typename Data_t, int dim, int n_data>
	void BasicOctree<Data_t,dim,n_data>::set_bbox(const Box_t& new_bbox)
	{
		//re-set the tree structure
		delete _root;
		_root = new Node_t(new_bbox);
		
		//move the data so that it can be re-inserted
		Data_t* old_data = _data;
		size_t  old_data_cursor = _data_cursor;

		//re-set the octree data
		_data = new Data_t[_capacity];
		_data_cursor = 0;

		//add the data to the new octree
		for (size_t d_idx=0; d_idx<old_data_cursor; d_idx++)
		{
			push_back(old_data[d_idx]);
		}

		//delete old tree structure and data. point to the newly created data.
		delete[] old_data;
	}


	template<typename Data_t, int dim, int n_data>
	std::vector<size_t> BasicOctree<Data_t,dim,n_data>::get_data_indices(const Box_t &box) const
	{
		assert(box.intersects(_root->bbox));
		std::vector<size_t> result;

		bool used_indices[_capacity] {false};
		std::vector<const Node_t*> nodes = _get_node(box);
		assert(nodes.size()>0);

		for (size_t n_idx=0; n_idx<nodes.size(); n_idx++)
		{
			const Node_t* const node = nodes[n_idx];
			for (int d_idx=0; d_idx<node->data_cursor; d_idx++)
			{
				size_t idx = node->data_idx[d_idx];
				if (!used_indices[idx] and is_data_valid(box, _data[idx]))
				{
					result.push_back(idx);
					used_indices[idx] = true;
				}
			}
		}

		return result;
	}

	template<typename Data_t, int dim, int n_data>
	std::vector<size_t> BasicOctree<Data_t,dim,n_data>::get_data_indices(const Point_t &coord) const
	{
		assert(_root->bbox.contains(coord));
		std::vector<size_t> result;

		bool used_indices[_capacity] {false}; //TODO: remove to reduce memory overhead
		const Node_t* node = _get_node(coord);

		for (int d_idx=0; d_idx<node->data_cursor; d_idx++)
		{
			size_t idx = node->data_idx[d_idx];
			if (!used_indices[idx])
			{
				result.push_back(idx);
				used_indices[idx] = true;
			}
		}

		return result;
	}

	template<typename Data_t, int dim, int n_data>
	const typename BasicOctree<Data_t,dim,n_data>::Node_t* BasicOctree<Data_t,dim,n_data>::_get_node(const Point_t &coord) const
	{
		assert(_root->bbox.contains(coord));
		return recursive_get_node(_root, coord);
	}

	template<typename Data_t, int dim, int n_data>
	std::vector<const typename BasicOctree<Data_t,dim,n_data>::Node_t*> BasicOctree<Data_t,dim,n_data>::_get_node(const Box_t &box) const
	{
		std::vector<const Node_t*> result;
		recursive_get_node(_root, result, box);
		return result;
	}

	template<typename Data_t, int dim, int n_data>
	size_t BasicOctree<Data_t,dim,n_data>::recursive_find(Node_t* const node, const Data_t &val) const
	{
		//it is assumed that is_data_valid(node,val) is true
		assert(is_data_valid(node->bbox, val));

		//recurse into first valid child
		if (is_divided(node))
		{
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx]; //convenient reference
				if (is_data_valid(child->bbox, val))
				{   //only recursing into the first child node will still find the data because every leaf node in which the data is valid
					//will store an index to the data
					return recursive_find(child, val); 
				}
			}
		}

		//in valid leaf. loop through data.
		for (int d_idx=0; d_idx<node->data_cursor; d_idx++)
		{
			size_t idx = node->data_idx[d_idx];
			if (val==_data[idx]) {return idx;}
		}

		//data could not be found
		return (size_t) -1;
	}

	template<typename Data_t, int dim, int n_data>
	bool BasicOctree<Data_t,dim,n_data>::recursive_insert(Node_t* const node, const Data_t &val, const size_t idx)
	{
		//insert data into ALL leaf nodes that are valid
		//it is assumed that is_data_valid(node,val) is true
		assert(is_data_valid(node->bbox,val));

		//recurse into all valid children
		if (is_divided(node))
		{
			bool success = true;
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx];

				//attempt to put data into all valid children
				if (is_data_valid(child->bbox, val))
				{
					success = success and recursive_insert(child, val, idx); //data must be successfully added to all leaves
				}
			}
			return success;
		}

		//in valid leaf node. store index to data if there is room
		if (node->data_cursor < n_data)
		{
			append_index(node,idx); return true;
		}

		//in valid leaf node, but it must divide
		assert(node->depth<gv::constants::OCTREE_MAX_DEPTH);
		if (node->depth>=gv::constants::OCTREE_MAX_DEPTH) {return false;} //could not insert data
		divide(node); //moves data, frees some memory, creates children. checks for a maximum tree depth.
		return recursive_insert(node, val, idx); //re-run insertion here now that it is divided
	}


	template<typename Data_t, int dim, int n_data>
	const typename BasicOctree<Data_t,dim,n_data>::Node_t* BasicOctree<Data_t,dim,n_data>::recursive_get_node(const Node_t* const node, const Point_t &coord) const
	{
		//it is assumed that we are in a valid node
		assert(node->bbox.contains(coord));

		//if divided, recurse into the first valid node
		if (is_divided(node))
		{
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx];
				if (child->bbox.contains(coord)) {return recursive_get_node(child, coord);}
			}
		}

		//in valid leaf
		return node;
	}

	template<typename Data_t, int dim, int n_data>
	void BasicOctree<Data_t,dim,n_data>::recursive_get_node(const Node_t* const node, std::vector<const Node_t*> &result, const Box_t &box) const
	{
		//it is assumed that the current node intersect the specified box
		assert(node->bbox.intersects(box));

		//if divided, recurse into each valid node
		if (is_divided(node))
		{
			for (int c_idx=0; c_idx<n_children; c_idx++)
			{
				Node_t* child = node->children[c_idx];
				if (child->bbox.intersects(box)) {recursive_get_node(child, result, box);}
			}
			return;
		}

		//in valid leaf node
		result.push_back(node);
	}

	template<typename Data_t, int dim, int n_data>
	void BasicOctree<Data_t,dim,n_data>::divide(Node_t* const node)
	{
		//it is assumed that the node is not divided already and that we will not violate the maximum tree depth
		assert(!is_divided(node));
		assert(node->depth<gv::constants::OCTREE_MAX_DEPTH);

		//create children and copy valid data into children
		for (int c_idx=0; c_idx<n_children; c_idx++)
		{
			Node_t* child = new Node_t(node, c_idx); //create child
			node->children[c_idx] = child; //use child as a convenient reference locally
			for (int d_idx=0; d_idx<node->data_cursor; d_idx++) //initialize data in child
			{
				size_t idx = node->data_idx[d_idx];
				if (is_data_valid(child->bbox, _data[idx]))
				{
					append_index(child, idx);
				}
			}
		}

		//free memory at current node
		if (node->data_idx!=nullptr)
		{
			delete[] node->data_idx;
			node->data_idx=nullptr; //mark that this has been deleted to avoid double frees
			node->data_cursor = 0;
		}
	}
}