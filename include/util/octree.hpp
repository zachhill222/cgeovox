#pragma once

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <stdexcept>
#include <limits>

namespace gv::util{
	//Data structure for generic octree nodes.
	template <typename data_t, int dim=3, size_t n_data=8>
	struct _OctreeNode
	{
		//tree logic
		static const int n_children = std::pow(2,dim);
		bool is_divided = false;
		const _OctreeNode<data_t, dim, n_data>* parent = NULL;
		_OctreeNode<data_t, dim, n_data>* children[n_children] {NULL};

		//geometry logic
		const Box<dim> bbox;

		//data logic
		size_t cursor = 0;
		size_t* data_idx = NULL; //don't default to data_idx[n_data] so that data in non-leaf nodes can be deleted.

		//constructor
		_OctreeNode(const _OctreeNode<data_t,dim,n_data>* parent, const Point<dim,double> &v1, const Point<dim,double> &v2) : parent(parent), bbox(Box<dim>(v1,v2))
		{
			data_idx = new size_t[n_data];
		}

		//destructor
		~_OctreeNode()
		{
			if (data_idx!=NULL) {delete[] data_idx;}
			if (is_divided)
			{
				for (int i=0; i<n_children; i++) {delete children[i];}
			}
		}
	};




	///Basic octree class, can be used in d=2 or d=3 (and possibly others). A octree that will be used should inherit from this class and define the member function is_data_valid().
	template <typename data_t, int dim=3, bool multiple_data=false, size_t n_data=8>
	class BasicOctree
	{
	public:
		static const int dimension = dim;
		static const int n_children = std::pow(2,dim);
	
	protected:
		//convenient reference to the type of nodes in this octree.
		typedef _OctreeNode<data_t,dim,n_data> Node;

		//tree logic
		Node* root = NULL;
		std::vector<const Node*> _nodelist; //this may be bad. using it as a method to go through all nodes. TODO: implement a Node* next_node(Node*) method.
		
		//data
		std::vector<data_t> _data;


		///member function for determining where to put data. Must be overridden by implemented octree classes.
		virtual bool is_data_valid(const Box<dim> &box, const data_t &data) const {return false;}
		
		//DIVISION
		void divide_multiple_data(Node* node)
		{
			//find box center for constructing bounding boxes of children
			Point<dim,double> _center = node->bbox.center();
			
			//make children and pass data
			for (int i=0; i<n_children; i++)
			{
				node->children[i] = new Node(node, node->bbox[i], _center);
				//pass data indices to children
				for (size_t j=0; j<node->cursor; j++)
				{	
					//check if data can be passed to child.
					if (is_data_valid(node->children[i]->bbox, _data[node->data_idx[j]]))
					{	
						//insert data
						node->children[i]->data_idx[node->children[i]->cursor] = node->data_idx[j];
						node->children[i]->cursor += 1;
					}
				}
			}
		}

		void divide_single_data(Node* node)
		{
			//find box center for constructing bounding boxes of children
			Point<dim,double> _center = node->bbox.center();
			
			//initialize array for ensuring only one copy of any data is passed
			bool data_passed[node->cursor] {false};

			//make children and pass data
			for (int i=0; i<n_children; i++)
			{
				node->children[i] = new Node(node, node->bbox[i], _center);
				//pass data indices to children
				for (size_t j=0; j<node->cursor; j++)
				{	
					if (data_passed[j]){continue;} //this data was already passed

					//check if data can be passed to child.
					if (is_data_valid(node->children[i]->bbox, _data[node->data_idx[j]]))
					{
						//insert data
						data_passed[j] = true;
						node->children[i]->data_idx[node->children[i]->cursor] = node->data_idx[j];
						node->children[i]->cursor += 1;
					}
				}
			}
		}

		void divide(Node* node)
		{
			//check if division is allowed
			if (node==NULL){return;}
			if (node->is_divided){return;}

			//call appropriate division based on data
			if(multiple_data){divide_multiple_data(node);}
			else{divide_single_data(node);}

			//free memory
			delete[] node->data_idx;
			node->data_idx = NULL;

			//set node to divided
			node->is_divided = true;

			//append children to list of nodes. TODO: delete when Node* next_node(Node*) is implemented.
			for (int i=0; i<n_children; i++)
			{
				_nodelist.push_back(node->children[i]);
			}
		}

		
		//INSERTION
		bool recursive_insert(Node* node, const size_t &idx)
		{
			//check if current node is divided
			if (not node->is_divided)
			{
				//node is not divided
				//check if current node has room for more data
				if (node->cursor < n_data)
				{
					// std::cout << "adding data at location " << node->cursor << std::endl;
					node->data_idx[node->cursor] = idx;
					node->cursor += 1;
					return true;
				}
				else
				{
					//node must divide
					divide(node);
					return recursive_insert(node, idx);
				}
			}
			else
			{
				//node is divided and data has not been inserted
				for (int i=0; i<n_children; i++)
				{
					if (is_data_valid(node->children[i]->bbox, _data[idx]))
					{
						// std::cout << "recurse to child " << i << " ";
						bool success = recursive_insert(node->children[i], idx);
						if (success and !multiple_data) {return true;}
					}
				}
			}
			//data could not be inserted
			return false;
		}


		//FIND NODES
		bool recursive_find(Node const* node, const data_t &val, size_t &idx) const
		{
			// std::cout << "octree: recursive_find at node " << node << std::endl;
			if (node->children[0]==NULL)
			{
				// std::cout << "node has " << node->cursor << " data indices" << std::endl;
				//not divided
				for (size_t j=0; j<node->cursor; j++)
				{
					// std::cout << "octree: recursive_find: compare data " << j << std::endl;
					if (val==_data[node->data_idx[j]])
					{
						idx = node->data_idx[j];
						return true;
					}
				}
			}
			else
			{
				for (int i=0; i<n_children; i++)
				{
					if (is_data_valid(node->children[i]->bbox, val))
					{
						if (recursive_find(node->children[i], val, idx)) {return true;}
					}
				}
			}

			return false;
		}


		//PRINT
		void print(Node* node, int depth) const
		{
			if (node->children[0]==NULL)
			{
				for (int k=0; k<depth; k++){std::cout << " - ";}
				for (size_t j=0; j<node->cursor; j++) {std::cout << node->data_idx[j] << " ";}
				std::cout << std::endl;
			}
			else
			{
				for (int i=0; i<n_children; i++){print(node->children[i], depth+1);}
			}
		}

		//TREE TRAVERSAL
		const Node* getnode( const Node* node, const Point<dim,double> &point ) const
		{
			if (node->children[0]==NULL)
			{
				//in a leaf node
				if (node->bbox.contains(point)) {return node;}
			}
			else
			{
				for (int i=0; i<n_children; i++)
				{
					if (node->children[i]->bbox.contains(point)) {return getnode(node->children[i], point);}
				}
			}

			return NULL;
		}

		const Node* getnode( const Point<dim,double> &point ) const
		{
			if (root->bbox.contains(point)) {return getnode(root, point);}
			return NULL;
		}

	
	public:
		BasicOctree() 
		{
			root = new Node(NULL, Point<dim,double>{0,0,0}, Point<dim,double> {1,1,1});
			_nodelist.push_back(root);
		}

		BasicOctree(const Box<dim> &bbox)
		{
			root = new Node(NULL, bbox.low(), bbox.high());
			_nodelist.push_back(root);
		}

		~BasicOctree() {delete root;}

		// size_t root_cursor() const {return root->cursor;}

		///method to re-size scope of the octree. usefull when the domain of the data is unknown a-priori, but this copies data and re-constructs the octree (it is slow if the octree is large).
		void set_bbox(const Box<dim> &bbox)
		{
			//delete current tree
			delete root;

			//initialize new tree
			root = new Node(NULL, bbox.low(), bbox.high());

			//copy temporary data and clear current data
			std::vector<data_t> old_data_copy = _data;
			_data.clear();

			//reconstruct tree from copied data
			for (size_t i=0; i<old_data_copy.size(); i++)
			{
				push_back(old_data_copy[i]);
			}
		}

		//methods to get bounding boxes of nodes and entire tree.
		Box<dim> bbox() const {return root->bbox;}
		Box<dim> bbox(const size_t idx) const {return _nodelist[idx]->bbox;}
		
		int nData(const size_t idx) const
		{
			const Node* node = _nodelist[idx];
			if (node->is_divided) {return 0;}
			return node->cursor;
		}
		bool isLeaf(const size_t idx) const {return not _nodelist[idx]->is_divided;}

		///return number of nodes (not just leaf nodes);
		size_t nNodes() const {return _nodelist.size();}


		//DATA LOGIC AND CONTROL
		///return index of data.
		size_t find(const data_t &val) const
		{
			size_t idx;
			if (recursive_find(root, val, idx)) {return idx;}
			return (size_t) (-1);
		}

		bool find(const data_t &val, size_t &idx) const {return recursive_find(root, val, idx);}

		bool contains(const data_t &val) const
		{
			size_t idx;
			return recursive_find(root, val, idx);
		}

		void push_back(const data_t &val)
		{
			if (contains(val)) {return;}
			_data.push_back(val);
			size_t idx = _data.size()-1;
			recursive_insert(root, idx);
		}

		void print() const {print(root,0);}

		inline const data_t& operator[](const size_t &idx) const {return _data[idx];}
		inline size_t size() const {return _data.size();}
		inline size_t capacity() const {return _data.capacity();}
		inline void reserve(size_t size){_data.reserve(size);}
		void clear()
		{
			_data.clear();
			for (int i=0; i<n_children; i++){delete root->children[i];}
		}
	};




	///Octree for points in space.
	template <int dim=3, typename T=double, size_t n_data=32>
	class PointOctree : public BasicOctree<Point<dim,double>, dim, false, n_data>
	{
	public:
		PointOctree() : BasicOctree<Point<dim,double>, dim, false, n_data>() {}
		PointOctree(const Box<dim> &bbox) : BasicOctree<Point<dim,double>, dim, false, n_data>(bbox) {}

		// size_t closest_point(const Point<dim,double> &point) const
		// {

		// }

	private:
		bool is_data_valid(const Box<dim> &box, const Point<dim,double> &data) const override {return box.contains(data);}

		// double dist_squared(const Node* node, const Point<dim,double> &point) const
		// {
		// 	if (not node->is_divided)
		// 	{
		// 		double dist = std::numeric_limits<double>::max();
		// 		for (size_t j=0; j<node->cursor; j++)
		// 		{
		// 			double temp_dist = (point - _data[node->data_idx[j]]).normSquared();
		// 			dist = gv::util::min(dist, temp_dist);
		// 		}
		// 		return dist;
		// 	}
		// 	return gv::util::dist_squared(node->bbox, point);
		// }

		// size_t recursive_closest_point(const Node* node, const Point<dim,double> &point) const
		// {
		// 	if (not node->is_divided and node->cursor>0)
		// 	{
		// 		size_t idx = 0;
		// 		double dist = (point-_data[node->data_idx[0]]).normSquared();
		// 		for (size_t j=1; j<node->cursor; j++)
		// 		{
		// 			double temp_dist = (point-_data[node->data_idx[j]]).normSquared();
		// 			if (temp_dist<dist)
		// 			{
		// 				dist = temp_dist;
		// 				idx = j;
		// 			}
		// 		}
		// 		return node->data_idx[idx];
		// 	}
		// 	else
		// 	{
				
		// 	}
		// }
	};
}

