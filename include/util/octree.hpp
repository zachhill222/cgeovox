#pragma once


#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <stdexcept>


namespace gv::util{
	
	///Basic octree class, can be used in d=2 or d=3 (and possibly others). A octree that will be used should inherit from this class and define the member function is_data_valid().
	template <typename data_t, int dim=3, bool multiple_data=false, size_t n_data=8, typename T=double>
	class BasicOctree
	{
	protected:
		static const int n_children = std::pow(2,dim);
		struct Node
		{
			size_t cursor = 0; //cursor for inserting data. points to next data to insert.
			// int traversal_index = 0; //cursor for tracking current progress of a tree traversal.
			size_t* data_idx = NULL;
			Node* children[n_children] {NULL};
			const Node* parent = NULL;
			const Box<dim,T> bbox;
			Node(const Node* parent, const Point<dim,T> &v1, const Point<dim,T> &v2) : parent(parent), bbox(Box<dim,T>(v1,v2)) {data_idx = new size_t[n_data];}
			~Node()
			{
				if (data_idx!=NULL){delete[] data_idx;}
				if (children[0]!=NULL)
				{
					for (int i=0; i<n_children; i++)
					{
						delete children[i];
					}
				}
			}
		};

		///member function for determining where to put data.
		virtual bool is_data_valid(Box<dim,T> const &box, const data_t &data) const {return false;}
		Node* root = NULL;

		///vector for storing data in a contiguous array.
		std::vector<data_t> _data;

		//DIVISION
		void divide_multiple_data(Node* node)
		{
			//check if division is allowed
			if (node==NULL){return;}
			if (node->children[0]==NULL){return;}
			if (node->data_idx==NULL){return;}

			//find box center for constructing bounding boxes of children
			Point<dim,T> _center = node->bbox.center();
			
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

			//check if division is allowed
			if (node==NULL){return;}
			if (node->children[0]!=NULL){return;}

			// std::cout << "--- DIVIDE ---\n";

			//find box center for constructing bounding boxes of children
			Point<dim,T> _center = node->bbox.center();
			
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
					// std::cout << "check data " << j << " with child " << i << std::endl;

					//check if data can be passed to child.
					if (is_data_valid(node->children[i]->bbox, _data[node->data_idx[j]]))
					{
						//insert data
						// std::cout << "move data " << j << " to child " << i << " at index " << node->children[i]->cursor << std::endl;
						data_passed[j] = true;
						node->children[i]->data_idx[node->children[i]->cursor] = node->data_idx[j];
						node->children[i]->cursor += 1;
					}
				}
			}

			
		}

		void divide(Node* node)
		{
			if(multiple_data){divide_multiple_data(node);}
			else{divide_single_data(node);}

			//free memory
			delete[] node->data_idx;
			node->data_idx = NULL;
		}

		
		//INSERTION
		bool recursive_insert(Node* node, const size_t &idx)
		{
			//check if current node is divided
			if (node->children[0]==NULL)
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
				}
			}
			
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

			return false;
		}


		//FIND NODES
		bool recursive_find(Node const* node, const data_t &val, size_t &idx) const
		{
			if (node->children[0]==NULL)
			{
				//not divided
				for (size_t j=0; j<node->cursor; j++)
				{
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
		const Node* getnode( const Node* node, const Point<dim,T> &point ) const
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

		const Node* getnode( const Point<dim,T> &point ) const
		{
			if (root->bbox.contains(point)) {return getnode(root, point);}
			return NULL;
		}

	
	public:
		BasicOctree() {root = new Node(NULL, Point<dim,T>{0,0,0}, Point<dim,T> {1,1,1}); }
		BasicOctree(const Box<dim,T> &bbox) {root = new Node(NULL, bbox.low(), bbox.high());}
		~BasicOctree() {delete root;}

		Box<dim,T> bbox() const {return root->bbox;}

		///return index of data.
		size_t find(const data_t &val) const
		{
			size_t idx;
			if (recursive_find(root, val, idx)) {return idx;}
			return (size_t) (-1);
		}

		bool find(const data_t &val, size_t &idx) const
		{
			return recursive_find(root, val, idx);
		}


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
	template <int dim=3, typename T=double, size_t n_data=4>
	class PointOctree : public BasicOctree<Point<dim,T>, dim, false, n_data, T>
	{
	public:
		PointOctree() : BasicOctree<Point<dim,T>, dim, false, n_data, T>() {}
		PointOctree(const Box<dim,T> &bbox) : BasicOctree<Point<dim,T>, dim, false, n_data, T>(bbox) {}
	private:
		bool is_data_valid(Box<dim,T> const &box, Point<dim,T> const &data) const override {return box.contains(data);} 
	};
}

