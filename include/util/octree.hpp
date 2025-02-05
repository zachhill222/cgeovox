#pragma once


#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <stdexcept>


namespace GeoVox::util{
	///Basic octree class, can be used in d=2 or d=3 (and possibly others). A octree that will be used should inherit from this class and define the member function is_data_valid().
	template <int dim=3, size_t n_data=4, bool multiple_data=false, typename data_t>
	class BasicOctree
	{
	private:
		static const n_children = std::pow(2,dim);
		struct Node
		{
			int data_index = 0; //cursor for inserting data. points to next data to insert.
			// int traversal_index = 0; //cursor for tracking current progress of a tree traversal.
			size_t* data_idx = NULL;
			Node* children[n_children] {NULL};
			Node* parent = NULL;
			Box<dim>* bbox;
			Node(Node* parent, Box<dim>* bbox) : parent(parent), bbox(bbox)
			~Node() {data_idx = new size_t[n_data];}
			{
				delete bbox;
				if (data_idx!=NULL){delete[] data_idx;}
				if (children[i]!=NULL)
				{
					for (int i=0; i<n_children; i++)
					{
						delete children[i];
					}
				}
			}
		}

		///member function for determining where to put data.
		virtual bool is_data_valid(Box<dim> const &box, data_t const &data){return false;}
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
			Point<dim> _center = node->bbox.center();
			
			//make children and pass data
			for (int i=0; i<n_children; i++)
			{
				node->children[i] = new Node(node, new Box<dim>(node->box[i], center));
				//pass data indices to children
				for (int j=0; j<n_data; j++)
				{	
					//check if end of data has been reached.
					if (node->data_idx[j]==NULL){break;}

					//check if data can be passed to child.
					if (is_data_valid(node->children[i]->bbox, _data[node->data_idx[j]]))
					{	
						//insert data
						node->children[i]->data_idx[node->children[i]->data_index] = node->data_idx[j];
						node->children[i]->data_index += 1;
					}
				}
			}

			//free memory
			delete[] node->data_idx;
		}

		void divide_single_data(Node* node)
		{
			//check if division is allowed
			if (node==NULL){return;}
			if (node->children[0]==NULL){return;}
			if (node->data_idx==NULL){return;}

			//find box center for constructing bounding boxes of children
			Point<dim> _center = node->bbox.center();
			
			//initialize array for ensuring only one copy of any data is passed
			bool data_passed[n_data] {false;}

			//make children and pass data
			for (int i=0; i<n_children; i++)
			{
				node->children[i] = new Node(node, new Box<dim>(node->box[i], center));
				//pass data indices to children
				for (int j=0; j<n_data; j++)
				{	
					//check if end of data has been reached.
					if (node->data_idx[j]==NULL){break;}
					if (data_passed[j]){continue;} //this data was already passed
					data_passed[j] = true;

					//check if data can be passed to child.
					if (is_data_valid(node->children[i]->bbox, _data[node->data_idx[j]]))
					{
						//insert data
						node->children[i]->data_idx[node->children[i]->data_index] = node->data_idx[j];
						node->children[i]->data_index += 1;
					}
				}
			}

			//free memory
			delete[] node->data_idx;
		}

		void divide(Node* node)
		{
			if(multiple_data){divide_multiple_data(node);}
			else(divide_single_data(node);)
		}

		
		//INSERTION
		void recursive_insert(Node* node, size_t &idx)
		{
			//check if data is valid
			if (not is_data_valid(node->bbox, _data[idx])){return;}

			//check if current node is divided
			if (node->children[0]==NULL)
			{
				//node is not divided
				//check if current node has room for more data
				if (node->data_index < n_data)
				{
					node->data_idx[node->data_index] = idx;
					node->data_index += 1;
					return;
				}

				//node must divide
				divide(node);

				//call data insertion on same node to pass to children branch
				recursive_insert(node, idx);
			}
			else
			{
				//node is divided
				for (int i=0; i<n_children; i++)
				{
					recursive_insert(node->children[i], idx);
					if (!multiple_data){return;}
				}
			}
		}


		//FIND NODES
		size_t recursive_find(Node const* node, const data_t &val) const
		{
			if (node->children[i]==NULL)
			{
				//not divided
				for (int j=0; j<node->data_index; j++)
				{
					if (val==_data[node->data_idx[j]]){return node->data_idx[j];}
				}
			}
			else
			{
				for (int i=0; i<n_children; i++)
				{
					if (is_data_valid(node->children[i]->bbox, val)){return recursive_find(node->children[i], val);}
				}
			}

			throw std::runtime_error("value not found.")
		}

		bool recursive_contains(Node const* node, const data_t &val) const
		{
			if (node->children[i]==NULL)
			{
				//not divided
				for (int j=0; j<node->data_index; j++)
				{
					if (val==_data[node->data_idx[j]]){return true;}
				}
			}
			else
			{
				for (int i=0; i<n_children; i++)
				{
					if (is_data_valid(node->children[i]->bbox, val)){return recursive_contains(node->children[i], val);}
				}
			}
		}


		//TREE TRAVERSAL
		// Node* next_node(Node const* node, const int next_child)
		// {

		// 	if (node->children[0]==NULL)
		// 	{
		// 		//node is not divided. check what index of the parent current node is.
		// 		if (node->parent==NULL){return NULL;} //tree has only the root node.
		// 		for (int i=0; i<n_children; i++)
		// 		{
		// 			if (node->parent->children[i] == node)
		// 			{
		// 				if (i==n_children-1)
		// 				{
		// 					//node is last child of parent
		// 					return next_node(node->parent->parent)
		// 				}
		// 			}
		// 		}
		// 	}
		// 	else
		// 	{
		// 		//node is divided.
		// 		return node->children[0];
		// 	}
		// }

	
	public:
		BasicOctree() {root = new Node(NULL, new Box);}
		BasicOctree(const Box &bbox) {root = new Node(NULL, &bbox)}
		~BasicOctree() {delete root;}

		void insert(const data_t &val)
		{
			if (recursive_contains(root, val)){return;}
			_data.push_back(val);
			recursive_insert(root, val);
		}

		size_t find(const data_t &val) const {return recursive_find(root, val);}

		bool contains(const data_t &val) const {return recursive_contains(root, val);}

		data_t operator[](const size_t &idx) const {return _data[idx];}

		// void sort_data()
		// {

		// }


	}






}

