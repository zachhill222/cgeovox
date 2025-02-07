#pragma once

#include "util/box.hpp"
#include "util/point.hpp"
#include "util/octree.hpp"
// #include "mesh/linear_elements.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

namespace gv::mesh
{
	///Basic octree class, can be used in d=2 or d=3 (and possibly others). A octree that will be used should inherit from this class and define the member function is_data_valid().
	template <typename T>
	class OctreeMesh
	{
	protected:
		static const int n_children = 8;
		static const int n_points = 8;
		struct Node
		{
			int depth = 0;
			size_t* point_idx = NULL;
			Node* children[n_children] {NULL};
			const Node* parent = NULL;
			bool in_domain = true;
			bool is_divided = false;
			Node(const Node* parent, int depth) : depth(depth), parent(parent) {point_idx = new size_t[n_points];}
			~Node()
			{
				if (point_idx!=NULL){delete[] point_idx;}
				if (children[0]!=NULL)
				{
					for (int i=0; i<n_children; i++)
					{
						delete children[i];
					}
				}
			}
		};


		Node* root = NULL;

		///vector for storing data in a contiguous array.
		// std::vector<gv::util::Point<3,T>> _points;
		gv::util::PointOctree<3,T,32> _points;

		///check if a node (voxel) contains a particular point
		bool is_in_node(const Node* node, const gv::util::Point<3,T> &point) const
		{
			return _points[node->point_idx[0]]<=point and point<=_points[node->point_idx[7]];
		}

		//FIND NODES
		bool recursive_find(Node const* node, const gv::util::Point<3,T> &point, size_t &idx) const
		{
			for (size_t j=0; j<n_points; j++)
			{
				if (point==_points[node->point_idx[j]])
				{
					idx = node->point_idx[j];
					return true;
				}
			}


			for (int i=0; i<n_children; i++)
			{
				if (node->children[i]!=NULL)
				{
					if (is_in_node(node->children[i], point))
					{
						return recursive_find(node->children[i], point, idx);
					}
				}
			}

			return false;
		}

		//DIVISION
		void divide_node(Node* node, const gv::util::Point<3,T> &center)
		{
			//check if division is allowed
			if (node==NULL){return;}
			if (node->is_divided){return;}
			if (not node->in_domain){return;}


			//check if desired center is in the interior of the voxel
			if (not (_points[node->point_idx[0]]<center and center < _points[node->point_idx[7]])) {return;}

			//add center to list of points
			size_t center_point_idx = _points.size();
			_points.push_back(center);
			// std::cout << _points.size() << "/" << _points.capacity() << std::endl;
			
			//make children and pass data
			for (int i=0; i<n_children; i++)
			{
				node->children[i] = new Node(node, node->depth+1);
				// std::cout << "created child " << i << " at depth " << node->children[i]->depth << " with parent " << node << std::endl;

				//assign node indices to children
				gv::util::Box<3,T> bbox(_points[node->point_idx[i]], center);
				for (int j=0; j<n_points; j++)
				{
					// std::cout << "\tadding point " << bbox[j] << " at vertex " << j << std::endl;
					if (bbox[j]==center)
					{
						node->children[i]->point_idx[j] = center_point_idx;
					}
					else
					{
						size_t idx;
						// if (recursive_find(root, bbox[j], idx))
						if (_points.find(bbox[j], idx))
						{
							// std::cout << "\tpoint exists at index " << idx << std::endl;
							//point exists
							node->children[i]->point_idx[j] = idx;
						}
						else
						{
							//create point
							gv::util::Point<3,T> new_point = bbox[j];
							size_t new_point_idx = _points.size();
							// std::cout << "\tnew point at index " << new_point_idx << std::endl;

							_points.push_back(new_point);
							// std::cout << _points.size() << "/" << _points.capacity() << std::endl;


							node->children[i]->point_idx[j] = new_point_idx;

							
						}
					}
				}

				//mark node as divided
				node->is_divided = true;
			}
		}
		


		//PRINT
		void print(Node* node, std::stringstream &ss) const
		{
			if ((not node->is_divided) and node->in_domain)
			{
				ss << n_points;
				for (size_t j=0; j<n_points; j++) {ss << " " << node->point_idx[j];}
				ss << "\n";
			}
			else
			{
				for (int i=0; i<n_children; i++){print(node->children[i], ss);}
			}
		}

		//TREE TRAVERSAL
		Node* getnode( Node* node, const gv::util::Point<3,T> &point )
		{
			if (not node->is_divided)
			{
				//in a leaf node
				if (is_in_node(node,point)) {return node;}
			}
			else
			{
				for (int i=0; i<n_children; i++)
				{
					if (is_in_node(node->children[i],point)) {return getnode(node->children[i], point);}
				}
			}

			return NULL;
		}

		Node* getnode( const gv::util::Point<3,T> &point )
		{
			if (is_in_node(root,point)) {return getnode(root, point);}
			return NULL;
		}

		const Node* getnode( const gv::util::Point<3,T> &point ) const
		{
			if (is_in_node(root,point)) {return getnode(root, point);}
			return NULL;
		}


		size_t recursive_count_elements( const Node* node) const
		{
			if (not node->is_divided) {return (size_t) node->in_domain;} //only count voxels that are in the domain
			else
			{
				size_t result = 0;
				for (int i=0; i<n_children; i++)
				{
					result += recursive_count_elements(node->children[i]);
				}
				return result;
			}
		}

	public:
		OctreeMesh() : _points(gv::util::Point<3,T> {0,0,0}, gv::util::Point<3,T> {1,1,1}) {root = new Node(NULL, 0);} 
		OctreeMesh(const gv::util::Box<3,T> &bbox) : _points(bbox)
		{
			root = new Node(NULL, 0);
			for (int j=0; j<n_points; j++)
			{
				_points.push_back(bbox[j]);
				root->point_idx[j] = (size_t) j;
			}
		}
		~OctreeMesh() {delete root;}

		
		///return index of point in the mesh.
		size_t find(const gv::util::Point<3,T> &point) const
		{
			size_t idx;
			if (recursive_find(root, point, idx)) {return idx;}
			return (size_t)(-1);
		}

		inline gv::util::Point<3,T> operator[](const size_t &idx) const {return _points[idx];}
		inline size_t size() const {return _points.size();}
		inline void reserve(size_t size){_points.reserve(size);}
		void clear()
		{
			_points.clear();
			for (int i=0; i<n_children; i++){delete root->children[i];}
		}


		///split voxel at its center
		void divide_center(const gv::util::Point<3,T> point)
		{
			Node* node = getnode(point);
			if (node!=NULL)
			{
				//split voxel at its center
				gv::util::Point<3,T> center = 0.5*(_points[node->point_idx[0]] + _points[node->point_idx[7]]);
				divide_node(node, center);
			}
		}

		///split voxel at the specified point
		void divide(const gv::util::Point<3,T> point)
		{
			Node* node = getnode(point);
			if (node!=NULL)
			{
				divide_node(node, point);
			}
		}


		///count number of voxels
		size_t count_elements() const {return recursive_count_elements(root);}

		///print to ostream in vtk format
		void vtkprint(std::ostream &stream) const
		{
			size_t nElements = count_elements();
			std::stringstream buffer;

			//HEADER
			buffer << "# vtk DataFile Version 2.0\n";
			buffer << "Mesh Data\n";
			buffer << "ASCII\n\n";
			buffer << "DATASET UNSTRUCTURED_GRID\n";

			//POINTS
			buffer << "POINTS " << _points.size() << " float\n";
			for (size_t i=0; i<_points.size(); i++) { buffer << _points[i] << "\n";}
			buffer << "\n";
			stream << buffer.rdbuf();
			buffer.str("");

			//ELEMENTS
			buffer << "CELLS " << nElements << " " << 9*nElements << "\n";
			print(root, buffer);
			buffer << "\n";
			stream << buffer.rdbuf();
			buffer.str("");

			//VTK IDs
			buffer << "CELL_TYPES " << nElements << "\n";
			for (size_t i=0; i<nElements; i++) {buffer << "11\n";}
			stream << buffer.rdbuf();
			buffer.str("");

			//TEMPORARY DATA
			buffer << "CELL_DATA " << nElements << std::endl;
			buffer << "SCALARS elemMarkers integer\n";
			buffer << "LOOKUP_TABLE default\n";
			for (size_t i=0; i<nElements; i++){
				buffer << i << "\n";
			}
			buffer << "\n";

			stream << buffer.rdbuf();
			buffer.str("");
		}


		///save mesh to file
		void save(const std::string filename) const
		{
			std::ofstream meshfile(filename);

			if (not meshfile.is_open()){
				std::cout << "Couldn't write to " << filename << std::endl;
				meshfile.close();
				return;
			}

			vtkprint(meshfile);
			meshfile.close();
		}
	};

}