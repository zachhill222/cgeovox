#pragma once

#include "util/box.hpp"
#include "util/point.hpp"
#include "util/octree.hpp"
#include "mesh/linear_elements.hpp"

// #include <armadillo>

#include <sstream>
#include <iostream>
#include <fstream>

namespace gv::mesh
{
	template <typename T=double>
	using Point_t = gv::util::Point<3,T>;

	template <typename T=double>
	struct VoxelNode
	{
		//structure information
		static const int n_children = 8;
		static const int n_points = 8;

		//nodes
		size_t point_idx[n_points];
		
		//tree structure
		VoxelNode<T>* children[n_children] {NULL};
		const VoxelNode<T>* parent = NULL;

		bool is_divided = false;
		int depth = 0;

		//mesh information
		bool in_domain = true;

		//construction and destruction
		VoxelNode(const VoxelNode<T>* parent, int depth) : parent(parent), depth(depth) {}
		~VoxelNode()
		{
			for (int i=0; i<n_children; i++)
			{
				if (children[i]!=NULL) {delete children[i];}
			}
		}
	};

	//Mesh with hierarchical basis
	template <typename T=double, typename Node=VoxelNode<T>>
	class OctreeMesh
	{
	protected:
		//root node, encloses entire mesh.
		Node* root = NULL;

		//storage for mesh nodes (points)
		gv::util::PointOctree<3,T,32> _points;

		//storage for tracking the hierarchy level of each basis function
		std::vector<int> basis_function_depth;

		///check if a node (voxel) contains a particular point
		inline bool is_in_node(const Node* node, const Point_t<T> &point) const {return _points[node->point_idx[0]]<=point and point<=_points[node->point_idx[7]];}

		///helper function to find all nodes at the specified depth that contain _points[idx].
		void recursive_find_all(const Node* node, const size_t &idx, const int &depth, std::vector<const Node*> &nodelist) const;

		///find all nodes at the specified depth that contain _points[idx].
		std::vector<const Node*> find_all(const size_t &idx, const int &depth) const;

		//divide specified node at its center
		void divide_node(Node* node);
		
		//print element structure in vtk format: <number of nodes> <node numbers ...>
		void print(Node* node, std::stringstream &ss) const;

		///helper function to get first leaf node that contains/encloses the specified point. intended to be used on interior points.
		Node* getnode( Node* node, const Point_t<T> &point );

		///helper function to get first node at specified depth that contains/encloses the specified point. inteded to be used on interior points.
		Node* getnode( Node* node, const Point_t<T> &point, int depth );

		///get first leaf node that contains/encloses the specified point. intended to be used on interior points.
		Node* getnode( const Point_t<T> &point )
		{
			if (is_in_node(root,point)) {return getnode(root, point);}
			return NULL;
		}

		///get first leaf node that contains/encloses the specified point. intended to be used on interior points.
		const Node* getnode( const Point_t<T> &point ) const
		{
			if (is_in_node(root,point)) {return getnode(root, point);}
			return NULL;
		}


		///get first node at specified depth that contains/encloses the specified point. intended to be used on interior points.
		Node* getnode( const Point_t<T> &point, int depth )
		{
			if (is_in_node(root,point)) {return getnode(root, point, depth);}
			return NULL;
		}

		///get first node at specified depth that contains/encloses the specified point. intended to be used on interior points.
		const Node* getnode( const Point_t<T> &point, int depth ) const
		{
			if (is_in_node(root,point)) {return getnode( root, point, depth);}
			return NULL;
		}


		///helper function to count the number of leaves (elements)
		size_t recursive_count_elements( const Node* node) const;

	public:
		OctreeMesh() : _points(Point_t<T> {0,0,0}, Point_t<T> {1,1,1}) {root = new Node(NULL, 0);} 
		OctreeMesh(const gv::util::Box<3,T> &bbox) : _points(bbox)
		{
			root = new Node(NULL, 0);
			for (int j=0; j<root->n_points; j++)
			{
				_points.push_back(bbox[j]);
				root->point_idx[j] = (size_t) j;
				basis_function_depth.push_back(0);
			}
		}
		~OctreeMesh() {delete root;}

		///access mesh nodes
		inline const Point_t<T>& operator[](const size_t &idx) const {return _points[idx];}
		
		///access number of mesh nodes
		inline size_t nNodes() const {return _points.size();}

		///reserve space for nodes before dividing
		inline void reserve(size_t size){_points.reserve(size);}

		///split voxel at its center
		void divide(const Point_t<T> point)
		{
			Node* node = getnode(point);
			if (node!=NULL)
			{
				divide_node(node);
			}
		}

		///count number of voxels
		inline size_t nElements() const {return recursive_count_elements(root);}

		///Get voxel element support for a basis function. Note that the voxels use pointers to the elements of _points[].
		std::vector<Voxel<T>> get_support(const size_t &idx) const; //TODO: remove explicit call to Voxel<T> when adding other types of element support.


		///Assemble mass and stiffness matrices.
		// void make_matrices( arma::SpMat<T> &M, arma::SpMat<T> &A) const;
		


		///print to ostream in vtk format
		void vtkprint(std::ostream &stream) const;

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


	////////////////////////////////////////////////////////////////////////////
	//////// IMPLEMENTATION OF OctMesh METHODS /////////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	template <typename T, typename Node>
	void OctreeMesh<T,Node>::recursive_find_all(const Node* node, const size_t &idx, const int &depth, std::vector<const Node*> &nodelist) const
		{
			if (node->depth == depth)
			{
				for (int j=0; j<node->n_points; j++)
				{
					if (node->point_idx[j] == idx)
					{
						nodelist.push_back(node);
						return;
					}
				}
			}
			else if (node->depth < depth and node->is_divided)
			{
				for (int i=0; i<node->n_children; i++)
				{
					if (is_in_node(node->children[i], _points[idx])) {recursive_find_all(node->children[i], idx, depth, nodelist);}	
				}
			}
		}


		template <typename T, typename Node>
		std::vector<const Node*> OctreeMesh<T,Node>::find_all(const size_t &idx, const int &depth) const
		{
			std::vector<const Node*> result;
			result.reserve(8);
			recursive_find_all(root, idx, depth, result);
			return result;
		}


		template <typename T, typename Node>
		void OctreeMesh<T,Node>::divide_node(Node* node)
		{
			//check if division is allowed
			if (node==NULL){return;}
			if (node->is_divided){return;}
			if (not node->in_domain){return;}


			//check if desired center is in the interior of the voxel
			Point_t<T> center = 0.5 * (_points[node->point_idx[0]] + _points[node->point_idx[7]]);

			//add center to list of points
			size_t center_point_idx = _points.size();
			_points.push_back(center);
			basis_function_depth.push_back(node->depth+1);
			
			//make children and pass data
			for (int i=0; i<node->n_children; i++)
			{
				node->children[i] = new Node(node, node->depth+1);

				//assign node indices to children
				gv::util::Box<3,T> bbox(_points[node->point_idx[i]], center);
				for (int j=0; j<node->n_points; j++)
				{
					if (bbox[j]==center)
					{
						node->children[i]->point_idx[j] = center_point_idx;
					}
					else
					{
						size_t idx;
						if (_points.find(bbox[j], idx))
						{
							//point exists
							node->children[i]->point_idx[j] = idx;
						}
						else
						{
							//create point
							Point_t<T> new_point = bbox[j];
							size_t new_point_idx = _points.size();

							_points.push_back(new_point);
							basis_function_depth.push_back(node->depth+1);


							node->children[i]->point_idx[j] = new_point_idx;
						}
					}
				}

				//mark node as divided
				node->is_divided = true;
			}
		}



		template <typename T, typename Node>
		void OctreeMesh<T,Node>::print(Node* node, std::stringstream &ss) const
		{
			if ((not node->is_divided) and node->in_domain)
			{
				ss << node->n_points;
				for (size_t j=0; j<node->n_points; j++) {ss << " " << node->point_idx[j];}
				ss << "\n";
			}
			else
			{
				for (int i=0; i<node->n_children; i++){print(node->children[i], ss);}
			}
		}



		template <typename T, typename Node>
		Node* OctreeMesh<T,Node>::getnode( Node* node, const Point_t<T> &point )
		{
			if (not node->is_divided)
			{
				//in a leaf node
				if (is_in_node(node,point)) {return node;}
			}
			else
			{
				for (int i=0; i<node->n_children; i++)
				{
					if (is_in_node(node->children[i],point)) {return getnode(node->children[i], point);}
				}
			}

			return NULL;
		}


		template <typename T, typename Node>
		Node* OctreeMesh<T,Node>::getnode( Node* node, const Point_t<T> &point, int depth )
		{
			if (not node->is_divided)
			{
				//in a leaf node
				if (node->depth > depth) {return NULL;}
				if (is_in_node(node,point) and node->depth==depth) {return node;}
			}
			else
			{
				for (int i=0; i<node->n_children; i++)
				{
					if (is_in_node(node->children[i],point)) {return getnode(node->children[i], point);}
				}
			}
			return NULL;
		}


		template <typename T, typename Node>
		size_t OctreeMesh<T,Node>::recursive_count_elements( const Node* node) const
		{
			if (not node->is_divided) {return (size_t) node->in_domain;} //only count voxels that are in the domain
			else
			{
				size_t result = 0;
				for (int i=0; i<node->n_children; i++)
				{
					result += recursive_count_elements(node->children[i]);
				}
				return result;
			}
		}



		template <typename T, typename Node>
		void OctreeMesh<T,Node>::vtkprint(std::ostream &stream) const
		{
			size_t nElems = nElements();
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
			buffer << "CELLS " << nElems << " " << 9*nElems << "\n";
			print(root, buffer);
			buffer << "\n";
			stream << buffer.rdbuf();
			buffer.str("");

			//VTK IDs
			buffer << "CELL_TYPES " << nElems << "\n";
			for (size_t i=0; i<nElems; i++) {buffer << "11\n";}
			stream << buffer.rdbuf();
			buffer.str("");

			// //TEMPORARY DATA
			// buffer << "CELL_DATA " << nElems << std::endl;
			// buffer << "SCALARS elemMarkers integer\n";
			// buffer << "LOOKUP_TABLE default\n";
			// for (size_t i=0; i<nElems; i++){
			// 	buffer << i << "\n";
			// }
			// buffer << "\n";

			// stream << buffer.rdbuf();
			// buffer.str("");

			// //TEMPORARY DATA
			// buffer << "POINT_DATA " << _points.size() << std::endl;
			// buffer << "SCALARS basis_level integer\n";
			// buffer << "LOOKUP_TABLE default\n";
			// for (size_t i=0; i<_points.size(); i++){
			// 	buffer << basis_function_depth[i] << "\n";
			// }
			// buffer << "\n";

			// stream << buffer.rdbuf();
			// buffer.str("");
		}



		template <typename T, typename Node>
		std::vector<Voxel<T>> OctreeMesh<T,Node>::get_support(const size_t &idx) const
		{
			//get nodes
			std::vector<const Node*> support_nodes = find_all(idx, basis_function_depth[idx]);


			std::vector<Voxel<T>> result;
			result.reserve(support_nodes.size());

			for (size_t i=0; i<support_nodes.size(); i++)
			{
				const Node* node = support_nodes[i];
				Voxel<T> voxel {
							&_points[node->point_idx[0]],\
							&_points[node->point_idx[1]],\
							&_points[node->point_idx[2]],\
							&_points[node->point_idx[3]],\
							&_points[node->point_idx[4]],\
							&_points[node->point_idx[5]],\
							&_points[node->point_idx[6]],\
							&_points[node->point_idx[7]]\
						};

				result.push_back(voxel);
				std::cout << i << ": ";
				result[i].print();
			}
			return result;
		}


		// template <typename T, typename Node>
		// void OctreeMesh<T,Node>::make_matrices( arma::SpMat<T> &M, arma::SpMat<T> &A) const
		// {
		// 	arma::umat locations(2,)
		// }

	///convert a generic octree data structure into a mesh to view its structure
	template <typename Octree_t>
	void view_octree_as_vtk(const Octree_t &octree, const std::string outfile="octree_structure.vtk")
	{
		
	}
}