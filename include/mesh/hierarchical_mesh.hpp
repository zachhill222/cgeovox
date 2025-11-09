#pragma once

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>

#include <cassert>
#include <cstring>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>

#include <mutex>
#include <shared_mutex>
#include <memory>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// A container for tracking the node information.
	/// Usually Point_t = gv::util::Point<3,double>, but 2-D meshes or different precisions are allowed.
	/// If the mesh has more than one element type, should be the maximum number of nodes required to define any of the used element types.
	/// This allows the nodes to be stored in a contiguous array as they all will require the same amount of memory. For a hexahedral mesh,=8.
	/////////////////////////////////////////////////
	template <typename Point_t>
	struct Node {
		/// The location of this node in space.
		Point_t vertex;

		/// Track the boundary faces this node belongs to
		std::vector<size_t> boundary_faces;

		/// Which elements this node belongs to. The number of elements cannot be known ahead of time, especially when a mesh is allowed to be refined.
		std::vector<size_t> elems;

		/// Constructor when the vertex is known.
		Node(const Point_t &coord, size_t nBoundary=0, size_t nElems=0) : vertex(coord), boundary_faces(nBoundary), elems(nElems) {}

		/// Default constructor
		Node() : vertex(), boundary_faces(), elems() {}
	};

	/// Equality check for mesh nodes.
	template <typename Point_t>
	bool operator==(const Node<Point_t> &A, const Node<Point_t> &B) {return A.vertex==B.vertex;}

	/////////////////////////////////////////////////
	/// A container for storing the nodes in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template <typename Node_t, int dim=3, int n_data=64, Float T=double>
	class NodeOctree : public gv::util::BasicOctree_Point<Node_t, dim, n_data, T>
	{
	public:
		using Data_t = Node_t;
		NodeOctree() : //if bounding box is unknown ahead of time
			gv::util::BasicOctree_Point<Node_t, dim, n_data, T>(1024) {}

		NodeOctree(const gv::util::Box<dim,T> &bbox, const size_t capacity=1024) :
			gv::util::BasicOctree_Point<Data_t, dim, n_data, T>(bbox, capacity) {}

	private:
		bool is_data_valid(const gv::util::Box<dim,T> &box, const Data_t &data) const override {return box.contains(data.vertex);}
	};

	/////////////////////////////////////////////////
	/// A simple struct to pass any meshing options. Specified here so that it is easy to extend in the future.
	/////////////////////////////////////////////////
	struct HierarchicalMeshOptions {
		bool useGreedy = false;
	};


	/////////////////////////////////////////////////
	/// This class defines a topological mesh that supports different element types. This class only tracks the topology of the mesh and topological operations.
	/// Points can be stored using various precisions (e.g., T=float, double, long double). 
	/// There is no requirement or guarentee that the elements are conforming or that they do not overlap.
	/// Be aware that overlapping elements (in space) are not topologically overlapping if they do not share a node and could have the same color.
	///
	/// @tparam T The precision that the vertices are stored in. It may be completely unnecessary to store the vertices in double precision for some meshes.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template <Float T=double>
	class HierarchicalMesh
	{
	public:
		//common typedefs
		template <int n>
		using Index         = gv::util::Point<n,size_t>;
		using Point_t       = gv::util::Point<3,T>;
		using Node_t        = Node<Point_t>;
		using NodeList_t    = NodeOctree<Node_t, 3, 64, T>;
		using Box_t         = gv::util::Box<3,T>;
		using Element_t     = Element;
		using ElementList_t = std::vector<Element_t>;

		//options
		HierarchicalMeshOptions  opts {false};

	protected:
		ElementList_t          _elements;        //track element types and which block of _elem2node belongs to each element
		NodeList_t             _nodes;           //reference to the node list. use this reference so that the nodes can refer to the nodes of some other mesh.
		std::vector<size_t>    _colorCount;      //track how many elements belong to each color group. The index is the color.
		
		bool                   _boundary_is_computed = false; //track if the boundary needs to be re-computed
		std::vector<Element_t> _boundary;                     //storage for the boundary faces

	private:
		std::unordered_set<size_t>	_element_refine_list; //indices of elements that are to be refined

		mutable std::shared_mutex   _rw_mtx_el_ref_list; //mutex for _element_refine_list
		mutable std::shared_mutex   _rw_mtx_color;       //mutex for _colorCount
		mutable std::shared_mutex   _rw_mtx_elements;    //mutex for _elements

		size_t getValidElementColor(const size_t elem_idx, const std::vector<size_t> &colorCountSnapshot) const;
	public:
		/// Default constructor.
		HierarchicalMesh() : _elements(),_nodes() {}


		/////////////////////////////////////////////////
		/// Constructor when the maximum extents of the mesh are known ahead of time but the elements will be constructed later.
		///
		/// @param bbox The maximum extents of the mesh.
		/////////////////////////////////////////////////
		HierarchicalMesh(const Box_t& bbox) : _elements(), _nodes(bbox), _colorCount() {}


		/////////////////////////////////////////////////
		/// Constructor for a uniform mesh of voxels in 3D.
		///
		/// @param bbox The region to be meshed.
		/// @param N The number of elements along each coordinate axis.
		/////////////////////////////////////////////////
		HierarchicalMesh(const Box_t& domain, const Index<3>& N) : _elements(),  _nodes(domain), _colorCount() {
			//reserve space
			_nodes.reserve((N[0]+1) * (N[1]+1) * (N[2]+1));
			_elements.reserve(N[0]*N[1]*N[2]);

			
			//construct the mesh
			Point_t H = domain.sidelength() / Point_t(N);
			for (size_t i=0; i<N[0]; i++) {
				for (size_t j=0; j<N[1]; j++) {
					for (size_t k=0; k<N[2]; k++) {
						//define element extents
						Point_t low  = domain.low() + Point_t{i,j,k} * H;
						Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;
						Box_t   elem  {low, high};
					
						//assemble the list of vertices
						std::vector<Point_t> element_vertices(8);
						for ( int l=0; l<8; l++) {element_vertices[l] = elem.voxelvertex(l);}

						//put the element into the mesh
						insert_element(element_vertices, 11);
					}
				}
			}

			compute_conformal_boundary();
		}


		/////////////////////////////////////////////////
		/// Constructor for a uniform mesh of pixels in 2D.
		///
		/// @param bbox The region to be meshed.
		/// @param N The number of elements along each coordinate axis.
		/////////////////////////////////////////////////
		HierarchicalMesh(const gv::util::Box<2,T>& domain, const Index<2>& N) :  _elements(),  _nodes(domain), _colorCount(10) {
			Point_t low, high;
			low[0]  = domain.low()[0]; low[1] = domain.low()[1]; low[2]=-1.0;
			high[0] = domain.high()[0]; high[1] = domain.high()[1]; high[2]=1.0;

			_nodes.set_bbox(Box_t{low,high});

			using Point2 = gv::util::Point<2,T>;

			//reserve space
			_nodes.reserve((N[0]+1) * (N[1]+1));
			_elements.reserve(N[0]*N[1]);
			
			//construct the mesh
			Point2 H = domain.sidelength() / Point2(N);
			for (size_t i=0; i<N[0]; i++) {
				for (size_t j=0; j<N[1]; j++) {
					//define element extents
					Point2 low  = domain.low() + Point2{i,j} * H;
					Point2 high = domain.low() + Point2{i+1,j+1} * H;
					gv::util::Box<2,T>   elem  {low, high};
				
					//assemble the list of vertices
					Point_t element_vertices[4];
					for ( int l=0; l<4; l++) {
						element_vertices[l][0] = elem.voxelvertex(l)[0];
						element_vertices[l][1] = elem.voxelvertex(l)[1];
						element_vertices[l][2] = 0.0;
					}

					//put the element into the mesh
					insert_element(element_vertices, 8);
				}
			}

			compute_conformal_boundary();
		}

		/////////////////////////////////////////////////
		/// Get the total number of elements in the mesh.
		///
		/// @param activeOnly When set to true, count the number of active elements. When set to false, count all elem
		/////////////////////////////////////////////////
		size_t nElems(const bool activeOnly=true) const {
			if (!activeOnly) {return _elements.size();}
			
			size_t count = 0;
			#pragma omp parallel for reduction(+:count)
			for (size_t i=0; i<_elements.size(); i++) {
				if (_elements[i].children.size()==0) {count+=1;}
			}
			return count;
		}


		/////////////////////////////////////////////////
		/// Get the total number of nodes in the mesh.
		/////////////////////////////////////////////////
		inline size_t nNodes() const {return _nodes.size();} //number of nodes in the mesh


		/////////////////////////////////////////////////
		/// Get the number of elements in each color group. colorCount()[i] is the number of elements colored by colors()[i].
		/////////////////////////////////////////////////
		inline std::vector<size_t> colorCount() const {return _colorCount;}


		/////////////////////////////////////////////////
		/// Process the elements in the refinement list
		/////////////////////////////////////////////////
		void process_refinement();


		/////////////////////////////////////////////////
		/// Check number of elements that need to be refined
		/////////////////////////////////////////////////
		inline size_t n_elems_to_process() const {return _element_refine_list.size();}


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		void get_element_neighbors(const size_t elem_idx, std::vector<size_t> &neighbors) const;


		/////////////////////////////////////////////////
		/// A method to get the active boundary faces that are also faces of the specified element.
		///
		/// @param elem_idx The index of the requested face (i.e., _boundary[elem_idx]).
		/// @param faces A reference to an existing vector where the result will be stored (via faces.push_back()).
		/////////////////////////////////////////////////
		void get_boundary_faces(const size_t elem_idx, std::vector<size_t> &faces) const;


		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified element.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get only the active descendents
		/////////////////////////////////////////////////
		void get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;
		

		/////////////////////////////////////////////////
		/// A method to get the descendent faces of the specified boundary face.
		///
		/// @param face_idx The index of the requested element (i.e., _boundary[face_idx]).
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get only the active descendents
		/////////////////////////////////////////////////
		void get_boundary_face_descendents(const size_t face_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;
		
		
		/////////////////////////////////////////////////
		/// A method to get a list of all element types and how many elements of each type exist.
		///
		/// @param vtkID A reference to an existing vector where the element types (vtk identifiers) will be stored.
		/// @param count A reference to an existing vector where the count for each element type will be stored
		/////////////////////////////////////////////////
		void get_all_elem_types(std::vector<int> &vtkID, std::vector<size_t> &count) const;
		

		/////////////////////////////////////////////////
		/// A method to get a list of all elements that are of the specified type.
		///
		/// @param vtkID The vtk identifier of the requested element type.
		/// @param elements A reference to an existing vector where the element indices will be stored.
		/////////////////////////////////////////////////
		void get_element_type_group(const int vtkID, std::vector<size_t> &elements) const;

		
		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		void insert_element(Element_t &ELEM);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void insert_element(const std::vector<Point_t> &vertices, const int vtkID);


		/////////////////////////////////////////////////
		/// A method to mark an element to be split/refined. The element that is split will have the new elements added as children, and the new elements that are created
		/// will have the element that was split as a parent. The new elements are of the same type as the original.
		/// New nodes will most likely be created and old nodes updated during this process.
		/// For certain elements (i.e., voxels or hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		/// If the specified element has already been split and re-joined (i.e., the children exist), then the children are simply activated and no new elements are created in memory.
		/// If this _elements[elem_idx].is_active is false, then the method returns without making any changes.
		///
		/// An element is a valid refinement target if it is currently active.
		/// If _elements[elem_idx].is_active is false, then the method immediately returns.
		/// If _elements[elem_idx].children is populated, then the children are activated and re-colored (while the specified element is deactivated).
		/// If _elements[elem_idx].children is empty, then the creation of the children elements is delayed until they are needed and then run in parallel. This is done by inserting elem_idx
		///    into _element_refine_list.
		///
		/// If elem_idx>=_elements.size(), then _element_refine_list is processed (if it is not empty) and then the elem_idx element is re-examined. If it is still out of bounds,
		///    an exception is thrown.
		///
		/// @param elem_idx The element to be split.
		///
		/// @todo Add support for more element types.
		/////////////////////////////////////////////////
		void split_element(const size_t elem_idx);
		

		/////////////////////////////////////////////////
		/// A method to join/unrefine previously split elements. If _elements[elem_idx] exists, then all of the descendents of that element are de-activated and the element is activated.
		/// The de-activated elements are not deleted. The element is re-colored.
		///
		/// @param elem_idx The element whose descendents are to be joined.
		/////////////////////////////////////////////////
		void join_descendents(const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Find a valid color with the least count for the give element
		///
		/// @parem elem_idx The element to find the color for
		/////////////////////////////////////////////////
		size_t findColorBalanced(const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Color or re-color the active elements of the mesh. The elements will be colored such that no active elements that share a node will have the same color.
		/////////////////////////////////////////////////
		void recolor();


		/////////////////////////////////////////////////
		/// Color or re-color a specific element. The color is chosen to be the existing color with the least count if possible.
		/// Otherwise a new color is created.
		///
		/// @param elem_idx The element to be re-colored.
		/////////////////////////////////////////////////
		void recolor(const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Check if the coloring is valid
		/////////////////////////////////////////////////
		bool colors_are_valid() const;


		/////////////////////////////////////////////////
		/// Compute the boundary elements. The mesh must be in a conformal state (i.e., the coarsest mesh).
		/////////////////////////////////////////////////
		void compute_conformal_boundary();


		/////////////////////////////////////////////////
		/// Return the boundary as a mesh that references the nodes of this mesh.
		/////////////////////////////////////////////////
		HierarchicalMesh<T> boundary_mesh() const;


		/////////////////////////////////////////////////
		/// Print the node locations and element connectivity to the output stream. Data can be appended to the stream after this is called.
		/// When saving information (e.g., a solution to a PDE define on this mesh), this will initialize the mesh and then the data can be appended.
		///
		/// @param os The output stream.
		/////////////////////////////////////////////////
		void print_topology_ascii_vtk(std::ostream &os=true) const;


		/////////////////////////////////////////////////
		/// Print the node locations and element connectivity to the output file in binary format. Data can be appended to the file after this is called.
		/// When saving information (e.g., a solution to a PDE define on this mesh), this will initialize the mesh and then the data can be appended.
		///
		/// @param filename The file to write to.
		/////////////////////////////////////////////////
		void print_topology_binary_vtk(const std::string &filename=true) const;


		/////////////////////////////////////////////////
		/// Print the details of the nodes and elements to the output stream. This includes element colors and which elements each node belongs to.
		/// Due to the way that field data is stored in ASCII VTK format, it will be difficult to append any additional information to a file afterwards.
		///
		/// @param os The output stream.
		/////////////////////////////////////////////////
		void print_mesh_details_ascii_vtk(std::ostream &os=true) const;


		/////////////////////////////////////////////////
		/// Print the details of the nodes and elements to the output legacy vtk binary file. This includes element colors and which elements each node belongs to.
		/// Due to the way that field data is stored in BINARY VTK format, it will be difficult to append any additional information to a file afterwards.
		///
		/// @param file The output output file that already contains the mesh topology in legacy vtk binary format.
		/////////////////////////////////////////////////
		void print_mesh_details_binary_vtk(const std::string &filename=true) const;
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file. If the file already exists, it will be over-written.
		/// Effectively this creates a file stream fs and calls print_topology_ascii_vtk(fs) and then (if include_details=true) calls print_mesh_details_ascii_vtk(fs).
		///
		/// @param filename The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology. This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, const bool include_details=false, const bool use_ascii=false) const;

		/// Friend function to print the mesh information
		template <Float U>
		friend std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<U> &mesh);
	};


	
	template <Float T>
	void HierarchicalMesh<T>::get_element_neighbors(const size_t elem_idx, std::vector<size_t> &neighbors) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t    = typename HierarchicalMesh<T>::Node_t;

		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique nodes
		std::unordered_set<size_t> neighbor_set;

		//loop through the nodes of the current element
		for (size_t n_idx=0; n_idx<ELEM.nNodes; n_idx++) {
			const Node_t &NODE = _nodes[ELEM.nodes[n_idx]];

			//loop through the elements of the current node
			for (size_t m=0; m<NODE.elems.size(); m++) {
				const size_t e_idx = NODE.elems[m];

				//only leaf elements can be neighbors
				if (e_idx!=elem_idx and _elements[e_idx].is_active) {
					// neighbors.push_back(e_idx);
					neighbor_set.insert(e_idx);
				}
			}
		}

		//convert the set to the vector
		neighbors.assign(neighbor_set.begin(), neighbor_set.end());
	}


	template <Float T>
	void HierarchicalMesh<T>::get_boundary_faces(const size_t elem_idx, std::vector<size_t> &faces) const {
		assert(_boundary_is_computed);

		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t    = typename HierarchicalMesh<T>::Node_t;

		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique nodes
		std::unordered_set<size_t> face_set;

		//loop through the nodes of the face
		for (size_t n_idx=0; n_idx<ELEM.nNodes; n_idx++) {
			const Node_t &NODE = _nodes[ELEM.nodes[n_idx]];

			//loop through the elements of the current node
			for (size_t m=0; m<NODE.boundary_faces.size(); m++) {
				const size_t f_idx = NODE.boundary_faces[m];
				const Element_t FACE = _boundary[f_idx];

				if (FACE.parent==elem_idx) {
					// faces.push_back(f_idx);
					face_set.insert(f_idx);
				}
			}
		}

		faces.assign(face_set.begin(), face_set.end());
	}
	

	template <Float T>
	void HierarchicalMesh<T>::get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		const Element_t &ELEM = _elements[elem_idx];

		//loop through the children
		for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
			const size_t child_idx = ELEM.children[c_idx];
			const Element_t &CHILD = _elements[child_idx];

			//add the relevent children
			if (!activeOnly or CHILD.is_active) {descendents.push_back(child_idx);}

			//recurse if needed
			if (CHILD.children.size()>0) {get_element_descendents(child_idx, descendents, activeOnly);}
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::get_boundary_face_descendents(const size_t face_idx, std::vector<size_t> &descendents, const bool activeOnly) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		const Element_t &FACE = _boundary[face_idx];

		//loop through the children
		for (size_t c_idx=0; c_idx<FACE.children.size(); c_idx++) {
			const size_t child_idx = FACE.children[c_idx];
			const Element_t &CHILD = _boundary[child_idx];

			//add the relevent children
			if (!activeOnly or CHILD.is_active) {descendents.push_back(child_idx);}

			//recurse if needed
			if (CHILD.children.size()>0) {get_element_descendents(child_idx, descendents, activeOnly);}
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::get_all_elem_types(std::vector<int> &vtkID, std::vector<size_t> &count) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];

			//check if the ID of the current element already exists
			auto it = std::find(vtkID.begin(), vtkID.end(), ELEM.vtkID);

			//if the ID of the current element already exists, then increment its count
			if (it!=vtkID.end()) {count[*it] += 1;}
			else {
				vtkID.push_back(ELEM.vtkID);
				count.push_back(1);
			}
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::get_element_type_group(const int vtkID, std::vector<size_t> &elements) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.vtkID == vtkID) {elements.push_back(e_idx);}
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::insert_element(Element_t &ELEM) {
		//add the element to the mesh
		size_t e_idx = _elements.size(); //index of the new element
		ELEM.index   = e_idx;

		//update existing nodes
		for (size_t n=0; n<ELEM.nNodes; n++) {
			size_t node_idx = ELEM.nodes[n];
			_nodes[node_idx].elems.push_back(e_idx);
		}

		//move ELEM to _elements and color
		size_t new_elem_idx = _elements.size();
		_elements.push_back(std::move(ELEM));

		recolor(new_elem_idx);
	}


	template <Float T>
	void HierarchicalMesh<T>::insert_element(const std::vector<typename HierarchicalMesh<T>::Point_t> &vertices, const int vtkID) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//construct the element information
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t    = typename HierarchicalMesh<T>::Node_t;


		//initialize the new element
		Element_t ELEM(vtkID);

		//create new nodes as needed and aggregate their indices
		for (size_t n=0; n<vertices.size(); n++) {
			Node_t NODE(vertices[n]);

			size_t n_idx = (size_t) -1;
			[[maybe_unused]] int flag = _nodes.push_back(NODE, n_idx);
			assert(flag>=0);

			assert(n_idx<_nodes.size());
			ELEM.nodes[n]=n_idx;
		}

		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		insert_element(ELEM);
	}



	template <Float T>
	void HierarchicalMesh<T>::recolor() {
		_colorCount.clear();

		#pragma omp parallel for
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			_elements[e_idx].color = (size_t) -1;
		}


		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			if (_elements[e_idx].is_active) {recolor(e_idx);}
		}
	}


	template <Float T>
	size_t HierarchicalMesh<T>::getValidElementColor(const size_t elem_idx, const std::vector<size_t> &colorCountSnapshot) const {
		//no colors
		if (colorCountSnapshot.empty()) {return 0;}

		//get the active neighbor elements
		std::vector<size_t> neighbors;
		get_element_neighbors(elem_idx, neighbors);

		//decide which colors are allowed
		std::vector<bool> color_allowed(colorCountSnapshot.size(), true);
		for (size_t n_idx=0; n_idx<neighbors.size(); n_idx++) {
			size_t color = _elements[neighbors[n_idx]].color;
			if (color < color_allowed.size()) {color_allowed[color] = false;}
		}

		//check if an existing color can be used
		size_t color;
		for (color=0; color<color_allowed.size(); color++) {
			if (color_allowed[color]) {break;}
			return colorCountSnapshot.size();
		}

		//check if we need to get the optimal color
		if (opts.useGreedy) {return color;}
		
		//get the color with minimal count (approximate due to race conditions)
		for (size_t c_idx=color; c_idx<color_allowed.size(); c_idx++) {
			if (color_allowed[c_idx] and colorCountSnapshot[c_idx]<colorCountSnapshot[color]) {
				color = c_idx;
			}
		}

		return color;
	}


	template <Float T>
	void HierarchicalMesh<T>::recolor(const size_t elem_idx) {
		using Element_t = HierarchicalMesh<T>::Element_t;
		std::unique_lock<std::shared_mutex> write_lock_color(_rw_mtx_color);

		Element_t &ELEM = _elements[elem_idx];
		ELEM.color = getValidElementColor(elem_idx, _colorCount);
		if (ELEM.color < _colorCount.size()) {_colorCount[ELEM.color] += 1;}
		else {_colorCount.push_back(1);}
	}


	template <Float T>
	bool HierarchicalMesh<T>::colors_are_valid() const {
		for (const Element_t& ELEM: _elements) {
			if (!ELEM.is_active) {continue;}

			std::vector<size_t> neighbors;
			get_element_neighbors(ELEM.index, neighbors);
			for (size_t n_idx: neighbors) {
				if (ELEM.color == _elements[n_idx].color) {
					std::cout << "elements " << ELEM.index << " and " << n_idx << " color colision" << std::endl;
					return false;}
			}
		}
		return true;
	}


	template <Float T>
	void HierarchicalMesh<T>::join_descendents(const size_t elem_idx) {
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		//get the active descendents of the element
		std::vector<size_t> descendents;
		get_element_descendents(elem_idx, descendents, true);

		//de-activate the descendents and any boundary faces
		for (size_t e_idx=0; e_idx<descendents.size(); e_idx++) {
			Element_t &ELEM = _elements[descendents[e_idx]];
			ELEM.is_active = false;
			_colorCount[ELEM.color] -= 1;
		}

		//activate the element
		_elements[elem_idx].is_active = true;

		//recolor the element
		recolor(elem_idx);

		//get the descendents of any boundary faces
		std::vector<size_t> boundary_faces;
		get_boundary_faces(elem_idx, boundary_faces);
		for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
			std::vector<size_t> descendents;
			get_boundary_face_descendents(boundary_faces[f_idx], descendents, true);

			//deactivate the descendent boundary elements
			for (size_t d_idx=0; d_idx<descendents.size(); d_idx++) {
				_boundary[descendents[d_idx]].is_active = false;
			}

			//activate the boundary faces of the coarse element
			_boundary[boundary_faces[f_idx]].is_active = true;

			//update the color of the boundary element
			_boundary[boundary_faces[f_idx]].color = _elements[elem_idx].color;
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::split_element(const size_t elem_idx) {
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		//any changes to _elements[elem_idx] are not protected by a unique mutex lock
		//calling split_element(k) for the same value of k in different threads will lead to undefined behavior
		//calling split_element(k) for different values of k in defferent threads is safe
		std::shared_lock<std::shared_mutex> read_lock_elements(_rw_mtx_elements);
		const size_t nElements = _elements.size();

		//if the specified does not exist, process any elements that need to be refined
		if (elem_idx >= nElements) {
			std::shared_lock<std::shared_mutex> read_lock_el_ref(_rw_mtx_el_ref_list);
			bool can_process_elements = !_element_refine_list.empty();
			read_lock_el_ref.unlock();
			if (can_process_elements) {process_refinement();}
		}
		

		//ensure that the element exists
		if (elem_idx >= nElements) {
			throw std::runtime_error("Index " + std::to_string(elem_idx) + " is out of bounds (_elements.size()= " + std::to_string(nElements) + ")");
			return;
		}


		//ensure that the element is active
		Element_t &ELEM = _elements[elem_idx];
		if (!ELEM.is_active) {return;}


		//if the element has been previously refined, activate and re-color its children
		if (!ELEM.children.empty()) {
			ELEM.is_active = false;
			std::unique_lock<std::shared_mutex> write_lock_color(_rw_mtx_color);
			_colorCount[ELEM.color] -= 1;

			//activate and color the children
			for (size_t e_idx: ELEM.children) {
				Element_t &CHILD = _elements[e_idx];
				CHILD.is_active  = true;
				recolor(CHILD.index);
			}

			//update any boundary faces
			std::vector<size_t> boundary_faces;
			get_boundary_faces(ELEM.index, boundary_faces);
			for (size_t i=0; i<boundary_faces.size(); i++) {
				Element_t &FACE = _boundary[boundary_faces[i]];
				FACE.is_active = false;

				assert(!FACE.children.empty());
				for (size_t f_idx : FACE.children) {
					Element_t &CHILDFACE = _boundary[f_idx];
					CHILDFACE.is_active  = true;
					CHILDFACE.color = _elements[CHILDFACE.parent].color; //the parent of a face is the element to which it is a face of (unique for boundary faces)
				}
			}
		}


		//the children of the element must be created. delay this until many elements can be processed in parallel.
		std::unique_lock<std::shared_mutex> write_lock_el_ref(_rw_mtx_el_ref_list);
		_element_refine_list.insert(elem_idx);
	}


	template <Float T>
	void HierarchicalMesh<T>::process_refinement() {
		//This method refines all elements in _element_refine_list by color batches
		//Each batch consists of all elements with a specific color
		//No two elements in the same color batch share a node
		//Each color batch is processed in parallel
		//There are some race conditions when using balanced coloring, so the optimal color may not be chosen for the children elements
		//The chosen color for the children elements is guarenteed to be valid and only new colors are generated when strictly necessary

		using Point_t   = typename HierarchicalMesh<T>::Point_t;
		using Node_t    = typename HierarchicalMesh<T>::Node_t;
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		std::unique_lock<std::shared_mutex> write_lock_el_ref(_rw_mtx_el_ref_list);
		if (_element_refine_list.empty()) {return;}


		//partition the specified that are active by their color
		//if any element has been split already and the children are already stored in the mesh, activate them
		std::vector<std::vector<size_t>> colored_elements_to_split;
		{
			std::shared_lock<std::shared_mutex> read_lock_color(_rw_mtx_color);
			colored_elements_to_split.resize(_colorCount.size());
			for (size_t e_idx: _element_refine_list) {
				assert(e_idx<_elements.size());
				Element_t &ELEM = _elements[e_idx];
				if (!ELEM.is_active) {continue;}

				//element is to be split and the children must be computed
				assert(ELEM.index == e_idx);
				assert(ELEM.color<colored_elements_to_split.size());
				if (ELEM.children.size()==0) {colored_elements_to_split[ELEM.color].push_back(ELEM.index); continue;}
			}
		}

		
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			std::vector<size_t> &this_color_elems = colored_elements_to_split[color];

			if (this_color_elems.size()==0) {continue;} //no elements of this color to split

			std::vector<size_t> child_element_index_start;
			std::vector<size_t> boundary_face_index_start;

			//the indices of where to store each child element must be known ahead of time
			//the local child j of element k will be stored at _elements[child_element_index_start[k] + j]
			const size_t nStartingElements = _elements.size();
			const size_t nStartingBoundary = _boundary.size();

			child_element_index_start.push_back(nStartingElements);
			boundary_face_index_start.push_back(nStartingBoundary);

			size_t nNewElements = vtk_n_children(_elements[this_color_elems[0]].vtkID);
			size_t maxNewNodes = vtk_n_nodes_when_split(_elements[this_color_elems[0]].vtkID);
			size_t nNewBoundaryFaces = 0;

			std::vector<size_t> boundary_faces;
			get_boundary_faces(this_color_elems[0], boundary_faces);
			for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
				nNewBoundaryFaces += vtk_n_children(_boundary[boundary_faces[f_idx]].vtkID);
			}


			//loop through elements and compute child offsets
			for (size_t i=1; i<this_color_elems.size(); i++) {
				const Element_t &CUR_ELEM  = _elements[this_color_elems[i]];
				const Element_t &PREV_ELEM = _elements[this_color_elems[i-1]];

				const size_t prev_children = vtk_n_children(PREV_ELEM.vtkID);
				const size_t prev_index_start = child_element_index_start[i-1];
				child_element_index_start.push_back(prev_index_start + prev_children);
				
				boundary_face_index_start.push_back(nNewBoundaryFaces);

				nNewElements += vtk_n_children(CUR_ELEM.vtkID);
				maxNewNodes  += vtk_n_nodes_when_split(CUR_ELEM.vtkID) - vtk_n_nodes(CUR_ELEM.vtkID);

				boundary_faces.clear();
				get_boundary_faces(this_color_elems[0], boundary_faces);
				for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
					nNewBoundaryFaces += vtk_n_children(_boundary[boundary_faces[f_idx]].vtkID);
				}
			}

			//reserve space and default-initialize the new elements
			_elements.resize(nStartingElements+nNewElements);
			_nodes.reserve(_nodes.size()+maxNewNodes);
			_boundary.reserve(nStartingBoundary+nNewBoundaryFaces);


			//decrement the _colorCount by the number of elements that will be deactivated
			_colorCount[color] -= this_color_elems.size();


			#pragma omp parallel
			{
				//get color count for each thread and initialize storage for any changes
				std::vector<size_t> t_color_count;
			    std::vector<size_t> t_color_delta;
			    
			    {
			        std::shared_lock<std::shared_mutex> read_lock_color(_rw_mtx_color);
			        t_color_count = _colorCount;
			        t_color_delta.resize(t_color_count.size());
			    }


				#pragma omp for
				for (size_t i=0; i<this_color_elems.size(); i++) {
					//create helper references and logical element
					Element_t &ELEM = _elements[this_color_elems[i]];
					assert(ELEM.index == this_color_elems[i]);
					VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(ELEM);

					//initialize storage for the values that will be needed to create the children elements
					std::vector<Point_t> child_vertex_coords;
					std::vector<size_t>  child_node_idx(vtk_n_nodes_when_split(ELEM.vtkID));
					
					//handle parent nodes/vertices that will be re-used
					size_t j;
					for (j=0; j<ELEM.nNodes; j++) {
						child_vertex_coords.push_back(_nodes[ELEM.nodes[j]].vertex);
						child_node_idx[j] = ELEM.nodes[j];
					}

					//get the verticices of the remaining nodes that the children will need
					vtk_elem->split(child_vertex_coords);

					//find any nodes that already exist and collect which nodes need to be added to _nodes
					std::vector<Node_t> nodes_to_create;
					std::vector<size_t> local_node;
					for (;j<child_vertex_coords.size(); j++) {
						Node_t NODE(child_vertex_coords[j]);
						size_t node_idx = _nodes.find(NODE);
						if (node_idx < _nodes.size()) {child_node_idx[j] = node_idx;}
						else {
							nodes_to_create.push_back(NODE);
							local_node.push_back(j); //track which nodes need to be corrected after NODE is inserted into _nodes and its index is known
						}
					}

					//create the nodes. insertion into _nodes must be serial, but the order of insertion does not matter
					for (size_t k=0; k<nodes_to_create.size(); k++) {
						size_t node_idx = (size_t) -1;
						[[maybe_unused]] int flag = _nodes.push_back(nodes_to_create[k], node_idx); //this has a mutex to make it thread-safe
						assert(flag==1); //the node must have been inserted. all elements are of the same color, so no other element will have added this node.
						assert(node_idx<_nodes.size());
						child_node_idx[local_node[k]] = node_idx; //now that the node has been created, track its index to create the children
					}

					//now all nodes have been created to create the children
					ELEM.is_active = false;

					for (size_t k=0; k<vtk_n_children(ELEM.vtkID); k++) {
						//get the indices of the nodes that define child k in the correct order
						std::vector<size_t> childNodes;
						vtk_elem->getChildNodes(childNodes, k, child_node_idx);

						//create the child
						const size_t global_child_index = child_element_index_start[i] + k;
						assert(global_child_index<_elements.size());
						_elements[global_child_index] = Element_t(childNodes, ELEM.vtkID);
						Element_t &CHILD = _elements[global_child_index];
						CHILD.index = global_child_index;
						CHILD.parent = ELEM.index;

						//update the nodes
						for (size_t n=0; n<childNodes.size(); n++) {
							_nodes[childNodes[n]].elems.push_back(global_child_index);
						}

						//update the parent element
						ELEM.children.push_back(global_child_index);

						//color the child
						CHILD.color = getValidElementColor(CHILD.index, t_color_count);
						if (CHILD.color >= t_color_count.size()) {
							//a color must be created
							t_color_count.push_back(1);
							t_color_delta.push_back(1);
						} else {
							t_color_count[CHILD.color] += 1;
							t_color_delta[CHILD.color] += 1;
						}
					}

					//clean up memory
					delete vtk_elem;

					//split any boundary faces
					std::vector<size_t> parent_boundary_faces;
					get_boundary_faces(ELEM.index, parent_boundary_faces);
					for (size_t f_idx=0; f_idx<parent_boundary_faces.size(); f_idx++) {
						#pragma omp critical //TODO: make this thread safe if needed.
						{
							assert(parent_boundary_faces[f_idx]<_boundary.size());
							//ensure that _boundary does not re-size and invalidate references
							while (_boundary.capacity() < _boundary.size() + vtk_n_children(_boundary[parent_boundary_faces[f_idx]].vtkID)) {_boundary.reserve(2*_boundary.capacity());}

							Element_t &FACE = _boundary[parent_boundary_faces[f_idx]];
							FACE.is_active = false;

							VTK_ELEMENT<Point_t>* vtk_face = _VTK_ELEMENT_FACTORY<Point_t>(FACE);
							//find the vertices for the split face
							std::vector<Point_t> child_vertex_coords;
							std::vector<size_t> child_node_idx(vtk_n_nodes_when_split(FACE.vtkID));
							size_t i;
							for (i=0; i<FACE.nNodes; i++) {
								child_vertex_coords.push_back(_nodes[FACE.nodes[i]].vertex);
								child_node_idx[i] = FACE.nodes[i];
							}
							vtk_face->split(child_vertex_coords);

							//get any new nodes
							for (;i<child_vertex_coords.size(); i++) {
								Node_t NODE(child_vertex_coords[i]);
								child_node_idx[i] = _nodes.find(NODE); //the node must have been generated by splitting the element
								assert(child_node_idx[i]<_nodes.size());
							}


							//create the children faces
							for (size_t j=0; j<vtk_n_children(FACE.vtkID); j++) {
								std::vector<size_t> childNodes;
								vtk_face->getChildNodes(childNodes, j, child_node_idx);
								Element_t CHILDFACE(childNodes, FACE.vtkID);
								CHILDFACE.parent = (size_t) -1;

								//determine the child element that this split face is a child of
								for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
									const Element_t &CHILD = _elements[ELEM.children[c_idx]];
									const VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(CHILD);
									for (int k=0; k<vtk_n_faces(CHILD.vtkID); k++) {
										if (CHILDFACE == vtk_elem->getFace(k)) {
											CHILDFACE.parent = CHILD.index;
											CHILDFACE.color  = CHILD.color;
											break;
										}
									}
									delete vtk_elem;
								}

								//add the face to the _boundary
								assert(CHILDFACE.parent<_elements.size());
								CHILDFACE.index = _boundary.size();
								FACE.children.push_back(CHILDFACE.index);
								_boundary.push_back(std::move(CHILDFACE));
							}

							//clean up memory
							delete vtk_face;
						}
					}
				}

				#pragma omp critical
				{
					for (size_t c=0; c<t_color_delta.size(); c++) {
						if (c<_colorCount.size()) {
							_colorCount[c] += t_color_delta[c];
						} else {
							_colorCount.push_back(t_color_delta[c]);
						}
					}
				}
			}
		}
	}

	template <Float T>
	void HierarchicalMesh<T>::compute_conformal_boundary() {
		using Point_t   = typename HierarchicalMesh<T>::Point_t;
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t    = typename HierarchicalMesh<T>::Node_t;

		if (_boundary_is_computed) {return;}

		//create unordered maps to track the count of each face
		std::unordered_map<Element_t, int, ElemHashBitPack> face_count;
		face_count.reserve(8*_elements.size()); //guess at the number of unique faces (exact if all elements are voxels or hexes)

		//loop through all elements and add the faces to the map
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (!ELEM.is_active) {continue;}

			VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(ELEM);

			for (int i=0; i<vtk_n_faces(ELEM.vtkID); i++) {
				Element_t FACE = vtk_elem->getFace(i);
				face_count[FACE] += 1;
			}

			delete vtk_elem;
		}

		//process boundary faces
		_boundary.clear();
		_boundary.reserve(face_count.size()/4);
		size_t b_idx = 0;
		for (const auto& [FACE, count] : face_count) {
			if (count==1) {
				_boundary.push_back(FACE);
				_boundary[b_idx].index = b_idx;

				for (size_t i=0; i<FACE.nNodes; i++) {
					Node_t &NODE = _nodes[FACE.nodes[i]];
					NODE.boundary_faces.push_back(b_idx);
				}

				b_idx += 1;
			}
		}
		_boundary.shrink_to_fit();

		//set flag
		_boundary_is_computed = true;
	}


	template <Float T>
	HierarchicalMesh<T> HierarchicalMesh<T>::boundary_mesh() const {
		using Point_t = typename HierarchicalMesh<T>::Point_t;

		//ensure the boundary is computed
		assert(_boundary_is_computed);

		//initialize boundary mesh
		HierarchicalMesh<T> mesh(_nodes.bbox());

		//add active faces to the boundary mesh
		for (size_t f_idx=0; f_idx<_boundary.size(); f_idx++) {
			const Element_t &FACE = _boundary[f_idx];
			if (!FACE.is_active) {continue;}

			std::vector<Point_t> vertices(FACE.nNodes);
			for (size_t i=0; i<FACE.nNodes; i++) {vertices[i] = _nodes[FACE.nodes[i]].vertex;}
			mesh.insert_element(vertices, FACE.vtkID);
		}

		return mesh;
	}




	template <Float T>
	void HierarchicalMesh<T>::print_topology_ascii_vtk(std::ostream &os) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;

		//create buffer
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes() << " float\n";
		for (size_t i=0; i<nNodes(); i++) { buffer << _nodes[i].vertex << "\n";}
		
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		
		//ELEMENTS
		//calculate the number of entries required (numberOfNodes + listOfNodes)
		size_t nEntries = 0;
		size_t nElements = 0;
		#pragma omp parallel for reduction(+:nEntries) reduction(+:nElements)
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			if (_elements[e_idx].is_active) {
				nElements += 1;
				nEntries  += 1 + _elements[e_idx].nNodes;
			}
		}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				buffer << ELEM.nNodes;
				for (size_t n=0; n<ELEM.nNodes; n++) {
					buffer << " " << ELEM.nodes[n];
				}
				buffer << "\n";
			}
		}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VTK_ID
		buffer << "CELL_TYPES " << nElements << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				buffer << ELEM.vtkID << " ";
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template <Float T>
	void HierarchicalMesh<T>::print_topology_binary_vtk(const std::string& filename) const {
	    using Element_t = typename HierarchicalMesh<T>::Element_t;
	    
	    //only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(T)==4 or sizeof(T)==8, "Unsupported floating point size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many nodes.
	    //additionally, the integers are expected to be signed in the legacy format. uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_nodes = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (_nodes.size()-1 > max_legacy_vtk_nodes) {throw std::runtime_error("Node index " + std::to_string(_nodes.size()-1) + " exceeds legacy VTK format limit.");}


	    //open the file in binary write mode
	    std::ofstream file(filename, std::ios::binary);
	    if (!file.is_open()) {
	        throw std::runtime_error("Cannot open file: " + filename);
	    }
	    
	    // Helper lambda to write big-endian integers (cast to int32_t due to legacy vtk)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_size_t = [&file](const size_t value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };

	    auto write_be_int = [&file](const int value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
	    
	    // Helper lambda to write big-endian floats (handles float, double, etc.)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_float = [&file](T value) {
	        if constexpr (sizeof(T) == 4) {
	            // 32-bit float
	            uint32_t temp;
	            std::memcpy(&temp, &value, sizeof(T));
	            uint32_t be_value = ((temp & 0xFF000000) >> 24) |
	                                ((temp & 0x00FF0000) >> 8)  |
	                                ((temp & 0x0000FF00) << 8)  |
	                                ((temp & 0x000000FF) << 24);
	            file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	        } else if constexpr (sizeof(T) == 8) {
	            // 64-bit double
	            uint64_t temp;
	            std::memcpy(&temp, &value, sizeof(T));
	            uint64_t be_value = ((temp & 0xFF00000000000000ULL) >> 56) |
	                                ((temp & 0x00FF000000000000ULL) >> 40) |
	                                ((temp & 0x0000FF0000000000ULL) >> 24) |
	                                ((temp & 0x000000FF00000000ULL) >> 8)  |
	                                ((temp & 0x00000000FF000000ULL) << 8)  |
	                                ((temp & 0x0000000000FF0000ULL) << 24) |
	                                ((temp & 0x000000000000FF00ULL) << 40) |
	                                ((temp & 0x00000000000000FFULL) << 56);
	            file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	        }
	    };
	    
	    // HEADER (note legacy vtk can combine ascii and binary data)
	    file << "# vtk DataFile Version 2.0\n";
	    file << "Mesh Data\n";
	    file << "BINARY\n\n";
	    file << "DATASET UNSTRUCTURED_GRID\n";
	    
	    // POINTS (binary data)
	    if constexpr (sizeof(T)==4) {file << "POINTS " << nNodes() << " float\n";}
	    else if constexpr (sizeof(T)==8) {file << "POINTS " << nNodes() << " double\n";}
	    
	    for (size_t i = 0; i < nNodes(); i++) {
	        write_be_float(_nodes[i].vertex[0]);
	        write_be_float(_nodes[i].vertex[1]);
	        write_be_float(_nodes[i].vertex[2]);
	    }
	    file << "\n";
	    
	    // ELEMENTS - calculate counts
	    size_t nEntries = 0;
	    size_t nElements = 0;
	    #pragma omp parallel for reduction(+:nEntries) reduction(+:nElements)
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
	        if (_elements[e_idx].is_active) {
	            nElements += 1;
	            nEntries  += 1 + _elements[e_idx].nNodes;
	        }
	    }
	    
	    // CELLS (binary data)
	    file << "CELLS " << nElements << " " << nEntries << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
	        const Element_t &ELEM = _elements[e_idx];
	        if (ELEM.is_active) {
	            write_be_size_t(ELEM.nNodes);
	            for (size_t n = 0; n < ELEM.nNodes; n++) {
	                write_be_size_t(ELEM.nodes[n]);
	            }
	        }
	    }
	    file << "\n";
	    
	    // CELL_TYPES (binary data)
	    file << "CELL_TYPES " << nElements << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
	        const Element_t &ELEM = _elements[e_idx];
	        if (ELEM.is_active) {
	            write_be_int(ELEM.vtkID);
	        }
	    }
	    file << "\n";
	    
	    file.close();
	}


	template <Float T>
	void HierarchicalMesh<T>::print_mesh_details_ascii_vtk(std::ostream &os) const {
		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t = typename HierarchicalMesh<T>::Node_t;
		std::stringstream buffer;

		//NODE DETAILS
		int n_node_fields = 2;
		if (!_boundary_is_computed) {n_node_fields-=1;}

		buffer << "POINT_DATA " << _nodes.size() << "\n";
		buffer << "FIELD node_info " << n_node_fields << "\n";

		//boundary
		size_t max_boundary_faces=0;
		if (_boundary_is_computed) {
			#pragma omp parallel for reduction(std::max:max_boundary_faces)
			for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
				max_boundary_faces = std::max(max_boundary_faces, _nodes[n_idx].boundary_faces.size());
			}

			buffer << "boundary " << max_boundary_faces << " " << _nodes.size() << " integer\n";
			for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
				const Node_t &NODE = _nodes[n_idx];

				size_t i;
				for (i=0; i<NODE.boundary_faces.size(); i++) { buffer << NODE.boundary_faces[i] << " ";	}
				for (;i<max_boundary_faces; i++) { buffer << "-1 ";}
			}
			buffer << "\n\n";
			os << buffer.rdbuf();
			buffer.str("");
		}
		

		//elements
		size_t max_elem=0;
		#pragma omp parallel for reduction(std::max:max_elem)
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			max_elem = std::max(max_elem, _nodes[n_idx].elems.size());
		}
		
		buffer << "elements " << max_elem << " " << _nodes.size() << " integer\n";
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];
			size_t i;
			for (i=0; i<NODE.elems.size(); i++) { buffer << NODE.elems[i] << " ";}
			for (; i<max_elem; i++) { buffer << "-1 ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


		//ELEMENT DETAILS
		//calculate the number of elements and children 
		size_t max_children = 0;
		size_t nElements = 0;
		#pragma omp parallel for reduction(std::max:max_children) reduction(+:nElements)
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				nElements += 1;
				max_children = std::max(max_children, ELEM.children.size());
			}
		}

		buffer << "CELL_DATA " << nElements << "\n";
		int n_fields = 5;
		if (max_children==0) {n_fields-=1;}

		buffer << "FIELD elem_info " << n_fields << "\n";

		//children
		if (max_children>0) {
			buffer << "children " << max_children << " " << nElements << " integer\n";
			for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
				const Element_t &ELEM = _elements[e_idx];
				if (ELEM.is_active) {
					size_t i;
					for (i=0; i<ELEM.children.size(); i++) {buffer << ELEM.children[i] << " ";}
					for (; i<max_children; i++) {buffer << "-1 ";}
				}
			}
			buffer << "\n\n";
			os << buffer.rdbuf();
			buffer.str("");
		}

		//parent
		buffer << "parent 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				if (ELEM.parent == (size_t) -1) {buffer << "-1 ";}
				else {buffer << ELEM.parent << " ";}
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//index
		buffer << "element_index 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				buffer << ELEM.index << " ";
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//color
		buffer << "color 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				if (ELEM.color == (size_t) -1) {buffer << "-1 ";}
				else {buffer << ELEM.color << " ";}
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//neighbors
		size_t max_neighbors=0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				get_element_neighbors(e_idx, neighbors[n_idx]);
				max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
				n_idx+=1;
			}
		}
		buffer << "neighbors " << max_neighbors << " " << nElements << " integer\n";
		
		n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				size_t i;
				for (i=0; i<neighbors[n_idx].size(); i++) {buffer << neighbors[n_idx][i] << " ";}
				for (; i<max_neighbors; i++) {buffer << "-1 ";}
				n_idx+=1;
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template <Float T>
	void HierarchicalMesh<T>::print_mesh_details_binary_vtk(const std::string& filename) const {
		//only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(T)==4 or sizeof(T)==8, "Unsupported floating point size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many nodes.
	    //additionally, the integers are expected to be signed in the legacy format. uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_nodes = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (_nodes.size()-1 > max_legacy_vtk_nodes) {throw std::runtime_error("Node index " + std::to_string(_nodes.size()-1) + " exceeds legacy VTK format limit.");}


		using Element_t = typename HierarchicalMesh<T>::Element_t;
		using Node_t = typename HierarchicalMesh<T>::Node_t;
		
		// Open file in append mode
		std::ofstream file(filename, std::ios::binary | std::ios::app);
		if (!file.is_open()) {
			throw std::runtime_error("Cannot open file for appending: " + filename);
		}
		
		// Helper lambda to write big-endian integers (cast to int32_t due to legacy vtk)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_size_t = [&file](const size_t value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };

	    auto write_be_int = [&file](const int value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
		
		//NODE DETAILS
		int n_node_fields = 2;
		if (!_boundary_is_computed) { n_node_fields -= 1; }
		
		file << "POINT_DATA " << _nodes.size() << "\n";
		file << "FIELD node_info " << n_node_fields << "\n";
		
		//boundary
		size_t max_boundary_faces = 0;
		if (_boundary_is_computed) {
			#pragma omp parallel for reduction(std::max:max_boundary_faces)
			for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
				max_boundary_faces = std::max(max_boundary_faces, _nodes[n_idx].boundary_faces.size());
			}
			
			file << "boundary " << max_boundary_faces << " " << _nodes.size() << " int\n";
			for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
				const Node_t &NODE = _nodes[n_idx];
				
				size_t i;
				for (i = 0; i < NODE.boundary_faces.size(); i++) {
					write_be_size_t(NODE.boundary_faces[i]);
				}
				for (; i < max_boundary_faces; i++) {
					write_be_int(-1);
				}
			}
			file << "\n";
		}
		
		//elements
		size_t max_elem = 0;
		#pragma omp parallel for reduction(std::max:max_elem)
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			max_elem = std::max(max_elem, _nodes[n_idx].elems.size());
		}
		
		file << "elements " << max_elem << " " << _nodes.size() << " int\n";
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];
			size_t i;
			for (i = 0; i < NODE.elems.size(); i++) {
				write_be_size_t(NODE.elems[i]);
			}
			for (; i < max_elem; i++) {
				write_be_int(-1);
			}
		}
		file << "\n";
		
		//ELEMENT DETAILS
		//calculate the number of elements and children 
		size_t max_children = 0;
		size_t nElements = 0;
		#pragma omp parallel for reduction(std::max:max_children) reduction(+:nElements)
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				nElements += 1;
				max_children = std::max(max_children, ELEM.children.size());
			}
		}
		
		file << "CELL_DATA " << nElements << "\n";
		int n_fields = 5;
		if (max_children == 0) { n_fields -= 1; }
		
		file << "FIELD elem_info " << n_fields << "\n";
		
		
		//children
		if (max_children > 0) {
			file << "children " << max_children << " " << nElements << " int\n";
			for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
				const Element_t &ELEM = _elements[e_idx];
				if (ELEM.is_active) {
					size_t i;
					for (i = 0; i < ELEM.children.size(); i++) {
						write_be_size_t(ELEM.children[i]);
					}
					for (; i < max_children; i++) {
						write_be_int(-1);
					}
				}
			}
			file << "\n";
		}
		
		//parent
		file << "parent 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				if (ELEM.parent == (size_t) -1) {
					write_be_int(-1);
				} else {
					write_be_size_t(ELEM.parent);
				}
			}
		}
		file << "\n";
		
		//index
		file << "element_index 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				write_be_size_t(ELEM.index);
			}
		}
		file << "\n";
		
		//color
		file << "color 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				if (ELEM.color == (size_t) -1) {
					write_be_int(-1);
				} else {
					write_be_size_t(ELEM.color);
				}
			}
		}
		file << "\n";
		
		//neighbors
		size_t max_neighbors = 0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				get_element_neighbors(e_idx, neighbors[n_idx]);
				max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
				n_idx += 1;
			}
		}
		file << "neighbors " << max_neighbors << " " << nElements << " int\n";
		
		n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.is_active) {
				size_t i;
				for (i = 0; i < neighbors[n_idx].size(); i++) {
					write_be_size_t(neighbors[n_idx][i]);
				}
				for (; i < max_neighbors; i++) {
					write_be_int(-1);
				}
				n_idx += 1;
			}
		}
		file << "\n";
		
		file.close();
	}




	template <Float T>
	void HierarchicalMesh<T>::save_as(const std::string filename, const bool include_details, const bool use_ascii) const {
		if (use_ascii) {
			//open and check file
			std::ofstream file(filename);
			if (not file.is_open()){
				std::cout << "Couldn't write to " << filename << std::endl;
				file.close();
				assert(false);
				return;
			}

			//print topology
			print_topology_ascii_vtk(file);

			//print details
			if (include_details) {print_mesh_details_ascii_vtk(file);}

			file.close();
		} else {
			//print topology
			print_topology_binary_vtk(filename);

			//print details
			if (include_details) {print_mesh_details_binary_vtk(filename);}
		}
	}


	template <Float T>
	std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<T> &mesh) {
		os << "nElems(true)= " << mesh.nElems(true) << " (active)\n";
		os << "nElems(false)= " << mesh.nElems(false) << " (total)\n";
		os << "nNodes= " << mesh.nNodes() << "\n";
		os << "colors (" << mesh._colorCount.size() << ") : ";
		for (size_t c_idx=0; c_idx<mesh._colorCount.size(); c_idx++) {
			os << " " << mesh._colorCount[c_idx];
		}
		os << "\n";
		
		return os;
	}
}

