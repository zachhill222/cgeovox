#pragma once

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>

#include <cassert>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>

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
	class TopologicalMesh
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

	protected:
		ElementList_t          _element_storage; //actual storage for elements if this mesh does not reference the elements from some other mesh
		ElementList_t         &_elements;        //track element types and which block of _elem2node belongs to each element
		NodeList_t             _node_storage;    //actual storage for the nodes if this mesh does not reference the nodes from some other mesh
		NodeList_t            &_nodes;           //reference to the node list. use this reference so that the nodes can refer to the nodes of some other mesh.
		std::vector<size_t>    _colorCount;      //track how many elements belong to each color group. The index is the color.
		
		bool                   _boundary_is_computed = false; //track if the boundary needs to be re-computed
		std::vector<Element_t> _boundary;                     //storage for the boundary faces

	public:
		/// Default constructor.
		TopologicalMesh() : _element_storage(), _elements(_element_storage), _node_storage(), _nodes(_node_storage) {}


		// /////////////////////////////////////////////////
		// /// Constructor when using nodes that are stored externally.
		// /// Note that the existing nodes may be changed during mesh operations.
		// /////////////////////////////////////////////////
		// TopologicalMesh(NodeList_t &external_nodes) : _element_storage(), _elements(_element_storage), _node_storage(), _nodes(external_nodes) {}


		// /////////////////////////////////////////////////
		// /// Constructor when using nodes and elements that are stored externally.
		// /// Note that the existing nodes and elements may be changed during mesh operations.
		// /////////////////////////////////////////////////
		// TopologicalMesh(ElementList_t &external_elements, NodeList_t &external_nodes) : _element_storage(), _elements(external_elements), _node_storage(), _nodes(external_nodes) {}


		/////////////////////////////////////////////////
		/// Constructor when using nodes and elements exist and should be copied over
		/////////////////////////////////////////////////
		TopologicalMesh(const ElementList_t &external_elements, const NodeList_t &external_nodes) : 
			_element_storage(external_elements), _elements(_element_storage),
			_node_storage(external_nodes), _nodes(_node_storage) {}


		/////////////////////////////////////////////////
		/// Constructor when the maximum extents of the mesh are known ahead of time but the elements will be constructed later.
		///
		/// @param bbox The maximum extents of the mesh.
		/////////////////////////////////////////////////
		TopologicalMesh(const Box_t& bbox) : _element_storage(), _elements(_element_storage), _node_storage(bbox), _nodes(_node_storage) {}


		/////////////////////////////////////////////////
		/// Constructor for a uniform mesh of voxels in 3D.
		///
		/// @param bbox The region to be meshed.
		/// @param N The number of elements along each coordinate axis.
		/////////////////////////////////////////////////
		TopologicalMesh(const Box_t& domain, const Index<3>& N) : _element_storage(), _elements(_element_storage), _node_storage(domain), _nodes(_node_storage) {
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
		TopologicalMesh(const gv::util::Box<2,T>& domain, const Index<2>& N) {
			Point_t low, high;
			low[0] = domain.low()[0]; low[1] = domain.low()[1]; low[2]=-1.0;
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
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/// @param activeOnly Optionally, the user can count inactive elements as neighbors
		/////////////////////////////////////////////////
		void get_element_neighbors(const size_t elem_idx, std::vector<size_t> &neighbors, const bool activeOnly=true) const;


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
		void get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly=false) const;
		

		/////////////////////////////////////////////////
		/// A method to get the descendent faces of the specified boundary face.
		///
		/// @param face_idx The index of the requested element (i.e., _boundary[face_idx]).
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get only the active descendents
		/////////////////////////////////////////////////
		void get_boundary_face_descendents(const size_t face_idx, std::vector<size_t> &descendents, const bool activeOnly=false) const;
		
		
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
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		void insert_element(Element_t &ELEM, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		void insert_element(const std::vector<Point_t> &vertices, const int vtkID, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// A method to split/refine an existing element. The element that is split is not automatically deleted.
		/// However, the new elements may be colored as if the original element was deleted. The new elements are of the same type as the original.
		/// New nodes will most likely be created and old nodes updated during this process.
		/// For certain elements (i.e., hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		/// If the specified element has already been split and re-joined (i.e., the children exist), then the children are simply activated and no new elements are created in memory.
		/// If this _elements[elem_idx].is_active is false, then the method returns without making any changes.
		///
		/// @param elem_idx The element to be split.
		/// @param useGreedy Flag to use the greedy algorithm for coloring
		///
		/// @todo Add support for more element types. Make VTK containers for each element type to clean up the code?
		/////////////////////////////////////////////////
		void split_element(const size_t elem_idx, const bool useGreedy=false);
		

		/////////////////////////////////////////////////
		/// A method to join/unrefine previously split elements. If _elements[elem_idx] exists, then all of the descendents of that element are de-activated and the element is activated.
		/// The de-activated elements are not deleted. The element is re-colored.
		///
		/// @param elem_idx The element whose descendents are to be joined.
		/// @param useGreedy Flag to use the greedy algorithm for coloring
		/////////////////////////////////////////////////
		void join_descendents(const size_t elem_idx, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// Color or re-color the active elements of the mesh. The elements will be colored such that no active elements that share a node will have the same color.
		///
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		void recolor(const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// Color or re-color a specific element. The color is chosen to be the existing color with the least count if possible.
		/// Otherwise a new color is created.
		///
		/// @param elem_idx The element to be re-colored.
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		void recolor(const size_t elem_idx, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// Compute the boundary elements. The mesh must be in a conformal state (i.e., the coarsest mesh).
		/////////////////////////////////////////////////
		void compute_conformal_boundary();

		/////////////////////////////////////////////////
		/// Return the boundary as a mesh that references the nodes of this mesh.
		/////////////////////////////////////////////////
		TopologicalMesh<T> boundary_mesh() const;

		/////////////////////////////////////////////////
		/// Delete the specified elements from the mesh. The nodes that define each specified element will be updated.
		/// Note that some nodes may be isolated (i.e., they do not belong to any element) after this operation.
		///
		/// @param elements A reference to the indices of the elements to be deleted.
		/////////////////////////////////////////////////
		void remove_elements(const std::vector<size_t> &elements); //remove the specified elements from the mesh


		/////////////////////////////////////////////////
		/// Delete all isolated nodes (i.e., nodes that do not belong to any element) from the mesh. Isolated nodes can be created when elements are deleted.
		/////////////////////////////////////////////////
		void remove_isolated_nodes();


		/////////////////////////////////////////////////
		/// Print the node locations and element connectivity to the output stream. Data can be appended to stream after this is called.
		/// When saving information (e.g., a solution to a PDE define on this mesh), this will initialize the mesh and then the data can be appended.
		///
		/// @param os The output stream.
		/// @param activeOnly When set to true, only the active elements will be written to the file
		/////////////////////////////////////////////////
		void print_topology_ascii_vtk(std::ostream &os, const bool activeOnly=true) const;


		/////////////////////////////////////////////////
		/// Print the details of the nodes and elements to the output stream. This includes element colors and which elements each node belongs to.
		/// Due to the way that field data is stored in ASCII VTK format, it will be difficult to append any additional information to a file afterwards.
		///
		/// @param os The output stream.
		/// @param activeOnly When set to true, only details for the active elements will be written to the file
		/////////////////////////////////////////////////
		void print_mesh_details_ascii_vtk(std::ostream &os, const bool activeOnly=true) const;
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file. If the file already exists, it will be over-written.
		/// Effectively this creates a file stream fs and calls print_topology_ascii_vtk(fs) and then (if include_details=true) calls print_mesh_details_ascii_vtk(fs).
		///
		/// @param filename The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology. This should usually be set to false if any additional data will be appended to the file.
		/// @param activeOnly When set to true, only active elements will be written to the file
		/////////////////////////////////////////////////
		void save_as(const std::string filename, const bool include_details=false, const bool activeOnly=true) const;

		/// Friend function to print the mesh information
		template <Float U>
		friend std::ostream& operator<<(std::ostream& os, const TopologicalMesh<U> &mesh);
	};


	
	template <Float T>
	void TopologicalMesh<T>::get_element_neighbors(const size_t elem_idx, std::vector<size_t> &neighbors, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t = typename TopologicalMesh<T>::Node_t;

		const Element_t &ELEM = _elements[elem_idx];

		//loop through the nodes of the current element
		for (size_t n_idx=0; n_idx<ELEM.nNodes; n_idx++) {
			const Node_t &NODE = _nodes[ELEM.nodes[n_idx]];

			//loop through the elements of the current node
			for (size_t m=0; m<NODE.elems.size(); m++) {
				const size_t e_idx = NODE.elems[m];

				//only leaf elements can be neighbors
				if (e_idx!=elem_idx and (!activeOnly or _elements[e_idx].is_active)) {
					neighbors.push_back(e_idx);
				}
			}
		}

		//make the vector sorted and unique
		std::sort(neighbors.begin(), neighbors.end()); 
		auto last = std::unique(neighbors.begin(), neighbors.end());
		neighbors.erase(last, neighbors.end());
	}


	template <Float T>
	void TopologicalMesh<T>::get_boundary_faces(const size_t elem_idx, std::vector<size_t> &faces) const {
		assert(_boundary_is_computed);

		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t = typename TopologicalMesh<T>::Node_t;

		const Element_t &ELEM = _elements[elem_idx];

		//loop through the nodes of the face
		for (size_t n_idx=0; n_idx<ELEM.nNodes; n_idx++) {
			const Node_t &NODE = _nodes[ELEM.nodes[n_idx]];

			//loop through the elements of the current node
			for (size_t m=0; m<NODE.boundary_faces.size(); m++) {
				const size_t f_idx = NODE.boundary_faces[m];
				const Element_t FACE = _boundary[f_idx];

				if (FACE.parent==elem_idx) {
					faces.push_back(f_idx);
				}
			}
		}

		//make the vector sorted and unique
		std::sort(faces.begin(), faces.end()); 
		auto last = std::unique(faces.begin(), faces.end());
		faces.erase(last, faces.end());
	}
	

	template <Float T>
	void TopologicalMesh<T>::get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;
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
	void TopologicalMesh<T>::get_boundary_face_descendents(const size_t face_idx, std::vector<size_t> &descendents, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;
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
	void TopologicalMesh<T>::get_all_elem_types(std::vector<int> &vtkID, std::vector<size_t> &count) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;

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
	void TopologicalMesh<T>::get_element_type_group(const int vtkID, std::vector<size_t> &elements) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.vtkID == vtkID) {elements.push_back(e_idx);}
		}
	}


	template <Float T>
	void TopologicalMesh<T>::insert_element(Element_t &ELEM, const bool useGreedy) {
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
		recolor(new_elem_idx, useGreedy);
	}


	template <Float T>
	void TopologicalMesh<T>::insert_element(const std::vector<typename TopologicalMesh<T>::Point_t> &vertices, const int vtkID, const bool useGreedy) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//construct the element information
		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t    = typename TopologicalMesh<T>::Node_t;


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
		insert_element(ELEM, useGreedy);
	}



	template <Float T>
	void TopologicalMesh<T>::recolor(const bool useGreedy) {
		_colorCount.clear();

		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			if (_elements[e_idx].is_active) {recolor(e_idx, useGreedy);}
		}
	}


	template <Float T>
	void TopologicalMesh<T>::recolor(const size_t elem_idx, const bool useGreedy) {
		//if no colors are recorded, initialize _colorCount
		if (_colorCount.size()==0) {
			_elements[elem_idx].color = 0;
			_colorCount.push_back(1);
			return;
		}

		//get the active neighbor elements
		std::vector<size_t> neighbors;
		get_element_neighbors(elem_idx, neighbors);

		//decide which colors are allowed
		std::vector<bool> color_allowed(_colorCount.size(), true);
		for (size_t n_idx=0; n_idx<neighbors.size(); n_idx++) {
			size_t color = _elements[neighbors[n_idx]].color;
			if (color_allowed[color]) {
				color_allowed[color] = false;
			}
		}

		bool free_color_exists = false;
		for (size_t i=0; i<color_allowed.size(); i++) {
			if (color_allowed[i]) {free_color_exists=true;}
		}

		//create a new color if needed
		if (!free_color_exists) {
			_elements[elem_idx].color = _colorCount.size();
			_colorCount.push_back(1);
			return;
		}

		//color the element (minimum color value)
		if (useGreedy) {
			for (size_t c_idx=0; c_idx<color_allowed.size(); c_idx++) {
				if (useGreedy and color_allowed[c_idx]) {
					_elements[elem_idx].color = c_idx;
					_colorCount[c_idx] += 1;
					return;
				}
			}
			assert(false);
		} 

		//color the element (minimum color count)
		size_t color=0;
		for (size_t c_idx=1; c_idx<color_allowed.size(); c_idx++) {
			if (color_allowed[c_idx] and _colorCount[c_idx]<_colorCount[color]) {
				color = c_idx;
			}
		}
		_elements[elem_idx].color = color;
		_colorCount[color] += 1;
	}


	template <Float T>
	void TopologicalMesh<T>::join_descendents(const size_t elem_idx, const bool useGreedy) {
		//get the active descendents of the element
		std::vector<size_t> descendents;
		get_element_descendents(elem_idx, descendents, true);

		//de-activate the descendents and any boundary faces
		for (size_t e_idx=0; e_idx<descendents.size(); e_idx++) {
			_elements[descendents[e_idx]].is_active = false;
		}

		//activate the element
		_elements[elem_idx].is_active = true;

		//recolor the element
		recolor(elem_idx,useGreedy);

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
	void TopologicalMesh<T>::split_element(const size_t elem_idx, const bool useGreedy) {
		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t    = typename TopologicalMesh<T>::Node_t;
		using Point_t   = typename TopologicalMesh<T>::Point_t;

		//ensure that _elements is large enough to store the new elements without re-sizing. When splitting, there will be at most 8 child elements
		while (_elements.capacity() < _elements.size()+8) {
			_elements.reserve(2*_elements.capacity());
		}

		//reference element to be split
		assert(elem_idx<_elements.size());
		Element_t &ELEM = _elements[elem_idx];
		
		//check if the element is already active
		if (!ELEM.is_active) {return;}
		ELEM.is_active = false;
		
		//if the element has already been refined, activate its children and return
		if (ELEM.children.size()>0) {
			for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
				_elements[ELEM.children[c_idx]].is_active = true;
				recolor(ELEM.children[c_idx], useGreedy);
			}
			return;
		}


		//create the vtk element to manage the splitting logic
		VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(ELEM);


		//create the new vertices
		std::vector<Point_t> new_coords;
		std::vector<size_t> new_node_idx(vtk_elem->nVerticesWhenSplit());
		size_t i;
		for (i=0; i<ELEM.nNodes; i++) {
			new_coords.push_back(_nodes[ELEM.nodes[i]].vertex);
			new_node_idx[i] = ELEM.nodes[i];
		}
		vtk_elem->split(new_coords);

		
		//create the new nodes at the new vertices
		for (;i<new_coords.size(); i++) {
			Node_t NODE(new_coords[i]);
			[[maybe_unused]] int flag = _nodes.push_back(NODE, new_node_idx[i]); //existing nodes will not be overwritten
			assert(flag>=0);
		}


		//create the children elements
		for (int j=0; j<vtk_elem->nChildrenWhenSplit(); j++) {
			std::vector<size_t> childNodes;
			vtk_elem->getChildNodes(childNodes, j, new_node_idx);

			//add the index of the child element to the element being split
			ELEM.children.push_back(_elements.size());

			//create the child element and add it to the mesh
			Element_t CHILD(childNodes, ELEM.vtkID);
			CHILD.parent = elem_idx;
			insert_element(CHILD, useGreedy);
		}

		//delete the vtk element
		delete vtk_elem;


		//split any boundary faces
		std::vector<size_t> parent_boundary_faces;
		get_boundary_faces(elem_idx, parent_boundary_faces);
		for (size_t f_idx=0; f_idx<parent_boundary_faces.size(); f_idx++) {
			assert(parent_boundary_faces[f_idx]<_boundary.size());
			//ensure that _boundary does not re-size and invalidate references
			while (_boundary.capacity() < _boundary.size() + vtk_n_children(_boundary[parent_boundary_faces[f_idx]].vtkID)) {_boundary.reserve(2*_boundary.capacity());}

			Element_t &FACE = _boundary[parent_boundary_faces[f_idx]];
			FACE.is_active = false;

			VTK_ELEMENT<Point_t>* vtk_face = _VTK_ELEMENT_FACTORY<Point_t>(FACE);
			//find the vertices for the split face
			std::vector<Point_t> new_coords;
			std::vector<size_t> new_node_idx(vtk_face->nVerticesWhenSplit());
			size_t i;
			for (i=0; i<FACE.nNodes; i++) {
				new_coords.push_back(_nodes[FACE.nodes[i]].vertex);
				new_node_idx[i] = FACE.nodes[i];
			}
			vtk_face->split(new_coords);

			//get any new nodes
			for (;i<new_coords.size(); i++) {
				Node_t NODE(new_coords[i]);
				new_node_idx[i] = _nodes.find(NODE); //the node must have been generated by splitting the element
				assert(new_node_idx[i]<_nodes.size());
			}


			//create the children faces
			for (int j=0; j<vtk_face->nChildrenWhenSplit(); j++) {
				std::vector<size_t> childNodes;
				vtk_face->getChildNodes(childNodes, j, new_node_idx);
				Element_t CHILDFACE(childNodes, FACE.vtkID);
				CHILDFACE.parent = (size_t) -1;

				//determine the child element that this split face is a child of
				for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
					const Element_t &CHILD = _elements[ELEM.children[c_idx]];
					const VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(CHILD);
					for (int k=0; k<vtk_n_faces(CHILD.vtkID); k++) {
						if (CHILDFACE == vtk_elem->getFace(k)) {
							CHILDFACE.parent = CHILD.index;
							CHILDFACE.color = CHILD.color;
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

			delete vtk_face;
		}
	}


	template <Float T>
	void TopologicalMesh<T>::compute_conformal_boundary() {
		using Point_t = typename TopologicalMesh<T>::Point_t;
		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t = typename TopologicalMesh<T>::Node_t;

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
	TopologicalMesh<T> TopologicalMesh<T>::boundary_mesh() const {
		using Point_t = typename TopologicalMesh<T>::Point_t;

		//ensure the boundary is computed
		assert(_boundary_is_computed);

		//initialize boundary mesh
		TopologicalMesh<T> mesh(_nodes.bbox());

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
	void TopologicalMesh<T>::print_topology_ascii_vtk(std::ostream &os, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;

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
			if (!activeOnly or _elements[e_idx].is_active) {
				nElements += 1;
				nEntries  += 1 + _elements[e_idx].nNodes;
			}
		}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (!activeOnly or ELEM.is_active) {
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
			if (!activeOnly or ELEM.is_active) {
				buffer << ELEM.vtkID << " ";
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template <Float T>
	void TopologicalMesh<T>::print_mesh_details_ascii_vtk(std::ostream &os, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<T>::Element_t;
		using Node_t = typename TopologicalMesh<T>::Node_t;
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
		//calculate the number of elements and 
		size_t max_children = 0;
		size_t nElements = 0;
		#pragma omp parallel for reduction(std::max:max_children) reduction(+:nElements)
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (!activeOnly or ELEM.is_active) {
				nElements += 1;
				max_children = std::max(max_children, ELEM.children.size());
			}
		}

		buffer << "CELL_DATA " << nElements << "\n";
		int n_fields = 6;
		if (max_children==0) {n_fields-=1;}
		if (activeOnly) {n_fields-=1;}

		buffer << "FIELD elem_info " << n_fields << "\n";

		//isActive
		if (!activeOnly) {
			buffer << "is_active 1 " << nElements << " integer\n";
			for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
				buffer << _elements[e_idx].is_active << " ";
			}
			buffer << "\n\n";
			os << buffer.rdbuf();
			buffer.str("");
		}

		//children
		if (max_children>0) {
			buffer << "children " << max_children << " " << nElements << " integer\n";
			for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
				const Element_t &ELEM = _elements[e_idx];
				if (!activeOnly or ELEM.is_active) {
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
			if (!activeOnly or ELEM.is_active) {
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
			if (!activeOnly or ELEM.is_active) {
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
			if (!activeOnly or ELEM.is_active) {
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
			if (!activeOnly or ELEM.is_active) {
				get_element_neighbors(e_idx, neighbors[n_idx], activeOnly);
				max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
				n_idx+=1;
			}
		}
		buffer << "neighbors " << max_neighbors << " " << nElements << " integer\n";
		
		n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (!activeOnly or ELEM.is_active) {
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
	void TopologicalMesh<T>::save_as(std::string filename, const bool include_details, const bool activeOnly) const {
		//open and check file
		std::ofstream file(filename);

		if (not file.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			file.close();
			assert(false);
			return;
		}

		//print topology
		print_topology_ascii_vtk(file, activeOnly);

		//print details
		if (include_details) {print_mesh_details_ascii_vtk(file, activeOnly);}

		file.close();
	}


	template <Float T>
	std::ostream& operator<<(std::ostream& os, const TopologicalMesh<T> &mesh) {
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

