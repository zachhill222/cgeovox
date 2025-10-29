#pragma once

#include "util/point.hpp"
#include "util/point_octree.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
#include <set>
#include <algorithm>

#include <cassert>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>


namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// A container for tracking the element information. Because elements of different types are allowed, we cannot store the node indices here and keep contiguous storage.
	/////////////////////////////////////////////////
	struct Element {
		///Element type defined in the VTK documentation.
		int vtkID;

		/// Which nodes define this element
		std::vector<size_t> nodes;

		/////////////////////////////////////////////////
		/// The color of this element in the overall mesh. Used to allow parallel operations.
		/// The color is used as an index of std::vector, so it is of type size_t. Uncolored elements should have color=(size_t) -1.
		/////////////////////////////////////////////////
		size_t color;

		/// Which element is the parent of this element
		size_t parent;

		/// Which elements are children of this element
		std::vector<size_t> children;

		/// Track if the element is active
		bool is_active;

		/// Constructor when some information is known
		Element(int vtkID, size_t nNodes, size_t color=-1, size_t parent=-1, size_t nChildren=0, bool is_active=true) : 
			vtkID(vtkID), nodes(nNodes), color(color), parent(parent), children(nChildren), is_active(is_active) {}
	};

	
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

		/// Track which boundary the node belongs to. Interior nodes are marked with -1.
		int boundary;

		/// Which elements this node belongs to. The number of elements cannot be known ahead of time, especially when a mesh is allowed to be refined.
		std::vector<size_t> elems;

		/// Constructor when the vertex is known.
		Node(const Point_t &coord, int boundary=-1, size_t nElems=0) : vertex(coord), boundary(boundary), elems(nElems) {}

		/// Default constructor
		Node() : vertex(), boundary(-1), elems() {}
	};

	/// Equality check for mesh nodes.
	template <typename Point_t>
	bool operator==(const Node<Point_t> &A, const Node<Point_t> &B) {return A.vertex==B.vertex;}

	/////////////////////////////////////////////////
	/// A container for storing the nodes in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template <typename Node_t, int dim=3, int n_data=64>
	class NodeOctree : public gv::util::BasicOctree_Point<Node_t, dim, n_data>
	{
	public:
		using Data_t = Node_t;
		NodeOctree() : //if bounding box is unknown ahead of time
			gv::util::BasicOctree_Point<Node_t, dim, n_data>(1024) {}

		NodeOctree(const gv::util::Box<dim> &bbox, const size_t capacity=1024) :
			gv::util::BasicOctree_Point<Data_t, dim, n_data>(bbox, capacity) {}

	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const Data_t &data) const override {return box.contains(data.vertex);}
	};


	/////////////////////////////////////////////////
	/// This class defines a topological mesh that supports different element types. This class only tracks the topology of the mesh and topological operations.
	/// Points can be stored using various precisions (e.g., T=float, double, long double). 
	/// There is no requirement or guarentee that the elements are conforming or that they do not overlap.
	/// Be aware that overlapping elements (in space) are not topologically overlapping if they do not share a node and could have the same color.
	///
	/// @tparam dim The dimension of the space that the mesh is embedded in. Usually dim=3.
	/// @tparam T The precision that the vertices are stored in. It may be completely unnecessary to store the vertices in double precision for some meshes.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template <int dim, Float T=float>
	class TopologicalMesh
	{
	public:
		//common typedefs
		using Index_t     = gv::util::Point<dim,size_t>;
		using Point_t     = gv::util::Point<dim,T>;
		using Node_t      = Node<Point_t>;
		using Element_t   = Element;
		using Box_t       = gv::util::Box<dim,T>;
		using NodeList_t  = NodeOctree<Node_t, dim, 64>;


	protected:
		NodeList_t              _nodes;        //store the vertices/nodes and acompanying data
		std::vector<Element_t>  _elements;     //track element types and which block of _elem2node belongs to each element
		std::vector<size_t>     _colorCount;   //track how many elements belong to each color group. The index is the color.

	public:
		/// Default constructor.
		TopologicalMesh() : _nodes() {}

		/////////////////////////////////////////////////
		/// Constructor when the maximum extents of the mesh are known ahead of time but the elements will be constructed later.
		///
		/// @param bbox The maximum extents of the mesh.
		/////////////////////////////////////////////////
		TopologicalMesh(const Box_t& bbox) : _nodes(bbox) {}


		/////////////////////////////////////////////////
		/// Constructor for a uniform mesh of voxels in 3D.
		///
		/// @param bbox The region to be meshed.
		/// @param N The number of elements along each coordinate axis.
		/////////////////////////////////////////////////
		template<int n=dim, std::enable_if_t<n==3, int> = 0>
		TopologicalMesh(const Box_t& domain, const Index_t& N) : _nodes(domain) {
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
						Point_t element_vertices[8];
						for ( int l=0; l<8; l++) {element_vertices[l] = elem.voxelvertex(l);}

						//put the element into the mesh
						insertElement(element_vertices, 11);
					}
				}
			}
		}


		/////////////////////////////////////////////////
		/// Constructor for a uniform mesh of pixels in 2D.
		///
		/// @param bbox The region to be meshed.
		/// @param N The number of elements along each coordinate axis.
		/////////////////////////////////////////////////
		template<int n=dim, std::enable_if_t<n==2, int> = 0>
		TopologicalMesh(const Box_t& domain, const Index_t& N) : _nodes(domain) {
			//reserve space
			_nodes.reserve((N[0]+1) * (N[1]+1));
			_elements.reserve(N[0]*N[1]);
			
			//construct the mesh
			Point_t H = domain.sidelength() / Point_t(N);
			for (size_t i=0; i<N[0]; i++) {
				for (size_t j=0; j<N[1]; j++) {
					//define element extents
					Point_t low  = domain.low() + Point_t{i,j} * H;
					Point_t high = domain.low() + Point_t{i+1,j+1} * H;
					Box_t   elem  {low, high};
				
					//assemble the list of vertices
					Point_t element_vertices[4];
					for ( int l=0; l<4; l++) {element_vertices[l] = elem.voxelvertex(l);}

					//put the element into the mesh
					insertElement(element_vertices, 8);
				}
			}
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
		/// A method to get the ACTIVE elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back(neighbor_elem_idx)).
		/// @param activeOnly Optionally, the user can count inactive elements as neighbors
		/////////////////////////////////////////////////
		void getElementNeighbors(const size_t elem_idx, std::vector<size_t> &neighbors, const bool activeOnly=true) const;
		
		
		/////////////////////////////////////////////////
		/// A method to get a list of all element types and how many elements of each type exist.
		///
		/// @param vtkID A reference to an existing vector where the element types (vtk identifiers) will be stored.
		/// @param count A reference to an existing vector where the count for each element type will be stored
		/////////////////////////////////////////////////
		void getAllElemTypes(std::vector<int> &vtkID, std::vector<size_t> &count) const;
		

		/////////////////////////////////////////////////
		/// A method to get a list of all elements that are of the specified type.
		///
		/// @param vtkID The vtk identifier of the requested element type.
		/// @param elements A reference to an existing vector where the element indices will be stored.
		/////////////////////////////////////////////////
		void getElementTypeGroup(const int vtkID, std::vector<size_t> &elements) const;

		
		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		void insertElement(Element_t &ELEM, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @tparam N_NODES The number of nodes required to define this element.
		/// @param vertices A reference to an existing array of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/// @param useGreedy When set to true, the first available color will be used. Otherwise the color with the least number of elements will be used (balanced coloring)
		/////////////////////////////////////////////////
		template<int N_NODES>
		void insertElement(const Point_t (&vertices)[N_NODES], const int vtkID, const bool useGreedy=false);


		/////////////////////////////////////////////////
		/// A method to split/refine an existing element. The element that is split is not automatically deleted.
		/// However, the new elements may be colored as if the original element was deleted. The new elements are of the same type as the original.
		/// New nodes will most likely be created and old nodes updated during this process.
		/// For certain elements (i.e., hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		///
		/// @param elem_idx The element to be split.
		/// @param colorAsDeleted When set to true, the new elements are colored as if the element elem_idx has been deleted. The user must delete this element later.
		///                       It will be more efficient to delete many elements at once if many refinement operations are done at once.
		///
		/// @todo Add support for more element types. Move the index logic to smaller functions to clean up the code?
		/////////////////////////////////////////////////
		void splitElement(const size_t elem_idx);
		

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


		/////////////////////////////////////////////////
		/// Helper function to split VTK_VOXEL elements. Should not be called by the user.
		///
		/// @param mesh A reference to this mesh.
		/// @param color_count_increasing A vector of the existing colors sorted by increasing count.
		/// @param elem_idx The index of the element being split
		/// @param sub_elem_idx The index of the element being created (from 0 to 7 for VTK_VOXEL elements)
		/////////////////////////////////////////////////
		friend void _SPLIT_VTK_VOXEL<dim,T>(TopologicalMesh<dim,T> &mesh,
			const size_t (&node_idx)[27], const std::vector<size_t> &color_count_increasing, const size_t elem_idx, const int sub_elem_idx);
	};


	
	template <int dim, Float T>
	void TopologicalMesh<dim,T>::getElementNeighbors(const size_t elem_idx, std::vector<size_t> &neighbors, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;
		using Node_t = typename TopologicalMesh<dim,T>::Node_t;

		const Element_t &ELEM = _elements[elem_idx];

		//loop through the nodes of the current element
		for (size_t n_idx=0; n_idx<ELEM.nodes.size(); n_idx++) {
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
	

	template <int dim, Float T>
	void TopologicalMesh<dim,T>::getAllElemTypes(std::vector<int> &vtkID, std::vector<size_t> &count) const {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;

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


	template <int dim, Float T>
	void TopologicalMesh<dim,T>::getElementTypeGroup(const int vtkID, std::vector<size_t> &elements) const {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.vtkID == vtkID) {elements.push_back(e_idx);}
		}
	}


	template <int dim, Float T>
	void TopologicalMesh<dim,T>::insertElement(typename TopologicalMesh<dim,T>::Element_t &ELEM, const bool useGreedy) {
		//add the element to the mesh
		size_t e_idx = _elements.size(); //index of the new element
		
		//update existing nodes
		for (size_t n=0; n<ELEM.nodes.size(); n++) {
			size_t node_idx = ELEM.nodes[n];
			_nodes[node_idx].elems.push_back(e_idx);
		}

		//move ELEM to _elements
		_elements.push_back(std::move(ELEM));

		recolor(_elements.size()-1, useGreedy);
	}


	template <int dim, Float T>
	template <int N_NODES>
	void TopologicalMesh<dim,T>::insertElement(const typename TopologicalMesh<dim,T>::Point_t (&vertices)[N_NODES], const int vtkID, const bool useGreedy) {
		//construct the element information
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;
		using Node_t    = typename TopologicalMesh<dim,T>::Node_t;

		//initialize the new element
		Element_t ELEM(vtkID, N_NODES);

		//create new nodes as needed and aggregate their indices
		for (int n=0; n<N_NODES; n++) {
			Node_t NODE(vertices[n]);

			size_t n_idx = (size_t) -1;
			[[maybe_unused]] int flag = _nodes.push_back(NODE, n_idx);
			assert(flag>=0);

			assert(n_idx<_nodes.size());
			ELEM.nodes[n]=n_idx;
		}

		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		insertElement(ELEM, useGreedy);
	}



	template <int dim, Float T>
	void TopologicalMesh<dim,T>::recolor(const bool useGreedy) {
		_colorCount.clear();

		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			if (_elements[e_idx].is_active) {recolor(e_idx, useGreedy);}
		}
	}


	template <int dim, Float T>
	void TopologicalMesh<dim,T>::recolor(const size_t elem_idx, const bool useGreedy) {
		//if no colors are recorded, initialize _colorCount
		if (_colorCount.size()==0) {
			_elements[elem_idx].color = 0;
			_colorCount.push_back(1);
			return;
		}

		//get the active neighbor elements
		std::vector<size_t> neighbors;
		getElementNeighbors(elem_idx, neighbors);

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



	template <int dim, Float T>
	void TopologicalMesh<dim,T>::print_topology_ascii_vtk(std::ostream &os, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;

		//create buffer
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes() << " float\n";
		if (dim==3) {
			for (size_t i=0; i<nNodes(); i++) { buffer << _nodes[i].vertex << "\n";}
		} else if (dim==2) {
			for (size_t i=0; i<nNodes(); i++) { buffer << _nodes[i].vertex << "0\n";}
		}
		
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
				nEntries  += 1 + _elements[e_idx].nodes.size();
			}
		}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (!activeOnly or ELEM.is_active) {
				buffer << ELEM.nodes.size();
				for (size_t n=0; n<ELEM.nodes.size(); n++) {
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


	template <int dim, Float T>
	void TopologicalMesh<dim,T>::print_mesh_details_ascii_vtk(std::ostream &os, const bool activeOnly) const {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;
		using Node_t = typename TopologicalMesh<dim,T>::Node_t;
		std::stringstream buffer;

		//NODE DETAILS
		buffer << "POINT_DATA " << _nodes.size() << "\n";
		buffer << "FIELD node_info 2\n";

		//boundary
		buffer << "boundary 1 " << _nodes.size() << " integer\n";
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) { buffer << _nodes[n_idx].boundary << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

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
				buffer << e_idx << " ";
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
				getElementNeighbors(e_idx, neighbors[n_idx], activeOnly);
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


	template <int dim, Float T>
	void TopologicalMesh<dim,T>::save_as(std::string filename, const bool include_details, const bool activeOnly) const {
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



	template <int dim, Float T>
	void TopologicalMesh<dim,T>::splitElement(const size_t elem_idx) {
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;
		using Node_t    = typename TopologicalMesh<dim,T>::Node_t;
		using Point_t   = typename TopologicalMesh<dim,T>::Point_t;


		//create 
		Element_t& ELEM = _elements[elem_idx];
		ELEM.is_active  = false;

		switch (ELEM.vtkID) {
		case 11: //VTK_VOXEL
			assert(dim==3);
			//set up 27 vertices required for the refined elements
			Point_t new_coords[27];
			
			//original (corners)
			for (int n=0; n<8; n++) {new_coords[n] = _nodes[ELEM.nodes[n]].vertex;}

			//edge midpoints
			new_coords[ 8] = 0.5*(new_coords[0]+new_coords[1]); //back face
			new_coords[ 9] = 0.5*(new_coords[1]+new_coords[3]); //back face
			new_coords[10] = 0.5*(new_coords[2]+new_coords[3]); //back face
			new_coords[11] = 0.5*(new_coords[0]+new_coords[2]); //back face

			new_coords[12] = 0.5*(new_coords[0]+new_coords[4]); //connecting edge
			new_coords[13] = 0.5*(new_coords[2]+new_coords[6]); //connecting edge
			new_coords[14] = 0.5*(new_coords[3]+new_coords[7]); //connecting edge
			new_coords[15] = 0.5*(new_coords[1]+new_coords[5]); //connecting edge
			
			new_coords[16] = 0.5*(new_coords[4]+new_coords[5]); //front face
			new_coords[17] = 0.5*(new_coords[5]+new_coords[7]); //front face
			new_coords[18] = 0.5*(new_coords[6]+new_coords[7]); //front face
			new_coords[19] = 0.5*(new_coords[4]+new_coords[6]); //front face

			//face midpoints
			new_coords[20] = 0.5*(new_coords[0]+new_coords[6]); //left face
			new_coords[21] = 0.5*(new_coords[1]+new_coords[7]); //right face
			new_coords[22] = 0.5*(new_coords[2]+new_coords[7]); //top face
			new_coords[23] = 0.5*(new_coords[0]+new_coords[5]); //bottom face
			new_coords[24] = 0.5*(new_coords[0]+new_coords[3]); //back face
			new_coords[25] = 0.5*(new_coords[4]+new_coords[7]); //front face

			//center
			new_coords[26] = 0.5*(new_coords[0]+new_coords[7]);

			//create the nodes and get their indices
			size_t node_idx[27];
			for (int i=0; i<27; i++)
			{
				Node_t NODE(new_coords[i]);
				[[maybe_unused]] int flag = _nodes.push_back(NODE, node_idx[i]); //existing nodes will not be overwritten
				assert(flag>=0);
			}

			//insert and color the new elements
			size_t elem_nodes[8]; //storage for element node indices
			std::vector<size_t> neighbors; //storage for neighbor elements
			std::vector<bool> unsorted_color_allowed; //storage for tracking if a color is allowed. refers to unsorted colors.

			std::vector<size_t> color_count_increasing(_colorCount.size(), 0);
			for (size_t i=0; i<_colorCount.size(); i++) {color_count_increasing[i]=i;} //increasing by color labels
			auto color_count_compare = [this](const size_t a, const size_t b) {return this->_colorCount[a]<this->_colorCount[b];}; //comparator to sort by increasing color count
			std::sort(color_count_increasing.begin(), color_count_increasing.end(), color_count_compare); //color_count_increasing is in increasing order now
			
			for (int i=0; i<8; i++) {
				_SPLIT_VTK_VOXEL(*this, node_idx, color_count_increasing, elem_idx, i);
			}
			break;

		case 12: //VTK_HEXAHEDRON
			assert(dim==3);
			break;
		case 8: //VTK_PIXEL
			break;
		case 9: //VTK_QUAD
			break;

		}
	}






	/// Helper function for splitting an element of VTK_VOXEL type. This should not be called by the user.
	template <int dim, Float T>
	void _SPLIT_VTK_VOXEL(TopologicalMesh<dim,T>& mesh,
		const size_t (&node_idx)[27],                      //indices of the nodes in the mesh that will define the new elements
		const std::vector<size_t> &color_count_increasing, //colors sorted by increasing color count (before any new elements are added or colored)
		const size_t elem_idx,                             //index of element being split
		const int sub_elem_idx)                            //which sub-element is being created
	{
		using Element_t = typename TopologicalMesh<dim,T>::Element_t;

		//define the new element
		Element_t ELEM(11, 8, -1, elem_idx);
		ELEM.nodes.resize(8);
		switch (sub_elem_idx) {
			case (0): //voxel element containing original vertex 0
				ELEM.nodes[0] = node_idx[ 0]; //0
				ELEM.nodes[1] = node_idx[ 8]; //0-1
				ELEM.nodes[2] = node_idx[11]; //0-2
				ELEM.nodes[3] = node_idx[24]; //0-3
				ELEM.nodes[4] = node_idx[12]; //0-4
				ELEM.nodes[5] = node_idx[23]; //0-5
				ELEM.nodes[6] = node_idx[20]; //0-6
				ELEM.nodes[7] = node_idx[26]; //0-7
				break;

			case (1): //voxel element containing original vertex 1
				ELEM.nodes[0] = node_idx[ 8]; //0-1
				ELEM.nodes[1] = node_idx[ 1]; //1
				ELEM.nodes[2] = node_idx[24]; //0-3
				ELEM.nodes[3] = node_idx[ 9]; //1-3
				ELEM.nodes[4] = node_idx[23]; //0-5
				ELEM.nodes[5] = node_idx[15]; //1-5
				ELEM.nodes[6] = node_idx[26]; //0-7
				ELEM.nodes[7] = node_idx[21]; //1-7
				break;

			case (2): //voxel element containing original vertex 2
				ELEM.nodes[0] = node_idx[11]; //0-2
				ELEM.nodes[1] = node_idx[24]; //0-3
				ELEM.nodes[2] = node_idx[ 2]; //2
				ELEM.nodes[3] = node_idx[10]; //2-3
				ELEM.nodes[4] = node_idx[20]; //0-6
				ELEM.nodes[5] = node_idx[26]; //0-7
				ELEM.nodes[6] = node_idx[13]; //2-6
				ELEM.nodes[7] = node_idx[22]; //2-7
				break;

			case (3): //voxel element containing original vertex 3
				ELEM.nodes[0] = node_idx[24]; //0-3
				ELEM.nodes[1] = node_idx[ 9]; //1-3
				ELEM.nodes[2] = node_idx[10]; //2-3
				ELEM.nodes[3] = node_idx[ 3]; //3
				ELEM.nodes[4] = node_idx[26]; //0-7
				ELEM.nodes[5] = node_idx[21]; //1-7
				ELEM.nodes[6] = node_idx[22]; //2-7
				ELEM.nodes[7] = node_idx[14]; //3-7
				break;

			case (4): //voxel element containing original vertex 4
				ELEM.nodes[0] = node_idx[12]; //0-4
				ELEM.nodes[1] = node_idx[23]; //0-5
				ELEM.nodes[2] = node_idx[20]; //0-6
				ELEM.nodes[3] = node_idx[26]; //0-7
				ELEM.nodes[4] = node_idx[ 4]; //4
				ELEM.nodes[5] = node_idx[16]; //4-5
				ELEM.nodes[6] = node_idx[19]; //4-6
				ELEM.nodes[7] = node_idx[25]; //4-7
				break;

			case (5): //voxel element containing original vertex 5
				ELEM.nodes[0] = node_idx[23]; //0-5
				ELEM.nodes[1] = node_idx[15]; //1-5
				ELEM.nodes[2] = node_idx[26]; //0-7
				ELEM.nodes[3] = node_idx[21]; //1-7
				ELEM.nodes[4] = node_idx[16]; //4-5
				ELEM.nodes[5] = node_idx[ 5]; //5
				ELEM.nodes[6] = node_idx[25]; //4-7
				ELEM.nodes[7] = node_idx[17]; //5-7
				break;

			case (6): //voxel element containing original vertex 6
				ELEM.nodes[0] = node_idx[20]; //0-6
				ELEM.nodes[1] = node_idx[26]; //0-7
				ELEM.nodes[2] = node_idx[13]; //2-6
				ELEM.nodes[3] = node_idx[22]; //2-7
				ELEM.nodes[4] = node_idx[19]; //4-6
				ELEM.nodes[5] = node_idx[25]; //4-7
				ELEM.nodes[6] = node_idx[ 6]; //6
				ELEM.nodes[7] = node_idx[18]; //6-7
				break;

			case (7): //voxel element containing original vertex 7
				ELEM.nodes[0] = node_idx[26]; //0-7
				ELEM.nodes[1] = node_idx[21]; //1-7
				ELEM.nodes[2] = node_idx[22]; //2-7
				ELEM.nodes[3] = node_idx[14]; //3-7
				ELEM.nodes[4] = node_idx[25]; //4-7
				ELEM.nodes[5] = node_idx[17]; //5-7
				ELEM.nodes[6] = node_idx[18]; //6-7
				ELEM.nodes[7] = node_idx[ 7]; //7
				break;
		}
		

		
		//insert the new element and get its neighbors for coloring
		mesh.insertElement(ELEM);
		size_t new_elem_idx = mesh._elements.size()-1;
		mesh._elements[elem_idx].children.push_back(new_elem_idx);

		std::vector<size_t> neighbors;
		mesh.getElementNeighbors(new_elem_idx, neighbors);

		//get which existing colors are allowed for the new element
		std::vector<bool> unsorted_color_allowed(color_count_increasing.size(), true); //compute the allowed colors
		size_t n_allowed = unsorted_color_allowed.size();
		assert(n_allowed>0);
		for (size_t i=0; i<neighbors.size(); i++) {
			const auto &NEIGHBOR = mesh._elements[neighbors[i]];

			//if a neighbor has color i, then color i is not allowed for this element
			if (NEIGHBOR.color < color_count_increasing.size()) { //uncolored elements will be out of range and should be skipped
				unsorted_color_allowed[NEIGHBOR.color] = false;
				n_allowed -= 1;
				if (n_allowed==0) {break;} //we need a new color. note that no other new element can share this color.
			}
		}

		//color the new element
		if (n_allowed==0) { //create a new color
			mesh._elements[new_elem_idx].color = mesh._colorCount.size();
			mesh._colorCount.push_back(1);
		} else { //use the best existing color
			for (size_t i=0; i<unsorted_color_allowed.size(); i++) {
				if (unsorted_color_allowed[i]) {
					size_t color_idx = color_count_increasing[i];
					mesh._elements[new_elem_idx].color = color_idx;
					mesh._colorCount[color_idx] += 1;
					break;
				}
			}
		}
	}
}

