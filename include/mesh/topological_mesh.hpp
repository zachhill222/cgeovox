#pragma once

#include "util/point.hpp"
#include "util/point_octree.hpp"
#include "util/box.hpp"

#include "concepts.hpp"

#include <vector>
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
	struct ElemInfo {

		///Element type defined in the VTK documentation.
		const int vtkID;

		///How many nodes define this type of element
		const int nNodes;

		///The color of this element in the overall mesh. Used to allow parallel operations.
		int color;

		/// The index to the start of the list of nodes that define this element (i.e., _elem2node[nodeStart] is the first node of this element)
		size_t nodeStart;
	};

	
	/////////////////////////////////////////////////
	/// A container for tracking the node information.
	/// Usually Point_t = gv::util::Point<3,double>, but 2-D meshes or different precisions are allowed.
	/// If the mesh has more than one element type, MAX_ELEMENT_PER_NODE should be the maximum number of nodes required to define any of the used element types.
	/// This allows the nodes to be stored in a contiguous array as they all will require the same amount of memory. For a hexahedral mesh, MAX_ELEMENT_PER_NODE=8.
	/////////////////////////////////////////////////
	template <typename Point_t, int MAX_ELEMENT_PER_NODE>
	struct Node {
		///Which elements this node belongs to. Maximum of 8 for a uniform hexahedral mesh.
		size_t elems[MAX_ELEMENT_PER_NODE];
		
		///Number of elements this node belongs to.
		int nElems;        //number of elements this node belongs to
		
		///Track which boundary the node belongs to. Interior nodes are marked with -1.
		int boundary;

		/// The location of this node in space.
		Point_t vertex;
	};


	/////////////////////////////////////////////////
	/// A container for storing the nodes in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template <typename Node_t, dim=3, n_data=64>
	class NodeOctree : public gv::util::BasicOctree_Point<Node_t, dim, n_data>
	{
	public:
		using Data_t = Node_t;
		PointOctree() : //if bounding box is unknown ahead of time
			gv::util::BasicOctree_Point<Node_t, dim, n_data>(1024) {}

		PointOctree(const gv::util::Box<dim> &bbox, const size_t capacity=1024) :
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
	/// @tparam MAX_ELEMENT_PER_NODE The maximum number of elements that any node can belong to. This is 8 for hexahedral meshes but could be unbounded for poor simplicial meshes.
	///                              In order to store data contiguously, this value must be known.
	/// @tparam T The precision that the vertices are stored in. It may be completely unnecessary to store the vertices in double precision for some meshes.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template <int dim, int MAX_ELEMENT_PER_NODE=8, Float T=float>
	class TopologicalMesh
	{
	public:
		//common typedefs
		using Index_t     = gv::util::Point<dim,size_t>;
		using Point_t     = gv::util::Point<dim,T>;
		using Node_t      = Node<Point_t, MAX_ELEMENT_PER_NODE>;
		using ElemInfo_t  = ElemInfo; //declare here in case it needs to be changed later
		using Box_t       = gv::util::Box<dim,T>;
		using NodeList_t  = NodeOctree<Node_t, dim, 64>;


	protected:
		NodeList_t              _nodes;        //store the vertices/nodes and acompanying data
		std::vector<ElemInfo_t> _elemInfo;     //track element types and which block of _elem2node belongs to each element
		std::vector<size_t>     _elem2node;    //store the nodes for each element in a contiguous array
		std::vector<int>        _uniqueColors; //track the unique colors in the mesh
		std::vector<size_t>     _colorCount;   //track how many elements belong to each color group. Indices corresponding to each

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
			_elemInfo.reserve(N[0]*N[1]*N[2]);

			
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


			//color the mesh
			recolor();
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
			_elemInfo.reserve(N[0]*N[1]);
			
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

			//color the mesh
			recolor();
		}

		/////////////////////////////////////////////////
		/// Get the total number of elements in the mesh.
		/////////////////////////////////////////////////
		inline size_t nElems() const {return _elemInfo.size();}


		/////////////////////////////////////////////////
		/// Get the total number of nodes in the mesh.
		/////////////////////////////////////////////////
		inline size_t nNodes() const {return _nodes.size();} //number of nodes in the mesh


		/////////////////////////////////////////////////
		/// Get a list of colors currently in the mesh.
		/////////////////////////////////////////////////
		inline std::vector<size_t> colors() const {return _uniqueColors;}


		/////////////////////////////////////////////////
		/// Get the number of elements in each color group. colorCount()[i] is the number of elements colored by colors()[i].
		/////////////////////////////////////////////////
		inline std::vector<size_t> colorCount const {return _colorCount;}


		/////////////////////////////////////////////////
		/// A method to get the nodes that define the specified element. This hides the lookup into the contiguous array _elem2nodes from the user.
		///
		/// @param elem_idx The index of the requested element (i.e., _elemInfo[elem_idx]).
		/// @param nodes A reference to an existing vector where the result will be stored (via nodes.push_back(node_idx)).
		/////////////////////////////////////////////////
		void getElementNodes(const size_t elem_idx, std::vector<size_t> &nodes) const;
		

		/////////////////////////////////////////////////
		/// A method to get the elements share a node with the specified element.
		///
		/// @param elem_idx The index of the requested element (i.e., _elemInfo[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back(neighbor_elem_idx)).
		/////////////////////////////////////////////////
		void getElementNeighbors(const size_t elem_idx, std::vector<size_t> &neighbors) const;
		
		
		/////////////////////////////////////////////////
		/// A method to re-compute _uniqueColors and _colorCount.
		/////////////////////////////////////////////////
		void countColors();
		

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
		/// A method to insert a new element into the mesh. The element is constructed from specified existing nodes.
		/// The existing nodes (e.g. _nodes[node_idx[i]] for 0 \leq i < N_NODES_PER_ELEMENT) will be updated but no new nodes will be created.
		///
		/// @param node_idx A reference to an existing array of the indices in _nodes that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		template<int N_NODES_PER_ELEMENT>
		void insertElement(const size_t (&node_idx)[N_NODES_PER_ELEMENT], const int vtkID);


		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing array of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		template<int N_NODES_PER_ELEMENT>
		void insertElement(const Point_t (&vertices)[N_NODES_PER_ELEMENT], const int vtkID);


		/////////////////////////////////////////////////
		/// A method to split/refine an existing element at its centroid. The element that is split is not automatically deleted.
		/// However, the new elements may be colored as if the original element was deleted. The new elements are of the same type as the original.
		/// Unless it was already in the mesh, a new node will be created at the centroid of the element. Otherwise, the existing node at the centroid will be updated.
		/// For certain elements (i.e., hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		///
		/// @param elem_idx The element to be split.
		/// @param colorAsDeleted When set to true, the new elements are colored as if the element elem_idx has been deleted. The user must delete this element later.
		///                       It will be more efficient to delete many elements at once if many refinement operations are done at once.
		/////////////////////////////////////////////////
		void splitElement(const size_t elem_idx, const bool colorAsDeleted=true);
		

		/////////////////////////////////////////////////
		/// A method to split/refine an existing element at the specified point. The element that is split is not automatically deleted.
		/// However, the new elements may be colored as if the original element was deleted. The new elements are of the same type as the original.
		/// Unless it was already in the mesh, a new node will be created at specified point. Otherwise, the existing node at the specified point will be updated.
		/// It is the user's responsibility to ensure that the specified point is valid. Usually this simply means that the specified point is interior to the element.
		/// For certain elements (i.e., hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		///
		/// @param elem_idx The element to be split.
		/// @param new_point The coordinate where element will be split.
		/// @param colorAsDeleted When set to true, the new elements are colored as if the element elem_idx has been deleted. The user must delete this element later.
		///                       It will be more efficient to delete many elements at once if many refinement operations are done at once.
		/////////////////////////////////////////////////
		void splitElement(const size_t elem_idx, const Point_t& new_point, const bool colorAsDeleted=true);


		/////////////////////////////////////////////////
		/// Color or re-color the mesh.
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
		/////////////////////////////////////////////////
		void print_topology_ascii_vtk(std::ostream &os) const;


		/////////////////////////////////////////////////
		/// Print the details of the nodes and elements to the output stream. This includes element colors and which elements each node belongs to.
		/// Due to the way that field data is stored in ASCII VTK format, it will be difficult to append any additional information to a file afterwards.
		///
		/// @param os The output stream.
		/////////////////////////////////////////////////
		void print_mesh_details_ascii_vtk(std::ostream &os) const; //
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file. If the file already exists, it will be over-written.
		/// Effectively this creates a file stream fs and calls print_topology_ascii_vtk(fs) and then (if include_details=true) calls print_mesh_details_ascii_vtk(fs).
		///
		/// @param filename The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology. This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, bool include_details=false) const; //save the mesh to a file with the option to include details for error checking
	};



	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::getElementNodes(const size_t elem_idx; std::vector<size_t>&nodes) const {
		const auto& ELEM = _elemInfo[elem_idx];
		for (size_t n=0; n<ELEM.nNodes; n++) {nodes.push_back(_elem2node[ELEM.nodeStart + n]);}
	}

	
	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::getElementNeighbors(const size_t elem_idx, std::vector<size_t> &neighbors) const {
		const auto& ELEM = _elemInfo[elem_idx];

		//loop through the nodes of the current element
		for (size_t n=0; n<ELEM.nNodes; n++) {
			const size_t node_idx = _elem2node[ELEM.nodeStart + n];
			const auto& NODE = _nodes[node_idx];

			//loop through the elements of the current node
			for (int m=0; m<NODE.nElems; m++) {
				const size_t e_idx = NODE.elems[m];
				//each element is either a neighbor or the original element
				if (e_idx!=elem_idx) {neighbors.push_back(e_idx);}
			}
		}
	}


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::countColors() {
		_uniqueColors.clear();
		_colorCount.clear();

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const auto& ELEM = _elemInfo[elem_idx];

			//check if the color of the current element already exists
			auto it = std::find(_uniqueColors.begin(), _uniqueColors.end(), ELEM.color);

			//if the color of the current element already exists, then increment its count
			if (it!=_uniqueColors.end()) {_colorCount[*it] += 1;}
			else {
				_uniqueColors.push_back(ELEM.color);
				_colorCount.push_back(1);
			}
		}
	}
	

	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::getAllElemTypes(std::vector<int> &vtkID, std::vector<size_t> &count) const {
		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const auto& ELEM = _elemInfo[elem_idx];

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


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::getElementTypeGroup(const int vtkID, std::vector<size_t> &elements) const {
		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const auto& ELEM = _elemInfo[elem_idx];
			if (ELEM.vtkID == vtkID) {elements.push_back(e_idx);}
		}
	}


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	template <int N_NODES_PER_ELEMENT>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::insertElement(const size_t (&node_idx)[N_NODES_PER_ELEMENT], const int vtkID) {
		//construct the element information
		using ElemInfo_t = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::ElemInfo_t;
		ElemInfo_t ELEM {vtkID, N_NODES_PER_ELEMENT, 0, _elem2node.size()};
		const elem_idx = nElems()-1;

		//add the element to the mesh and update the existing nodes
		_elemInfo.push_back(ELEM);
		for (int n=0; n<N_NODES_PER_ELEMENT; n++) {
			//append the node indices to the list of element definitions
			_elem2node.push_back(node_idx[n]);

			//update the list of elements the node belongs to
			assert(nElems+1<MAX_ELEMENT_PER_NODE);
			auto& NODE = _nodes[node_idx[n]];
			NODE.elems[NODE.nElems] = elem_idx;
			NODE.nElems += 1;
		}

		//color the new element
		recolor(elem_idx);
	}


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	template <int N_NODES_PER_ELEMENT>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::insertElement(const typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Point_t (&vertices)[N_NODES_PER_ELEMENT], const int vtkID) {
		//construct the element information
		using ElemInfo_t = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::ElemInfo_t;
		using Point_t    = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Point_t;
		using Node_t     = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Node_t;

		//create new nodes as needed and aggregate their indices
		size_t node_idx[N_NODES_PER_ELEMENT];
		for (int n=0; n<N_NODES_PER_ELEMENT; n++) {
			Node_t NODE { {}, 0, -1, vertices[n]};

			size_t n_idx = (size_t) -1;
			[[maybe_unused]] int flag = _nodes.push_back(NODE, n_idx);
			assert(flag>=0);

			node_idx[n] = n_idx;
		}

		//now that the nodes are initialized, insert and color the element.
		//the nodes will be updated to link back to the new element.
		insertElement(node_idx, vtkID);
	}


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::splitElement(const size_t elem_idx, const bool colorAsDeleted)
	{
		using Point_t    = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Point_t;
		
		const auto& ELEM = _elemInfo[elem_idx];

		//compute the centroid
		Point_t centroid(0);
		switch (ELEM.vtkID) {
		case 11: //VTK_VOXEL
			assert(dim==3);
			size_t n1 = _elem2node[ELEM.nodeStart];
			size_t n2 = _elem2node[ELEM.nodeStart+7];
			centroid  = 0.5*(_nodes[n1].vertex + _nodes[n2].vertex);
			break;
		case 8: //VTK_PIXEL
			size_t n1 = _elem2node[ELEM.nodeStart];
			size_t n2 = _elem2node[ELEM.nodeStart+3];
			centroid  = 0.5*(_nodes[n1].vertex + _nodes[n2].vertex);
			break;
		default: //compute the average of the vertices
			for (int n=0; n<ELEM.nNodes; n++) {
				centroid += _nodes[ELEM.nodeStart + n].vertex;
			}
			centroid /= (T) ELEM.nNodes;
			break;
		}


		//split the element
		splitElement(elem_idx, centroid, colorAsDeleted);
	}

}

