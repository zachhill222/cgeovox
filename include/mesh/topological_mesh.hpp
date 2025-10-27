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
	struct ElemInfo {

		///Element type defined in the VTK documentation.
		const int vtkID;

		///How many nodes define this type of element
		const int nNodes;
		/////////////////////////////////////////////////
		/// The color of this element in the overall mesh. Used to allow parallel operations.
		/// The color is used as an index of std::vector, so it is of type size_t. Uncolored elements should have color=(size_t) -1.
		/////////////////////////////////////////////////
		size_t color;

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
		std::vector<size_t>     _colorCount;   //track how many elements belong to each color group. The index is the color (converted from size_t to int).

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
		/// To color the element, call recolor(elem_idx) where elem_idx is the index of the inserted element afterwards. Immediatly after insertion, elem_idx=nElems()-1.
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
		void splitElement(const size_t elem_idx, const bool colorAsDeleted=true);
		

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
		void print_mesh_details_ascii_vtk(std::ostream &os) const;
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file. If the file already exists, it will be over-written.
		/// Effectively this creates a file stream fs and calls print_topology_ascii_vtk(fs) and then (if include_details=true) calls print_mesh_details_ascii_vtk(fs).
		///
		/// @param filename The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology. This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, bool include_details=false) const;

		/////////////////////////////////////////////////
		/// Helper function to split VTK_VOXEL elements. Should not be called by the user.
		///
		/// @param mesh A reference to this mesh.
		/// @param color_count_increasing A vector of the existing colors sorted by increasing count.
		/// @param colorAsDeleted A flag when set to true colors the new elements as if the parent element (at elem_idx) has been deleted.
		/// @param elem_idx The index of the element being split
		/// @param sub_elem_idx The index of the element being created (from 0 to 7 for VTK_VOXEL elements)
		/////////////////////////////////////////////////
		friend void _SPLIT_VTK_VOXEL(TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T> &mesh,
			size_t (&node_idx)[27], const std::vector<size_t> &color_count_increasing, const bool colorAsDeleted, const size_t elem_idx, const int sub_elem_idx);
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
		_colorCount.clear();

		//loop through all elements
		for (size_t e_idx=0; e_idx<nElems(); e_idx++) {
			const auto& ELEM = _elemInfo[elem_idx];

			//if the color does not exist, append counts of 0 until the color does exist.
			while ((size_t) ELEM.color >= _colorCount.size()) {_colorCount.push_back(0);}

			//increment the color count
			_colorCount[(size_t ELEM.color)] += 1;
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
		ElemInfo_t ELEM {vtkID, N_NODES_PER_ELEMENT, -1, _elem2node.size()};
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

		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		insertElement(node_idx, vtkID);
	}


	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::splitElement(const size_t elem_idx, const bool colorAsDeleted) {
		using ElemInfo_t = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::ElemInfo_t;
		using Point_t    = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Point_t;
		using Node_t     = typename TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>::Node_t;


		//create 
		const ElemInfo_t& ELEM = _elemInfo[elem_idx];
		switch (ELEM.vtkID) {
		case 11: //VTK_VOXEL
			assert(dim==3);
			//set up 27 vertices required for the refined elements
			Point_t new_coords[27];
			
			//original (corners)
			for (int n=0; n<8; n++) {new_coords[n] = _nodes[ELEM.nodeStart + n].vertex;}

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
				Node_t NODE { {}, 0, -1, new_coords[i]};
				int [[maybe_unused]] flag = _nodes.push_back(NODE, node_idx[i]); //existing nodes will not be overwritten
				assert(flag>=0);
			}

			//insert and color the new elements
			size_t elem_nodes[8]; //storage for element node indices
			std::vector<size_t> neighbors; //storage for neighbor elements
			std::vector<bool> unsorted_color_allowed; //storage for tracking if a color is allowed. refers to unsorted colors.

			std::vector<size_t> color_count_increasing(_colorCount.size(), 0);
			for (size_t i=0; i<_colorCount.size(); i++) {color_count_increasing[i]=i;} //increasing by color labels
			auto color_count_compare = [_colorCount&](const size_t a, const size_t b) {return _colorCount[a]<_colorCount[b];} //comparator to sort by increasing color count
			std::sort(color_count_increasing.begin(), color_count_increasing.end(), color_count_compare); //color_count_increasing is in increasing order now
			
			for (int i=0; i<8; i++) {_SPLIT_VTK_VOXEL(*this, node_idx, color_count_increasing, colorAsDeleted, elem_idx, i);}
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
	template <int dim, int MAX_ELEMENT_PER_NODE, Float T>
	void _SPLIT_VTK_VOXEL(TopologicalMesh<dim,MAX_ELEMENT_PER_NODE,T>& mesh,
		const size_t (&node_idx)[27],                      //indices of the nodes in the mesh that will define the new elements
		const std::vector<size_t> &color_count_increasing, //colors sorted by increasing color count (before any new elements are added or colored)
		const bool colorAsDeleted,                         //set to true if the child element can have the same color as the original
		const size_t elem_idx,                             //index of element being split
		const int sub_elem_idx)                            //which sub-element is being created
	{
		//define and insert the new element
		size_t elem_nodes[8];
		switch (sub_elem_idx) {
			case (0): //voxel element containing original vertex 0
				elem_nodes[0] = node_idx[ 0]; //0
				elem_nodes[1] = node_idx[ 8]; //0-1
				elem_nodes[2] = node_idx[11]; //0-2
				elem_nodes[3] = node_idx[24]; //0-3
				elem_nodes[4] = node_idx[12]; //0-4
				elem_nodes[5] = node_idx[23]; //0-5
				elem_nodes[6] = node_idx[20]; //0-6
				elem_nodes[7] = node_idx[26]; //0-7
				break;

			case (1): //voxel element containing original vertex 1
				elem_nodes[0] = node_idx[ 8]; //0-1
				elem_nodes[1] = node_idx[ 1]; //1
				elem_nodes[2] = node_idx[24]; //0-3
				elem_nodes[3] = node_idx[ 9]; //1-3
				elem_nodes[4] = node_idx[23]; //0-5
				elem_nodes[5] = node_idx[15]; //1-5
				elem_nodes[6] = node_idx[26]; //0-7
				elem_nodes[7] = node_idx[21]; //1-7
				break;

			case (2): //voxel element containing original vertex 2
				elem_nodes[0] = node_idx[11]; //0-2
				elem_nodes[1] = node_idx[24]; //0-3
				elem_nodes[2] = node_idx[ 2]; //2
				elem_nodes[3] = node_idx[10]; //2-3
				elem_nodes[4] = node_idx[20]; //0-6
				elem_nodes[5] = node_idx[26]; //0-7
				elem_nodes[6] = node_idx[13]; //2-6
				elem_nodes[7] = node_idx[22]; //2-7
				break;

			case (3): //voxel element containing original vertex 3
				elem_nodes[0] = node_idx[24]; //0-3
				elem_nodes[1] = node_idx[ 9]; //1-3
				elem_nodes[2] = node_idx[10]; //2-3
				elem_nodes[3] = node_idx[ 3]; //3
				elem_nodes[4] = node_idx[26]; //0-7
				elem_nodes[5] = node_idx[21]; //1-7
				elem_nodes[6] = node_idx[22]; //2-7
				elem_nodes[7] = node_idx[14]; //3-7
				break;

			case (4): //voxel element containing original vertex 4
				elem_nodes[0] = node_idx[12]; //0-4
				elem_nodes[1] = node_idx[23]; //0-5
				elem_nodes[2] = node_idx[20]; //0-6
				elem_nodes[3] = node_idx[26]; //0-7
				elem_nodes[4] = node_idx[ 4]; //4
				elem_nodes[5] = node_idx[16]; //4-5
				elem_nodes[6] = node_idx[19]; //4-6
				elem_nodes[7] = node_idx[25]; //4-7
				break;

			case (5): //voxel element containing original vertex 5
				elem_nodes[0] = node_idx[23]; //0-5
				elem_nodes[1] = node_idx[15]; //1-5
				elem_nodes[2] = node_idx[26]; //0-7
				elem_nodes[3] = node_idx[21]; //1-7
				elem_nodes[4] = node_idx[16]; //4-5
				elem_nodes[5] = node_idx[ 5]; //5
				elem_nodes[6] = node_idx[25]; //4-7
				elem_nodes[7] = node_idx[17]; //5-7
				break;

			case (6): //voxel element containing original vertex 6
				elem_nodes[0] = node_idx[20]; //0-6
				elem_nodes[1] = node_idx[26]; //0-7
				elem_nodes[2] = node_idx[13]; //2-6
				elem_nodes[3] = node_idx[22]; //2-7
				elem_nodes[4] = node_idx[19]; //4-6
				elem_nodes[5] = node_idx[25]; //4-7
				elem_nodes[6] = node_idx[ 6]; //6
				elem_nodes[7] = node_idx[18]; //6-7
				break;

			case (7): //voxel element containing original vertex 7
				elem_nodes[0] = node_idx[26]; //0-7
				elem_nodes[1] = node_idx[21]; //1-7
				elem_nodes[2] = node_idx[22]; //2-7
				elem_nodes[3] = node_idx[14]; //3-7
				elem_nodes[4] = node_idx[25]; //4-7
				elem_nodes[5] = node_idx[17]; //5-7
				elem_nodes[6] = node_idx[18]; //6-7
				elem_nodes[7] = node_idx[ 7]; //7
				break;
		}
		
		mesh.insertElement(elem_nodes, 11);
		auto &ELEM = mesh._elemInfo[mesh.nElems()-1];
		

		//get the neighbors of the new element
		std::vector<size_t> neighbors;
		getElementNeighbors(mesh.nElems()-1, neighbors);

		//get which existing colors are allowed for the new element
		std::vector<bool> unsorted_color_allowed(color_count_increasing.size(), true); //compute the allowed colors
		size_t n_allowed = unsorted_color_allowed.size();
		assert(n_allowed>0);
		for (size_t i=0; i<neighbors.size(); i++) {
			if (colorAsDeleted and neighbors[i]==elem_idx) {continue;} //the 'parent' element does not count as a neighbor
			const auto NEIGHBOR = mesh._elemInfo[neighbors[i]];

			//if a neighbor has color i, then color i is not allowed for this element
			if (NEIGHBOR.color < color_count_increasing.size()) { //uncolored elements will be out of range and should be skipped
				unsorted_color_allowed[NEIGHBOR.color] = false;
				n_allowed -= 1;
				if (n_allowed==0) {break;} //we need a new color. note that no other new element can share this color.
			}
		}

		//color the new element
		if (n_allowed==0) { //create a new color
			ELEM.color = mesh._colorCount.size();
			mesh._colorCount.push_back(1);
		} else { //use the best existing color
			for (size_t i=0; i<unsorted_color_allowed.size(); i++) {
				if (unsorted_color_allowed[i]) {
					size_t color_idx = color_count_increasing[i];
					ELEM.color = color_idx;
					mesh._colorCount[color_idx] += 1;
					break;
				}
			}
		}
	}
}

