#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/color_manager.hpp"

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

#include <shared_mutex>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// A simple struct to pass any meshing options. Specified here so that it is easy to extend in the future.
	/////////////////////////////////////////////////
	struct TopologicalMeshOptions {
		const ColorMethod COLOR_METHOD = ColorMethod::GREEDY;
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
	template<BasicMeshNode Node_t=BasicNode<gv::util::Point<3,double>>, ColorableMeshElement Element_t=ColorableElement, ColorMethod COLOR_METHOD = ColorMethod::GREEDY>
	class TopologicalMesh
	{
	public:
		//common typedefs
		template <int n>
		using Index_t       = gv::util::Point<n,size_t>;
		using Vertex_t      = Node_t::Vertex_t;
		using NodeList_t    = NodeOctree<Node_t, 64>;
		using Box_t         = gv::util::Box<3, typename Node_t::Scalar_t>;
		using ElementList_t = std::vector<Element_t>;

	protected:
		ElementList_t              _elements;        //track element types and which block of _elem2node belongs to each element
		std::vector<BasicElement>  _boundary;        //storage for the boundary faces
		NodeList_t                 _nodes;           //reference to the node list. use this reference so that the nodes can refer to the nodes of some other mesh.
		
		MeshColorManager<COLOR_METHOD, Element_t, 1024> _color_manager;   //used to manage the color of the elements

	private:
		mutable std::shared_mutex _rw_mtx;

	public:
		/// Default constructor.
		TopologicalMesh() : _elements(),_nodes(), _color_manager(_elements) {}


		/////////////////////////////////////////////////
		/// Get the total number of elements in the mesh. Marked as virtual as hierarchical meshes need to check for active elements.
		/////////////////////////////////////////////////
		virtual size_t nElems() const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			return _elements.size();
		}


		/////////////////////////////////////////////////
		/// Get the total number of nodes in the mesh.
		/////////////////////////////////////////////////
		size_t nNodes() const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			return _nodes.size();
		}

		/////////////////////////////////////////////////
		/// Mesh a box using voxels of equal size
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void setVoxelMesh(const Box_t &domain, const Index_t<3>& N);


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// This is a virtual method as it must be changed for a hierarchical mesh.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		virtual void getElementNeighborsUnlocked(const size_t elem_idx, std::vector<size_t> &neighbors) const;


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// This is a virtual method as it must be changed for a hierarchical mesh.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		virtual void getElementNeighbors(const size_t elem_idx, std::vector<size_t> &neighbors) const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			getElementNeighborsUnlocked(elem_idx, neighbors);
		}


		/////////////////////////////////////////////////
		/// A method to get the active boundary faces that are also faces of the specified element.
		///
		/// @param elem_idx The index of the requested face (i.e., _boundary[elem_idx]).
		/// @param faces A reference to an existing vector where the result will be stored (via faces.push_back()).
		/////////////////////////////////////////////////
		void getBoundaryFaces(const size_t elem_idx, std::vector<size_t> &faces) const;


		/////////////////////////////////////////////////
		/// A method to prepare and create the nodes for a new element to be created. If a node at the specified vertex already exists, that index is used.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/// @param nodes The vector that will store the indices to the appropriate nodes
		/////////////////////////////////////////////////
		void prepare_nodes(const std::vector<Vertex_t> &vertices, const int vtkID, std::vector<size_t> &nodes);
		
		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		void insertElementThreadLocked(Element_t &ELEM);


		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// The method that calls this must ensure that it is done in a thread-safe way.
		/// If only one color of element is being inserted, then it will be safe.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will moved to _elements[elem_idx].
		/// @param elem_idx The inded where the element is to be inserted.
		/////////////////////////////////////////////////
		void insertElementUnlocked(Element_t &ELEM, const size_t elem_idx);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElementThreadLocked(const std::vector<Vertex_t> &vertices, const int vtkID);

		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElementUnlocked(const std::vector<Vertex_t> &vertices, const int vtkID, const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Color the specified element. Locked to a single thread.
		///
		/// @param elem_idx The index of the element to color
		/////////////////////////////////////////////////
		void colorWithLock(const size_t elem_idx) {
			std::unique_lock<std::shared_mutex> lock(_rw_mtx);
			std::vector<size_t> neighbors;
			getElementNeighbors(elem_idx, neighbors);
			_color_manager.setColorLockToSingleThread(elem_idx, neighbors);
		}


		/////////////////////////////////////////////////
		/// Color the specified element. Not locked to a single thread.
		/// The method that calls this must ensure tht _elements[elem_idx] is writable and
		///     _elements[k] is readible for any element k that is a neigbor of element elem_idx.
		///
		/// @param elem_idx The index of the element to color
		/////////////////////////////////////////////////
		void colorUnlocked(const size_t elem_idx) {
			std::vector<size_t> neighbors;
			getElementNeighbors(elem_idx, neighbors);
			_color_manager.setColorUnlocked(elem_idx, neighbors);
		}

		/////////////////////////////////////////////////
		/// Check if the coloring is valid
		/////////////////////////////////////////////////
		bool colors_are_valid() const;


		/////////////////////////////////////////////////
		/// Compute the boundary elements. The mesh must be in a conformal state (i.e., the coarsest mesh).
		/////////////////////////////////////////////////
		void compute_conformal_boundary();


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
		void save_as(const std::string filename, const bool include_details=false=true, const bool use_ascii=false) const;

		/// Friend function to print the mesh information
		template <BasicMeshNode U, ColorableMeshElement Element_u, ColorMethod COLORMETHOD>
		friend std::ostream& operator<<(std::ostream& os, const TopologicalMesh<U,Element_u,COLORMETHOD> &mesh);
	};



	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::setVoxelMesh(
			const Box_t &domain,
			const typename TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::Index_t<3>& N) {
		
		using Vertex_t = Node_t::Vertex_t;

		//reserve space
		_nodes.clear();
		_elements.clear();

		_nodes.reserve((N[0]+1) * (N[1]+1) * (N[2]+1));
		_elements.reserve(N[0]*N[1]*N[2]);
		
		//construct the mesh
		const Vertex_t H = domain.sidelength() / Vertex_t(N);
		#pragma omp parallel for collapse(3)
		for (size_t i=0; i<N[0]; i++) {
			for (size_t j=0; j<N[1]; j++) {
				for (size_t k=0; k<N[2]; k++) {
					//define element extents
					Vertex_t low  = domain.low() + Vertex_t{i,j,k} * H;
					Vertex_t high = domain.low() + Vertex_t{i+1,j+1,k+1} * H;
					Box_t   elem  {low, high};
				
					//assemble the list of vertices
					std::vector<Vertex_t> element_vertices(vtk_n_nodes(VOXEL_VTK_ID));
					for ( int l=0; l<8; l++) {element_vertices[l] = elem.voxelvertex(l);}

					//put the element into the mesh
					constructElementThreadLocked(element_vertices, VOXEL_VTK_ID);
				}
			}
		}
		compute_conformal_boundary();
	}


	
	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::getElementNeighborsUnlocked(const size_t elem_idx, std::vector<size_t> &neighbors) const {
		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique nodes
		std::unordered_set<size_t> neighbor_set;

		//loop through the nodes of the current element
		for (size_t n_idx : ELEM.nodes) {
			const Node_t &NODE = _nodes[n_idx];

			//loop through the elements of the current node
			for (size_t e_idx : NODE.elems) {
				if (e_idx!=elem_idx) {
					neighbor_set.insert(e_idx);
				}
			}
		}

		//convert the set to the vector
		neighbors.assign(neighbor_set.begin(), neighbor_set.end());
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::getBoundaryFaces(const size_t elem_idx, std::vector<size_t> &faces) const {

		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique nodes
		std::unordered_set<size_t> face_set;

		//loop through the nodes of the face
		for (size_t n_idx=0; n_idx<ELEM.nodes.size(); n_idx++) {
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
	

	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::prepare_nodes(const std::vector<typename TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::Vertex_t> &vertices,
			const int vtkID,
			std::vector<size_t> &nodes) {

		assert(vertices.size()==vtk_n_nodes(vtkID));
		//prepare the nodes vector
		nodes.resize(vtk_n_nodes(vtkID));

		//create new nodes as needed and aggregate their indices
		for (size_t n=0; n<vertices.size(); n++) {
			Node_t NODE(vertices[n]);
			size_t n_idx = _nodes.find(NODE); //_nodes has its own mutex lock (multiple threads can read)
			if (n_idx<_nodes.size()) {nodes[n]=n_idx;}
			else {
				[[maybe_unused]] int flag = _nodes.push_back(NODE, nodes[n]); //_nodes has its own mutex lock (single thread can write)
				assert(flag>=0);
			}
		}
	}
	

	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::insertElementThreadLocked(Element_t &ELEM) {
		std::unique_lock<std::shared_mutex> lock(_rw_mtx);

		//add the element to the mesh
		size_t e_idx = _elements.size(); //index of the new element

		//update existing nodes
		for (size_t node_idx : ELEM.nodes) {
			_nodes[node_idx].elems.push_back(e_idx);
		}

		//move ELEM to _elements
		_elements.push_back(std::move(ELEM));

		//color
		std::vector<size_t> neighbors;
		getElementNeighborsUnlocked(e_idx, neighbors); //lock is already active
		_color_manager.setColorThreadLocked(e_idx, neighbors);
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::insertElementUnlocked(Element_t &ELEM, const size_t elem_idx) {
		//The method that calls this must ensure that this is thread-safe.
		//If only one color of elements are being inserted, it will be safe

		//update existing nodes
		for (size_t node_idx : ELEM.nodes) {
			_nodes[node_idx].elems.push_back(elem_idx);
		}

		//move ELEM to _elements
		_elements[elem_idx] = std::move(ELEM);

		//color
		std::vector<size_t> neighbors;
		getElementNeighborsUnlocked(elem_idx, neighbors);
		_color_manager.setColorUnlocked(elem_idx, neighbors);
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::constructElementThreadLocked(
			const std::vector<typename TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::Vertex_t> &vertices,
			const int vtkID) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new nodes as needed and aggregate their indices
		prepare_nodes(vertices, vtkID, ELEM.nodes);
		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		insertElementThreadLocked(ELEM);
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::constructElementUnlocked(
			const std::vector<typename TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::Vertex_t> &vertices,
			const int vtkID,
			const size_t elem_idx) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new nodes as needed and aggregate their indices
		prepare_nodes(vertices, vtkID, ELEM.nodes);

		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		//this will finalize the logic of coloring and linking the nodes back to this element.
		insertElementUnlocked(ELEM, elem_idx);
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	bool TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::colors_are_valid() const {
		for (const Element_t& ELEM: _elements) {
			std::vector<size_t> neighbors;
			getElementNeighbors(ELEM.index, neighbors);
			for (size_t n_idx: neighbors) {
				if (ELEM.color == _elements[n_idx].color) {
					std::cout << "elements " << ELEM.index << " and " << n_idx << " color colision" << std::endl;
					return false;}
			}
		}
		return true;
	}

	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::compute_conformal_boundary() {
		using Vertex_t   = typename TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::Vertex_t;
		

		//create unordered maps to track the count of each face
		std::unordered_map<BasicElement, int, ElemHashBitPack> face_count;
		face_count.reserve(8*_elements.size()); //guess at the number of unique faces (exact if all elements are voxels or hexes)

		//loop through all elements and add the faces to the map
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			VTK_ELEMENT<Vertex_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Vertex_t>(ELEM);

			for (int i=0; i<vtk_n_faces(ELEM.vtkID); i++) {
				BasicElement FACE = vtk_elem->getFace(i);
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

				for (size_t i=0; i<FACE.nodes.size(); i++) {
					Node_t &NODE = _nodes[FACE.nodes[i]];
					NODE.boundary_faces.push_back(b_idx);
				}

				b_idx += 1;
			}
		}
		_boundary.shrink_to_fit();
	}



	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::print_topology_ascii_vtk(std::ostream &os) const {
		

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
		size_t nElements = nElems();
		#pragma omp parallel for reduction(+:nEntries)
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			nEntries  += 1 + _elements[e_idx].nodes.size();
		}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			buffer << ELEM.nodes.size();
			for (size_t n=0; n<ELEM.nodes.size(); n++) {
				buffer << " " << ELEM.nodes[n];
			}
			buffer << "\n";
		}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VTK_ID
		buffer << "CELL_TYPES " << nElements << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			buffer << ELEM.vtkID << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::print_topology_binary_vtk(const std::string& filename) const {
	    
	    
	    //only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(typename Node_t::Scalar_t)==4 or sizeof(typename Node_t::Scalar_t)==8, "Unsupported floating point size");

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
	    auto write_be_float = [&file](Node_t::Scalar_t value) {
	        if constexpr (sizeof(typename Node_t::Scalar_t) == 4) {
	            // 32-bit float
	            uint32_t temp;
	            std::memcpy(&temp, &value, sizeof(typename Node_t::Scalar_t));
	            uint32_t be_value = ((temp & 0xFF000000) >> 24) |
	                                ((temp & 0x00FF0000) >> 8)  |
	                                ((temp & 0x0000FF00) << 8)  |
	                                ((temp & 0x000000FF) << 24);
	            file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	        } else if constexpr (sizeof(typename Node_t::Scalar_t) == 8) {
	            // 64-bit double
	            uint64_t temp;
	            std::memcpy(&temp, &value, sizeof(typename Node_t::Scalar_t));
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
	    if constexpr (sizeof(typename Node_t::Scalar_t)==4) {file << "POINTS " << nNodes() << " float\n";}
	    else if constexpr (sizeof(typename Node_t::Scalar_t)==8) {file << "POINTS " << nNodes() << " double\n";}
	    
	    for (size_t i = 0; i < nNodes(); i++) {
	        write_be_float(_nodes[i].vertex[0]);
	        write_be_float(_nodes[i].vertex[1]);
	        write_be_float(_nodes[i].vertex[2]);
	    }
	    file << "\n";
	    
	    // ELEMENTS - calculate counts
	    size_t nEntries = 0;
	    size_t nElements = nElems();
	    #pragma omp parallel for reduction(+:nEntries)
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			nEntries  += 1 + _elements[e_idx].nodes.size();
	    }
	    
	    // CELLS (binary data)
	    file << "CELLS " << nElements << " " << nEntries << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			write_be_size_t(ELEM.nodes.size());
			for (size_t n = 0; n < ELEM.nodes.size(); n++) {
				write_be_size_t(ELEM.nodes[n]);
			}
	    }
	    file << "\n";
	    
	    // CELL_TYPES (binary data)
	    file << "CELL_TYPES " << nElements << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
	        const Element_t &ELEM = _elements[e_idx];
	        write_be_int(ELEM.vtkID);
	    }
	    file << "\n";
	    
	    file.close();
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::print_mesh_details_ascii_vtk(std::ostream &os) const {
		
		
		std::stringstream buffer;

		//NODE DETAILS
		int n_node_fields = 2;
		buffer << "POINT_DATA " << _nodes.size() << "\n";
		buffer << "FIELD node_info " << n_node_fields << "\n";

		//boundary
		size_t max_boundary_faces=0;
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
		size_t nElements = _elements.size();

		buffer << "CELL_DATA " << nElements << "\n";
		int n_fields = 3;
		buffer << "FIELD elem_info " << n_fields << "\n";

		
		//index
		buffer << "element_index 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			buffer << e_idx << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//color
		buffer << "color 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.color == (size_t) -1) {buffer << "-1 ";}
			else {buffer << ELEM.color << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//neighbors
		size_t max_neighbors=0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			getElementNeighbors(e_idx, neighbors[n_idx]);
			max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
			n_idx+=1;
		}
		buffer << "neighbors " << max_neighbors << " " << nElements << " integer\n";
		
		n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			size_t i;
			for (i=0; i<neighbors[n_idx].size(); i++) {buffer << neighbors[n_idx][i] << " ";}
			for (; i<max_neighbors; i++) {buffer << "-1 ";}
			n_idx+=1;
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::print_mesh_details_binary_vtk(const std::string& filename) const {
		//only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many nodes.
	    //additionally, the integers are expected to be signed in the legacy format. uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_nodes = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (_nodes.size()-1 > max_legacy_vtk_nodes) {throw std::runtime_error("Node index " + std::to_string(_nodes.size()-1) + " exceeds legacy VTK format limit.");}


		
		
		
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
		
		file << "POINT_DATA " << _nodes.size() << "\n";
		file << "FIELD node_info " << n_node_fields << "\n";
		
		//boundary
		size_t max_boundary_faces = 0;
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
		size_t nElements = nElems();
		file << "CELL_DATA " << nElements << "\n";
		int n_fields = 3;
		file << "FIELD elem_info " << n_fields << "\n";
		

		//index
		file << "element_index 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			write_be_size_t(e_idx);
		}
		file << "\n";
		
		//color
		file << "color 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			if (ELEM.color == (size_t) -1) {
				write_be_int(-1);
			} else {
				write_be_size_t(ELEM.color);
			}
		}
		file << "\n";
		
		//neighbors
		size_t max_neighbors = 0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			getElementNeighbors(e_idx, neighbors[n_idx]);
			max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
			n_idx += 1;
		}
		file << "neighbors " << max_neighbors << " " << nElements << " int\n";
		
		n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			size_t i;
			for (i = 0; i < neighbors[n_idx].size(); i++) {
				write_be_size_t(neighbors[n_idx][i]);
			}
			for (; i < max_neighbors; i++) {
				write_be_int(-1);
			}
			n_idx += 1;
		}
		file << "\n";
		
		file.close();
	}




	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	void TopologicalMesh<Node_t,Element_t,COLOR_METHOD>::save_as(const std::string filename, const bool include_details, const bool use_ascii) const {
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


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	std::ostream& operator<<(std::ostream& os, const TopologicalMesh<Node_t,Element_t,COLOR_METHOD> &mesh) {
		os << "nElems= " << mesh.nElems() << "\n";
		os << "nNodes= " << mesh.nNodes() << "\n";

		const size_t nColors = mesh._color_manager.nColors();
		os << "colors (" << nColors << ") : ";
		for (size_t c_idx=0; c_idx<nColors; c_idx++) {
			os << " " << mesh._color_manager.colorCount(c_idx);
		}
		os << "\n";
		
		return os;
	}
}

