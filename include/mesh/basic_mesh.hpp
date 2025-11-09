#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_iterator.hpp"
#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>

#include <cassert>
#include <cstring>

#include <sstream>
#include <iostream>
#include <fstream>

#include <shared_mutex>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// This class defines the basic mesh operations and data structure. It provides some methods that are thread safe by locking a mutex
	/// as well as "unlocked" methods that can be **carefully** used in parallel. Using the unlocked methods on some _element[k] (one k per thread)
	/// is generally safe so long as none of the elements are neighbors. This can be guarenteed using a colored mesh (not implemented in this class).
	///
	/// The locked methods are appended by _ThreadLocked while the unlocked versions are appended by _Unlocked.
	/// Methods that do not care about threads are appended by _Safe.
	///
	/// Some methods that are locked will still run in parallel if they are read only, but others will be locked to a single thread if they write to shared memory.
	///
	/// Note that below, the "type of element" does not refer to Voxel/Quad/... It refers to the data stored in the struct (color, is_active, children, ...)
	/// All element types (in the C++ sense) contain an "int vtkID" and "std::vector<size_t> nodes" fields to track their type (in the FEM sense).
	///
	/// @tparam Node_t The type of node to use. Usually BasicNode<gv::util::Point<3,double>>.
	/// @tparam Element_t The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam Face_t The type of boundary element to use. This is usually set by the class that inherits from this class.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template<BasicMeshNode    Node_t   =BasicNode<gv::util::Point<3,double>>,
			 BasicMeshElement Element_t=BasicElement,
			 BasicMeshElement Face_t   =BasicElement>
	class BasicMesh	{
		/// Make the ElementIterator class a friend
		template<typename M, ContainerType C>
		friend class ElementIterator;

	public:
		//aliases
		template<int n=3>
		using Index_t            = gv::util::Point<n,size_t>;
		template<int n=3>
		using Box_t              = gv::util::Box<n, typename Node_t::Scalar_t>;
		using Vertex_t           = Node_t::Vertex_t;
		using NodeList_t         = NodeOctree<Node_t, 64>;
		using ElementIterator_t  = ElementIterator<BasicMesh<Node_t,Element_t,Face_t>, ContainerType::ELEMENTS>;
		using BoundaryIterator_t = ElementIterator<BasicMesh<Node_t,Element_t,Face_t>, ContainerType::BOUNDARY>;

		//template aliases are useful for other classes
		using node_type    = Node_t;
		using element_type = Element_t;
		using face_type    = Face_t;
	protected:
		/////////////////////////////////////////////////
		/// Container for the elements.
		/////////////////////////////////////////////////
		std::vector<Element_t> _elements;
		
		/////////////////////////////////////////////////
		/// Container for the boundary faces. Usually this is determined by the class that inherits from this.
		/////////////////////////////////////////////////
		std::vector<Face_t> _boundary;
		
		/////////////////////////////////////////////////
		/// Container for the mesh nodes. This has an octree structure for quickly finding nodes from their location in space.
		/// This method has parallel read (_nodes.find) and serial write (_nodes.push_back) built in.
		/////////////////////////////////////////////////
		NodeList_t _nodes;

	private:
		mutable std::shared_mutex _rw_mtx;

	public:
		/// Default constructor.
		BasicMesh() : _elements(),_nodes() {}
		BasicMesh(const Box_t<3> &domain, const Index_t<3> &N) : _elements(), _nodes(domain) {setVoxelMesh_ThreadLocked(domain, N);}
		BasicMesh(const Box_t<2> &domain, const Index_t<2> &N) : _elements(), _nodes() {
			Vertex_t low, high;
			low[0]  = domain.low()[0];  low[1]  = domain.low()[1];  low[2]  = -1.0;
			high[0] = domain.high()[0]; high[1] = domain.high()[1]; high[2] =  1.0;
			_nodes.set_bbox(Box_t{low,high});

			setPixelMesh_ThreadLocked(domain, N);
		}


		/////////////////////////////////////////////////
		/// Get the total number of elements in the mesh. Marked as virtual as hierarchical meshes need to check for active elements.
		/////////////////////////////////////////////////
		virtual size_t nElems_ThreadLocked() const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			return _elements.size();
		}


		/////////////////////////////////////////////////
		/// Get the total number of nodes in the mesh.
		/////////////////////////////////////////////////
		size_t nNodes_ThreadLocked() const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			return _nodes.size();
		}


		/////////////////////////////////////////////////
		/// Mesh a 3D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void setVoxelMesh_ThreadLocked(const Box_t<3> &domain, const Index_t<3>& N);


		/////////////////////////////////////////////////
		/// Mesh a 2D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void setPixelMesh_ThreadLocked(const Box_t<2> &domain, const Index_t<2>& N);


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// This is a virtual method as it must be changed for a hierarchical mesh.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		virtual void getElementNeighbors_Unlocked(const size_t elem_idx, std::vector<size_t> &neighbors) const;


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		/// The neighbors vector will be sorted and made unique before the method returns.
		///
		/// This is a virtual method as it must be changed for a hierarchical mesh.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		virtual void getElementNeighbors_ThreadLocked(const size_t elem_idx, std::vector<size_t> &neighbors) const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			getElementNeighbors_Unlocked(elem_idx, neighbors);
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
		void prepareNodes_Safe(const std::vector<Vertex_t> &vertices, const int vtkID, std::vector<size_t> &nodes);
		

		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		void insertElement_ThreadLocked(Element_t &ELEM);


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
		void insertElement_Unlocked(Element_t &ELEM, const size_t elem_idx);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElement_ThreadLocked(const std::vector<Vertex_t> &vertices, const int vtkID);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing nodes.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElement_Unlocked(const std::vector<Vertex_t> &vertices, const int vtkID, const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Compute the boundary elements. The mesh must be in a conformal state (i.e., the coarsest mesh).
		/////////////////////////////////////////////////
		void computeConformalBoundary_ThreadLocked();


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
		/// Save the mesh to a file.
		///
		/// @param filename        The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology.
		///                        This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, const bool include_details=false=true, const bool use_ascii=false) const;


		/////////////////////////////////////////////////
		/// Friend function to print the mesh information
		/////////////////////////////////////////////////
		template <BasicMeshNode U, BasicMeshElement Element_u, BasicMeshElement Face_u>
		friend std::ostream& operator<<(std::ostream& os, const BasicMesh<U,Element_u,Face_u> &mesh);


		/////////////////////////////////////////////////
		/// Check if an element is valid. This is so that derived classes do not need to re-implement the entire iterator if
		/// not all elements stored in _elements should be considered part of the mesh (e.g., hierarchical meshes)
		///
		/// @param ELEM The element to check.
		/////////////////////////////////////////////////
		virtual bool isElementValid(const Element_t &ELEM) const {return true;}
		

		/////////////////////////////////////////////////
		/// Check if a boundary face is valid. This is so that derived classes do not need to re-implement the entire iterator if
		/// not all elements stored in _elements should be considered part of the mesh (e.g., hierarchical meshes)
		///
		/// @param FACE The face to check.
		/////////////////////////////////////////////////
		virtual bool isFaceValid(const Face_t &FACE) const {return true;}


		/////////////////////////////////////////////////
		/// Iterators for _elements
		/////////////////////////////////////////////////
		ElementIterator_t elemBegin() {
			ElementIterator_t iter(this,0);
			iter.moveToBegin();
			return iter;
		}

		ElementIterator_t elemEnd() {return ElementIterator_t(this, _elements.size());}

	
		/////////////////////////////////////////////////
		/// Iterators for _boundary
		/////////////////////////////////////////////////
		BoundaryIterator_t boundaryBegin() {
			BoundaryIterator_t iter(this,0);
			iter.moveToBegin();
			return iter;
		}

		BoundaryIterator_t boundaryEnd() {return BoundaryIterator_t(this, _boundary.size());}
	};
	static_assert(BasicMeshType< BasicMesh<BasicNode<gv::util::Point<3,double>>, BasicElement, BasicElement >>,
		"BasicMesh is not a BasicMeshType with default template parameters.");



	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::setVoxelMesh_ThreadLocked(
			const Box_t<3> &domain,
			const typename BasicMesh<Node_t,Element_t,Face_t>::Index_t<3>& N) {
		
		using Vertex_t = Node_t::Vertex_t;

		//reserve space
		_nodes.clear();
		_elements.clear();

		_nodes.reserve((N[0]+1) * (N[1]+1) * (N[2]+1));
		_elements.reserve(N[0]*N[1]*N[2]);
		
		//construct the mesh
		const Vertex_t H = domain.sidelength() / Vertex_t(N);
		for (size_t i=0; i<N[0]; i++) {
			for (size_t j=0; j<N[1]; j++) {
				for (size_t k=0; k<N[2]; k++) {
					//define element extents
					Vertex_t low  = domain.low() + Vertex_t{i,j,k} * H;
					Vertex_t high = domain.low() + Vertex_t{i+1,j+1,k+1} * H;
					Box_t   elem  {low, high};
				
					//assemble the list of vertices
					std::vector<Vertex_t> element_vertices(vtk_n_nodes(VOXEL_VTK_ID));
					for (size_t l=0; l<vtk_n_nodes(VOXEL_VTK_ID); l++) {
						element_vertices[l] = elem.voxelvertex(l);
					}

					//put the element into the mesh
					constructElement_ThreadLocked(element_vertices, VOXEL_VTK_ID);
				}
			}
		}
		computeConformalBoundary_ThreadLocked();
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::setPixelMesh_ThreadLocked(
			const Box_t<2> &domain,
			const typename BasicMesh<Node_t,Element_t,Face_t>::Index_t<2>& N) {
		
		using Vertex_t = Node_t::Vertex_t;

		//reserve space
		_nodes.clear();
		_elements.clear();

		_nodes.reserve((N[0]+1) * (N[1]+1));
		_elements.reserve(N[0]*N[1]);
		
		//construct the mesh
		const Vertex_t H = domain.sidelength() / Vertex_t(N);
		for (size_t i=0; i<N[0]; i++) {
			for (size_t j=0; j<N[1]; j++) {
				//define element extents
				Vertex_t low  = domain.low() + Vertex_t{i,j} * H;
				Vertex_t high = domain.low() + Vertex_t{i+1,j+1} * H;
				Box_t   elem  {low, high};
			
				//assemble the list of vertices
				std::vector<Vertex_t> element_vertices(vtk_n_nodes(PIXEL_VTK_ID));
				for (size_t l=0; l<vtk_n_nodes(PIXEL_VTK_ID); l++) {
					element_vertices[l] = elem.voxelvertex(l);
				}

				//put the element into the mesh
				constructElement_ThreadLocked(element_vertices, PIXEL_VTK_ID);
			}
		}
		computeConformalBoundary_ThreadLocked();
	}


	
	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::getElementNeighbors_Unlocked(const size_t elem_idx, std::vector<size_t> &neighbors) const {
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


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::getBoundaryFaces(const size_t elem_idx, std::vector<size_t> &faces) const {

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
	

	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::prepareNodes_Safe(const std::vector<typename BasicMesh<Node_t,Element_t,Face_t>::Vertex_t> &vertices,
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
	

	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::insertElement_ThreadLocked(Element_t &ELEM) {
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
		getElementNeighbors_Unlocked(e_idx, neighbors); //lock is already active
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::insertElement_Unlocked(Element_t &ELEM, const size_t elem_idx) {
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
		getElementNeighbors_Unlocked(elem_idx, neighbors);
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::constructElement_ThreadLocked(
			const std::vector<typename BasicMesh<Node_t,Element_t,Face_t>::Vertex_t> &vertices,
			const int vtkID) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new nodes as needed and aggregate their indices
		prepareNodes_Safe(vertices, vtkID, ELEM.nodes);
		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		insertElement_ThreadLocked(ELEM);
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::constructElement_Unlocked(
			const std::vector<typename BasicMesh<Node_t,Element_t,Face_t>::Vertex_t> &vertices,
			const int vtkID,
			const size_t elem_idx) {
		assert(vertices.size()==vtk_n_nodes(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new nodes as needed and aggregate their indices
		prepareNodes_Safe(vertices, vtkID, ELEM.nodes);

		//now that the nodes are initialized, insert the element.
		//the nodes will be updated to link back to the new element.
		//this will finalize the logic of coloring and linking the nodes back to this element.
		insertElement_Unlocked(ELEM, elem_idx);
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::computeConformalBoundary_ThreadLocked() {
		std::unique_lock<std::shared_mutex> lock(_rw_mtx);

		using Vertex_t   = typename BasicMesh<Node_t,Element_t,Face_t>::Vertex_t;
		

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


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::save_as(const std::string filename, const bool include_details, const bool use_ascii) const {
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
	


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	std::ostream& operator<<(std::ostream& os, const BasicMesh<Node_t,Element_t,Face_t> &mesh) {
		os << "nElems= " << mesh.nElems_ThreadLocked() << "\n";
		os << "nNodes= " << mesh.nNodes_ThreadLocked() << "\n";
		
		return os;
	}
}

