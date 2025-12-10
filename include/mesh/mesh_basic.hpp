#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_iterator.hpp"
#include "mesh/mesh_vtk_out.hpp"
#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <cassert>
#include <cstring>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <shared_mutex>
#include <mutex>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// This class defines the basic mesh operations and data structure. It provides some methods that are thread safe by locking a mutex
	/// as well as "unlocked" methods that can be **carefully** used in parallel. Using the unlocked methods on some _element[k] (one k per thread)
	/// is generally safe so long as none of the elements are neighbors. This can be guarenteed using a colored mesh (not implemented in this class).
	///
	/// The locked methods are appended by _Locked while the unlocked versions are appended by _Unlocked.
	/// Methods that do not care about threads are appended by _Safe.
	///
	/// Some methods that are locked will still run in parallel if they are read only, but others will be locked to a single thread if they write to shared memory.
	///
	/// Note that below, the "type of element" does not refer to Voxel/Quad/... It refers to the data stored in the struct (color, is_active, children, ...)
	/// All element types (in the C++ sense) contain an "int vtkID" and "std::vector<size_t> vertices" fields to track their type (in the FEM sense).
	///
	/// @tparam Vertex_t The type of node to use. Usually BasicVertex<gv::util::Point<3,T>>. Where T is a class that emulates the real numbers
	///                    with a stable comparison (e.g., FixedPrecision rather than double).
	/// @tparam Element_t The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam Face_t The type of boundary element to use. This is usually set by the class that inherits from this class.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template<BasicMeshVertex    Vertex_t  = BasicVertex<gv::util::Point<3,float>>,
			 BasicMeshElement Element_t = BasicElement,
			 BasicMeshElement Face_t    = BasicElement>
	class BasicMesh	{
		/// Make the ElementIterator class a friend
		template<typename M, ContainerType C>
		friend class ElementIterator;

		/// Make the MeshView class a friend
		template<BasicMeshVertex Vertex_u, BasicMeshElement Element_u, BasicMeshElement Face_u>
		friend class MeshView;

		static_assert(HierarchicalMeshElement<Element_t> ? HierarchicalMeshElement<Face_t> : true, "If Element_t is Hierarchical, then Face_t must be Hierarchical");
		static_assert(HierarchicalMeshElement<Face_t> ? HierarchicalMeshElement<Element_t> : true, "If Face_t is Hierarchical, then Element_t must be Hierarchical");
		static_assert(Vertex_t::Point_t::dimension==3, "All meshes are considered to be embedded in three dimensions.");
	public:
		//aliases
		using OctreeScalar_t     = typename Vertex_t::Scalar_t;
		template<int n=3>
		using Index_t            = gv::util::Point<n,size_t>;
		template<int n=3>
		using Box_t              = gv::util::Box<n, OctreeScalar_t>;
		using Point_t            = Vertex_t::Point_t;
		using VertexList_t       = NodeOctree<Vertex_t, 64, OctreeScalar_t>;
		using ElementIterator_t  = ElementIterator<BasicMesh<Vertex_t,Element_t,Face_t>, ContainerType::ELEMENTS>;
		using BoundaryIterator_t = ElementIterator<BasicMesh<Vertex_t,Element_t,Face_t>, ContainerType::BOUNDARY>;

		//template aliases are useful for other classes
		using node_type    = Vertex_t;
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
		/// For each boundary face, track which element it belonged to and which face of that element it is.
		/////////////////////////////////////////////////
		std::vector<FaceTracker> _boundary_track;

		/////////////////////////////////////////////////
		/// Container for the mesh vertices. This has an octree structure for quickly finding vertices from their location in space.
		/// This method has parallel read (_vertices.find) and serial write (_vertices.push_back) built in.
		/////////////////////////////////////////////////
		VertexList_t _vertices;

	private:
		mutable std::shared_mutex _rw_mtx;

	public:
		BasicMesh() : _elements(),_vertices() {}
		BasicMesh(const Box_t<3> &domain) : _elements(), _vertices(1.125*domain) {}
		BasicMesh(const Box_t<2> &domain) : _elements(), _vertices() {
			Point_t low, high;
			low[0]  = domain.low()[0];  low[1]  = domain.low()[1];  low[2]  = -1.0;
			high[0] = domain.high()[0]; high[1] = domain.high()[1]; high[2] =  1.0;
			_vertices.set_bbox(1.125*Box_t{low,high});
		}
		BasicMesh(const Box_t<3> &domain, const Index_t<3> &N, const bool useIsopar=false) : BasicMesh(domain) {setVoxelMesh_Locked(domain, N, useIsopar);}
		BasicMesh(const Box_t<2> &domain, const Index_t<2> &N, const bool useIsopar=false) : BasicMesh(domain) {setPixelMesh_Locked(domain, N, useIsopar);}
		virtual ~BasicMesh() {}

		/////////////////////////////////////////////////
		/// Get the total number of elements in the mesh. Marked as virtual as hierarchical meshes need to check for active elements.
		/////////////////////////////////////////////////
		virtual size_t nElems() const {
			return _elements.size();
		}


		/////////////////////////////////////////////////
		/// Get the total number of vertices in the mesh.
		/////////////////////////////////////////////////
		virtual size_t nNodes() const {
			return _vertices.size();
		}


		/////////////////////////////////////////////////
		/// Get the total number of boundary faces in the mesh.
		/////////////////////////////////////////////////
		virtual size_t nBoundaryFaces() const {
			return _boundary.size();
		}


		/////////////////////////////////////////////////
		/// Read elements and vertices externally
		/////////////////////////////////////////////////
		const Vertex_t&    getNode(const size_t idx) const {return _vertices[idx];}
		const Element_t& getElement(const size_t idx) const {return _elements[idx];}
		const Face_t&    getBoundaryFace(const size_t idx) const {return _boundary[idx];}
		const Box_t<3>   bbox() const {return _vertices.bbox();}
		const VertexList_t& getNodeOctree() const {return _vertices;}

		/////////////////////////////////////////////////
		/// Allocate space
		/////////////////////////////////////////////////
		inline void reserveElements(const size_t length) {_elements.reserve(length);}
		inline void reserveNodes(const size_t length) {_vertices.reserve(length);}
		inline void reserveBoundary(const size_t length) {_boundary.reserve(length); _boundary_track.reserve(length);}

		/////////////////////////////////////////////////
		/// Get the boundary as a separate mesh
		/////////////////////////////////////////////////
		template<BasicMeshType Mesh_t>
		void getBoundaryMesh(Mesh_t &mesh) const {
			for (auto it=boundaryBegin(); it!=boundaryEnd(); ++it) {
				std::vector<Point_t> vertices;
				for (size_t n : it->vertices) {vertices.push_back(_vertices[n].coord);}
				mesh.constructElement_Locked(vertices, it->vtkID);
			}
		}

		/////////////////////////////////////////////////
		/// Mesh a 3D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void setVoxelMesh_Locked(const Box_t<3> &domain, const Index_t<3>& N, const bool useIsopar=false);


		/////////////////////////////////////////////////
		/// Mesh a 2D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void setPixelMesh_Locked(const Box_t<2> &domain, const Index_t<2>& N, const bool useIsopar=false);


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
		virtual void getElementNeighbors_Locked(const size_t elem_idx, std::vector<size_t> &neighbors) const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			getElementNeighbors_Unlocked(elem_idx, neighbors);
		}


		/////////////////////////////////////////////////
		/// A method to get the active boundary faces that are also faces of the specified element.
		///
		/// @param elem_idx The index of the requested face (i.e., _boundary[elem_idx]).
		/// @param faces A reference to an existing vector where the result will be stored (via faces.push_back()).
		/////////////////////////////////////////////////
		virtual void getBoundaryFaces_Unlocked(const size_t elem_idx, std::vector<size_t> &faces) const;


		/////////////////////////////////////////////////
		/// A method to get the active boundary faces that are also faces of the specified element.
		///
		/// @param elem_idx The index of the requested face (i.e., _boundary[elem_idx]).
		/// @param faces A reference to an existing vector where the result will be stored (via faces.push_back()).
		/////////////////////////////////////////////////
		virtual void getBoundaryFaces_Locked(const size_t elem_idx, std::vector<size_t> &faces) const {
			std::shared_lock<std::shared_mutex> lock(_rw_mtx);
			getBoundaryFaces_Unlocked(elem_idx, faces);
		}


		/////////////////////////////////////////////////
		/// A method to prepare and create the vertices for a new element to be created. If a node at the specified vertex already exists, that index is used.
		///
		/// @param vertex_coords A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/// @param vertices The vector that will store the indices to the appropriate vertices
		/////////////////////////////////////////////////
		void prepareNodes_Safe(const std::vector<Point_t> &vertex_coords, const int vtkID, std::vector<size_t> &vertices);
		

		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing vertices.
		/// The existing vertices will be updated but no new vertices will be created.
		///
		/// Marked as virtual so classes that inherit can populate special fields (e.g., color).
		///
		/// @param ELEM The element to be inserted. The vertices must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		virtual void insertElement_Locked(Element_t &ELEM);
		void insertBoundaryFace_Locked(Face_t &FACE, FaceTracker &bt);

		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing vertices.
		/// The existing vertices will be updated but no new vertices will be created.
		///
		/// The method that calls this must ensure that it is done in a thread-safe way.
		/// If only one color of element is being inserted, then it will be safe.
		///
		/// Marked as virtual so classes that inherit can populate special fields (e.g., color).
		///
		/// @param ELEM The element to be inserted. The vertices must already be populated. The element will moved to _elements[elem_idx].
		/// @param elem_idx The inded where the element is to be inserted.
		/////////////////////////////////////////////////
		virtual void insertElement_Unlocked(Element_t &ELEM, const size_t elem_idx);
		void insertBoundaryFace_Unlocked(Face_t &FACE, const size_t face_idx, FaceTracker &bt);

		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing vertices.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElement_Locked(const std::vector<Point_t> &vertices, const int vtkID);


		/////////////////////////////////////////////////
		/// A method to create a new element by its vertices and insert it into the mesh. The element is constructed from specified vertices, which may or may not correspond to existing vertices.
		/// If a vertex corresponds to an existing node, that node will be updated. Otherwise a new node will be created.
		///
		/// @param vertices A reference to an existing vector of vertices (usually of type gv::util::Point<3,double>) that define the new element. These must be in the proper order.
		/// @param vtkID The vtk identifier to track the type of element. Look up the vtk documentation to see which node order is required.
		/////////////////////////////////////////////////
		void constructElement_Unlocked(const std::vector<Point_t> &vertices, const int vtkID, const size_t elem_idx);


		/////////////////////////////////////////////////
		/// Compute the boundary elements. The mesh must be in a conformal state (i.e., the coarsest mesh).
		/////////////////////////////////////////////////
		void computeConformalBoundary();


		/////////////////////////////////////////////////
		/// Method to move a vertex in the mesh. If any of the elements associated with this node
		/// are not isoparametric, they are converted to their isoparametric versions.
		///
		/// @param vertex_idx  The index of the node that will be moved
		/// @param new_coord   The coordinate where the node will be moved to
		/////////////////////////////////////////////////
		void moveVertex(const size_t vertex_idx, Point_t new_coord) {
			Vertex_t &VERTEX = _vertices[vertex_idx];
			for (size_t e_idx : VERTEX.elems) {makeIsoparametric(_elements[e_idx]);}
			for (size_t f_idx : VERTEX.boundary_faces) {makeIsoparametric(_boundary[f_idx]);}
			
			VERTEX.coord = new_coord;
			_vertices.reinsert(vertex_idx);
		}
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file.
		///
		/// @param filename        The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology.
		///                        This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, const bool include_details=false, const bool use_ascii=false) const;


		/////////////////////////////////////////////////
		/// Friend functions to print the mesh information
		/////////////////////////////////////////////////
		template <BasicMeshVertex U, BasicMeshElement Element_u, BasicMeshElement Face_u>
		friend std::ostream& operator<<(std::ostream& os, const BasicMesh<U,Element_u,Face_u> &mesh);

		template<BasicMeshType Mesh_t>
		friend void print_topology_ascii_vtk(std::ofstream &file, const Mesh_t &mesh, const std::string description);

		template<BasicMeshType Mesh_t>
		friend void print_mesh_details_ascii_vtk(std::ofstream &file, const Mesh_t &mesh);

		template<BasicMeshType Mesh_t>
		friend void print_topology_binary_vtk(std::ofstream &file, const Mesh_t &mesh, const std::string description);

		template<BasicMeshType Mesh_t>
		friend void print_mesh_details_binary_vtk(std::ofstream &file, const Mesh_t &mesh);

		template<BasicMeshVertex Vertex_u, BasicMeshElement Element_u, BasicMeshElement Face_u>
		friend void memorySummary(const BasicMesh<Vertex_u,Element_u,Face_u> &mesh);

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
		/// Iterators for _elements. These are the most common iterators and get the defualt begin() and end() methods.
		/// Marked as virtual so they can be pointed at other arrays for views into the mesh. (i.e., the MeshView class).
		/////////////////////////////////////////////////
		virtual ElementIterator_t begin()       {return ElementIterator_t(this, 0);}
		virtual ElementIterator_t end()         {return ElementIterator_t(this, _elements.size());}
		virtual ElementIterator_t begin() const {return ElementIterator_t(const_cast<BasicMesh<Vertex_t,Element_t,Face_t>*>(this), 0);}
		virtual ElementIterator_t end()   const {return ElementIterator_t(const_cast<BasicMesh<Vertex_t,Element_t,Face_t>*>(this), _elements.size());}

	
		/////////////////////////////////////////////////
		/// Iterators for _boundary
		/////////////////////////////////////////////////
		virtual BoundaryIterator_t boundaryBegin()       {return BoundaryIterator_t(this,0);}
		virtual BoundaryIterator_t boundaryEnd()         {return BoundaryIterator_t(this, _boundary.size());}
		virtual BoundaryIterator_t boundaryBegin() const {return BoundaryIterator_t(const_cast<BasicMesh<Vertex_t,Element_t,Face_t>*>(this), 0);}
		virtual BoundaryIterator_t boundaryEnd()   const {return BoundaryIterator_t(const_cast<BasicMesh<Vertex_t,Element_t,Face_t>*>(this), _boundary.size());}


		/////////////////////////////////////////////////
		/// Iterators for _vertices
		/////////////////////////////////////////////////
		virtual std::vector<Vertex_t>::iterator nodeBegin()             {return _vertices.begin();}
		virtual std::vector<Vertex_t>::iterator nodeEnd()               {return _vertices.end();}
		virtual std::vector<Vertex_t>::const_iterator nodeBegin() const {return _vertices.cbegin();}
		virtual std::vector<Vertex_t>::const_iterator nodeEnd()   const {return _vertices.cend();}

	};
	static_assert(BasicMeshType< BasicMesh<BasicVertex<gv::util::Point<3,double>>, BasicElement, BasicElement >>,
		"BasicMesh is not a BasicMeshType with default template parameters.");



	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::setVoxelMesh_Locked(
			const Box_t<3> &domain,
			const typename BasicMesh<Vertex_t,Element_t,Face_t>::Index_t<3>& N,
			const bool useIsopar) {
		
		using Point_t = Vertex_t::Point_t;

		//switch to hex if needed
		int ID = VOXEL_VTK_ID;
		if (useIsopar) {ID=HEXAHEDRON_VTK_ID;}

		//reserve space
		_vertices.clear();
		_elements.clear();

		_vertices.reserve((N[0]+1) * (N[1]+1) * (N[2]+1));
		_elements.resize(N[0]*N[1]*N[2]);
		
		//initialize the vertices
		const Point_t H = domain.sidelength() / Point_t(N);
		for (size_t i=0; i<=N[0]; i++) {
			for (size_t j=0; j<=N[1]; j++) {
				for (size_t k=0; k<=N[2]; k++) {
					Point_t vertex  = domain.low() + Point_t{i,j,k} * H;
					Vertex_t VERTEX(vertex);
					size_t idx = _vertices.push_back(VERTEX);
					_vertices[idx].index = idx;
				}
			}
		}



		//construct the mesh
		//by staggering which elements we construct, we can make them in 8 parallel batches
		for (size_t ii=0; ii<2; ii++) {
			for (size_t jj=0; jj<2; jj++) {
				for (size_t kk=0; kk<2; kk++) {
					#pragma omp parallel for collapse(3) //after offsets, these groups do not interact (cubes have a known coloring)
					for (size_t i=ii; i<N[0]; i+=2) {
						for (size_t j=jj; j<N[1]; j+=2) {
							for (size_t k=kk; k<N[2]; k+=2) {
								//define element extents
								Point_t low  = domain.low() + Point_t{i,j,k} * H;
								Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;
								Box_t   elem  {low, high};
							
								//assemble the list of vertices
								std::vector<Point_t> element_vertices(vtk_n_vertices(ID));
								for (size_t l=0; l<vtk_n_vertices(ID); l++) {
									if (useIsopar) {element_vertices[l] = elem.hexvertex(l);}
									else {element_vertices[l] = elem.voxelvertex(l);}
								}

								//put the element into the mesh
								const size_t idx = i + N[0]*(j + k*N[1]);
								constructElement_Unlocked(element_vertices, ID, idx);
							}
						}
					}
				}
			}
		}

		computeConformalBoundary();
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::setPixelMesh_Locked(
			const Box_t<2> &domain,
			const typename BasicMesh<Vertex_t,Element_t,Face_t>::Index_t<2>& N,
			const bool useIsopar) {
		
		using Point_t = Vertex_t::Point_t;
		using Point_2 = gv::util::Point<2,typename Vertex_t::Scalar_t>;
		
		//switch to quad if needed
		int ID = PIXEL_VTK_ID;
		if (useIsopar) {ID=QUAD_VTK_ID;}


		//reserve space
		_vertices.clear();
		_elements.clear();

		_vertices.reserve((N[0]+1) * (N[1]+1));
		_elements.reserve(N[0]*N[1]);
		
		//construct the mesh
		const Point_2 H(domain.sidelength() / Point_2(N));
		for (size_t i=0; i<N[0]; i++) {
			for (size_t j=0; j<N[1]; j++) {
				//define element extents
				Point_2 low  = domain.low() + Point_2{i,j} * H;
				Point_2 high = domain.low() + Point_2{i+1,j+1} * H;
				Box_t<2>   elem  {low, high};
			
				//assemble the list of vertices
				std::vector<Point_t> element_vertices(vtk_n_vertices(ID));
				for (size_t l=0; l<vtk_n_vertices(ID); l++) {
					if (useIsopar) {element_vertices[l] = Point_t(elem.hexvertex(l));}
					else {element_vertices[l] = Point_t(elem.voxelvertex(l));}
				}

				//put the element into the mesh
				constructElement_Locked(element_vertices, ID);
			}
		}
		computeConformalBoundary();
	}


	
	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::getElementNeighbors_Unlocked(const size_t elem_idx, std::vector<size_t> &neighbors) const {
		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique vertices
		std::unordered_set<size_t> neighbor_set;

		//loop through the vertices of the current element
		for (size_t n_idx : ELEM.vertices) {
			const Vertex_t &VERTEX = _vertices[n_idx];

			//loop through the elements of the current node
			for (size_t e_idx : VERTEX.elems) {
				bool isNeighbor = e_idx!=elem_idx;
				if constexpr (HierarchicalMeshElement<Element_t>) {
					isNeighbor = isNeighbor and _elements[e_idx].is_active;
				}
				
				if (isNeighbor) {
					neighbor_set.insert(e_idx);
				}
			}
		}

		//convert the set to the vector
		neighbors.insert(neighbors.end(), neighbor_set.begin(), neighbor_set.end());
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::getBoundaryFaces_Unlocked(const size_t elem_idx, std::vector<size_t> &faces) const {

		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique vertices
		std::unordered_set<size_t> face_set;

		//loop through the vertices of the element
		for (size_t n_idx=0; n_idx<ELEM.vertices.size(); n_idx++) {
			const Vertex_t &VERTEX = _vertices[ELEM.vertices[n_idx]];

			//loop through the boundary faces of the current node
			for (size_t f_idx : VERTEX.boundary_faces) {
				if (_boundary_track[f_idx].elem_idx == elem_idx) {
					face_set.insert(f_idx);
				}
			}
		}

		faces.insert(faces.end(), face_set.begin(), face_set.end());
	}
	

	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::prepareNodes_Safe(
			const std::vector<typename BasicMesh<Vertex_t,Element_t,Face_t>::Point_t> &vertex_coords,
			const int vtkID,
			std::vector<size_t> &vertices) {

		assert(vertex_coords.size()==vtk_n_vertices(vtkID));
		//prepare the vertices vector
		vertices.resize(vtk_n_vertices(vtkID));

		//create new vertices as needed and aggregate their indices
		for (size_t n=0; n<vertex_coords.size(); n++) {
			Vertex_t VERTEX(vertex_coords[n]);
			VERTEX.index = _vertices.size();
			size_t n_idx = _vertices.push_back(VERTEX);
			vertices[n] = n_idx;
		}
	}
	

	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::insertElement_Locked(Element_t &ELEM) {
		std::unique_lock<std::shared_mutex> lock(_rw_mtx);

		//add the element to the mesh
		if constexpr (HierarchicalMeshElement<Element_t>) {ELEM.is_active=true;}
		size_t e_idx = _elements.size(); //index of the new element
		ELEM.index   = _elements.size();
		
		//update existing vertices
		for (size_t vertex_idx : ELEM.vertices) {
			_vertices[vertex_idx].elems.push_back(e_idx);
		}

		//move ELEM to _elements
		ELEM.index = _elements.size();
		_elements.push_back(std::move(ELEM));
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::insertElement_Unlocked(Element_t &ELEM, const size_t elem_idx) {
		//The method that calls this must ensure that this is thread-safe.
		//If only one color of elements are being inserted, it will be safe
		if constexpr (HierarchicalMeshElement<Element_t>) {ELEM.is_active=true;}

		ELEM.index = elem_idx;

		//update existing vertices
		for (size_t vertex_idx : ELEM.vertices) {
			_vertices[vertex_idx].elems.push_back(elem_idx);
		}

		//move ELEM to _elements
		_elements[elem_idx] = std::move(ELEM);
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::insertBoundaryFace_Locked(Face_t &FACE, FaceTracker &bt) {
		std::unique_lock<std::shared_mutex> lock(_rw_mtx);

		//add the element to the mesh
		if constexpr (HierarchicalMeshElement<Face_t>) {FACE.is_active=true;}
		size_t e_idx = _boundary.size(); //index of the new element
		FACE.index   = _boundary.size();
		
		//update existing vertices
		for (size_t vertex_idx : FACE.vertices) {
			_vertices[vertex_idx].boundary_faces.push_back(e_idx);
		}

		//move FACE to _boundary
		_boundary.push_back(std::move(FACE));
		_boundary_track.push_back(std::move(bt));
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::insertBoundaryFace_Unlocked(Face_t &FACE, const size_t face_idx, FaceTracker &bt) {
		//The method that calls this must ensure that this is thread-safe.
		//If only one color of elements are being inserted, it will be safe
		if constexpr (HierarchicalMeshElement<Face_t>) {FACE.is_active=true;}

		FACE.index = face_idx;

		//update existing vertices
		for (size_t vertex_idx : FACE.vertices) {
			_vertices[vertex_idx].boundary_faces.push_back(face_idx);
		}

		//move FACE to _boundary
		_boundary[face_idx] = std::move(FACE);
		_boundary_track[face_idx] = std::move(bt);
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::constructElement_Locked(
			const std::vector<typename BasicMesh<Vertex_t,Element_t,Face_t>::Point_t> &vertex_coords,
			const int vtkID) {
		assert(vertex_coords.size()==vtk_n_vertices(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new vertices as needed and aggregate their indices
		prepareNodes_Safe(vertex_coords, vtkID, ELEM.vertices);
		//now that the vertices are initialized, insert the element.
		//the vertices will be updated to link back to the new element.
		insertElement_Locked(ELEM);
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::constructElement_Unlocked(
			const std::vector<typename BasicMesh<Vertex_t,Element_t,Face_t>::Point_t> &vertex_coords,
			const int vtkID,
			const size_t elem_idx) {
		assert(vertex_coords.size()==vtk_n_vertices(vtkID));

		//initialize the new element
		Element_t ELEM = Element_t(vtkID);

		//create new vertices as needed and aggregate their indices
		prepareNodes_Safe(vertex_coords, vtkID, ELEM.vertices);

		//now that the vertices are initialized, insert the element.
		//the vertices will be updated to link back to the new element.
		//this will finalize the logic of coloring and linking the vertices back to this element.
		insertElement_Unlocked(ELEM, elem_idx);
	}



	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::computeConformalBoundary() {

		using Point_t   = typename BasicMesh<Vertex_t,Element_t,Face_t>::Point_t;
		

		//create unordered maps to track the count of each face
		struct CountFace {
			int count = 0;
			size_t elem = (size_t) -1;
			int elem_face = -1;
		};

		std::unordered_map<Face_t, CountFace, ElemHashBitPack> all_faces;
		all_faces.reserve(8*_elements.size()); //guess at the number of unique faces (exact if all elements are voxels or hexes)

		//loop through all elements and add the faces to the map
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(ELEM);

			std::vector<size_t> neighbors;
			getElementNeighbors_Unlocked(e_idx, neighbors);

			for (int i=0; i<vtk_n_faces(ELEM.vtkID); i++) {
				Face_t FACE = vtk_elem->getFace(i);

				//add each face or increment the existing count
				all_faces[FACE].count +=1;
				all_faces[FACE].elem = e_idx;
				all_faces[FACE].elem_face = i;
			}

			delete vtk_elem;
		}

		//process boundary faces
		_boundary.clear();
		_boundary.reserve(all_faces.size()/4);
		for (const auto& [FACE, face_count] : all_faces) {
			if (face_count.count==1) {
				Face_t face(FACE);
				FaceTracker bt {face_count.elem, face_count.elem_face};
				insertBoundaryFace_Locked(face, bt);
			}
		}
		_boundary.shrink_to_fit();
	}


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Vertex_t,Element_t,Face_t>::save_as(const std::string filename, const bool include_details, const bool use_ascii) const {
		if (use_ascii) {
			//open and check file
			std::ofstream file(filename);
			if (not file.is_open()){
				throw std::runtime_error("Couldn't open " + filename);
				return;
			}

			//print topology
			print_topology_ascii_vtk(file, *this);

			//print details
			if (include_details) {print_mesh_details_ascii_vtk(file, *this);}

			file.close();
		} else {
			//open and check file
			std::ofstream file(filename, std::ios::binary);
			if (not file.is_open()){
				throw std::runtime_error("Couldn't open " + filename);
				return;
			}

			//print topology
			print_topology_binary_vtk(file, *this);

			//print details
			if (include_details) {print_mesh_details_binary_vtk(file, *this);}

			file.close();
		}
	}
	


	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	std::ostream& operator<<(std::ostream& os, const BasicMesh<Vertex_t,Element_t,Face_t> &mesh) {
		os << "\n" << std::string(50, '=') << "\n"
		   << "Mesh Summary\n"
		   << std::string(50, '-') << "\n";

		os << std::left;
		os << "C++ types\n" << std::string(50, '-') << "\n";
		os << std::setw(15) << "Element_t " << std::setw(10) << elementTypeName<Element_t>() << "\n"
		   << std::setw(15) << "Face_t    " << std::setw(10) << elementTypeName<Face_t>()    << "\n"
		   << std::setw(15) << "Vertex_t    " << std::setw(10) << nodeTypeName<Vertex_t>()       << "\n"
		   << std::string(50, '-') << "\n";

		os << std::left;
		os << "Feature Counts\n" << std::string(50, '-') << "\n"; 
		os << std::setw(15) << "Elements       " << std::setw(10) << std::right << mesh.nElems()         << "\n"
		   << std::setw(15) << "Boundary_Faces " << std::setw(10) << std::right << mesh.nBoundaryFaces() << "\n"
		   << std::setw(15) << "Nodes          " << std::setw(10) << std::right << mesh.nNodes()         << "\n"
		   << std::string(50, '-') << "\n";


		//get distribution of element types
		std::unordered_map<int,size_t> elem_type_count;
		for (const Element_t &ELEM : mesh) {
			elem_type_count[ELEM.vtkID] += 1;
		}

		os << "FEM Element Types\n" << std::string(50, '-') << "\n";
		for (const auto &pair : elem_type_count) {
			os << std::left << std::setw(15) << vtk_id_to_string(pair.first) << std::right << std::setw(10) << pair.second << "\n";
		}
		os << std::string(50, '-') << "\n";
		return os;
	}



	template<BasicMeshVertex Vertex_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void memorySummary(const BasicMesh<Vertex_t,Element_t,Face_t> &mesh) {
		double verticesUsed     = (double) sizeof(Vertex_t) * (double) mesh._vertices.size();
		double verticesCap      = (double) sizeof(Vertex_t) * (double) mesh._vertices.capacity();
		double verticesElemUsed = 0;
		double verticesElemCap  = 0;
		size_t verticesElemCount = 0;
		for (size_t n=0; n<mesh._vertices.size(); n++) {
			const std::vector<size_t> &vec = mesh._vertices[n].elems;
			verticesElemUsed += (double) sizeof(size_t) * (double) vec.size();
			verticesElemCap  += (double) sizeof(size_t) * (double) vec.capacity();
			verticesElemCount += vec.size();
		}
		double verticesBoundaryUsed = 0;
		double verticesBoundaryCap  = 0;
		size_t verticesBoundaryCount = 0;
		for (size_t n=0; n<mesh._vertices.size(); n++) {
			const std::vector<size_t> &vec = mesh._vertices[n].boundary_faces;
			verticesBoundaryUsed += (double) sizeof(size_t) * (double) vec.size();
			verticesBoundaryCap  += (double) sizeof(size_t) * (double) vec.capacity();
			verticesBoundaryCount += vec.size();
		}

		double total_vertices_used = verticesUsed + verticesElemUsed + verticesBoundaryUsed;
		double total_vertices_cap  = verticesCap  + verticesElemCap  + verticesBoundaryCap;


		//memory for octree structure of _vertices
		size_t nOctreeNodes{0}, nOctreeIdx{0}, nOctreeIdxCap{0}, nLeafs{0};
		int maxDepth{0};
		mesh._vertices.treeSummary(nOctreeNodes, nOctreeIdx, nOctreeIdxCap, nLeafs, maxDepth);

		double octreeNodeMemory      = (double) sizeof(typename std::decay_t<decltype(mesh._vertices)>::Node_t) * nOctreeNodes;
		double octreeIndexMemoryUsed = (double) sizeof(size_t) * nOctreeIdx;
		double octreeIndexMemoryCap  = (double) sizeof(size_t) * nOctreeIdxCap;

		double total_vertices_octree_used = octreeNodeMemory + octreeIndexMemoryUsed;
		double total_vertices_octree_cap  = octreeIndexMemoryCap;


		//memory for _elements
		double elementsUsed     = (double) sizeof(Vertex_t) * (double) mesh._elements.size();
		double elementsCap      = (double) sizeof(Vertex_t) * (double) mesh._elements.capacity();
		double elementsNodesUsed = 0;
		double elementsNodesCap  = 0;
		size_t elementsNodesCount = 0;
		for (size_t n=0; n<mesh._elements.size(); n++) {
			const std::vector<size_t> &vec = mesh._elements[n].vertices;
			elementsNodesUsed += (double) sizeof(size_t) * (double) vec.size();
			elementsNodesCap  += (double) sizeof(size_t) * (double) vec.capacity();
			elementsNodesCount += vec.size();
			assert(vec.size()==vec.capacity());
		}

		[[maybe_unused]] double elementsChildrenUsed = 0;
		[[maybe_unused]] double elementsChildrenCap  = 0;
		[[maybe_unused]] size_t elementsChildrenCount = 0;
		if constexpr (HierarchicalMeshElement<Element_t>) {
			for (size_t n=0; n<mesh._elements.size(); n++) {
				const std::vector<size_t> &vec = mesh._elements[n].children;
				elementsChildrenUsed += (double) sizeof(size_t) * (double) vec.size();
				elementsChildrenCap  += (double) sizeof(size_t) * (double) vec.capacity();
				elementsChildrenCount += vec.size();
			}
		}

		double total_elements_used = elementsUsed + elementsNodesUsed + elementsChildrenUsed;
		double total_elements_cap  = elementsCap  + elementsNodesCap  + elementsChildrenCap;

		//memory for _boundary
		double boundaryUsed     = (double) sizeof(Vertex_t) * (double) mesh._boundary.size();
		double boundaryCap      = (double) sizeof(Vertex_t) * (double) mesh._boundary.capacity();
		double boundaryNodesUsed = 0;
		double boundaryNodesCap  = 0;
		size_t boundaryNodesCount = 0;
		for (size_t n=0; n<mesh._boundary.size(); n++) {
			const std::vector<size_t> &vec = mesh._boundary[n].vertices;
			boundaryNodesUsed += (double) sizeof(size_t) * (double) vec.size();
			boundaryNodesCap  += (double) sizeof(size_t) * (double) vec.capacity();
			boundaryNodesCount += vec.size();
			assert(vec.size()==vec.capacity());
		}

		[[maybe_unused]] double boundaryChildrenUsed = 0;
		[[maybe_unused]] double boundaryChildrenCap  = 0;
		[[maybe_unused]] size_t boundaryChildrenCount = 0;
		if constexpr (HierarchicalMeshElement<Face_t>) {
			for (size_t n=0; n<mesh._boundary.size(); n++) {
				const std::vector<size_t> &vec = mesh._boundary[n].children;
				boundaryChildrenUsed += (double) sizeof(size_t) * (double) vec.size();
				boundaryChildrenCap  += (double) sizeof(size_t) * (double) vec.capacity();
				boundaryChildrenCount += vec.size();
			}
		}

		double total_boundary_used = boundaryUsed + boundaryNodesUsed + boundaryChildrenUsed;
		double total_boundary_cap  = boundaryCap  + boundaryNodesCap  + boundaryChildrenCap;

		//memory for _boundary_track
		double boundaryTrackUsed = (double) sizeof(FaceTracker) * (double) mesh._boundary_track.size();
		double boundaryTrackCap  = (double) sizeof(FaceTracker) * (double) mesh._boundary_track.capacity();

		////////////////////////////////
		//////// assemble table ////////
		////////////////////////////////

		//header
		std::cout << "\n" << std::string(90, '=') << "\n"
				  << std::left << "Mesh Memory Summary\n" 
				  << std::string(90, '-') << "\n";

		std::cout << std::left
				  << std::setw(30) << "Container"
				  << std::right
				  << std::setw(20) << "Count"
				  << std::setw(20) << "Size (MiB)"
				  << std::setw(20) << "Reserved (MiB)"
				  << "\n" 
				  << std::string(90, '-') << "\n";

		//_vertices (data)
		std::cout << std::left << std::setw(30) << "_vertices"
				  << std::right
				  << std::setw(20) << mesh._vertices.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesCap  / 1048576.0
				  << "\n"
				  << std::left << std::setw(34) << "  \u251c\u2500 elems"
				  << std::right
				  << std::setw(20) << verticesElemCount
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesElemUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesElemCap  / 1048576.0
				  << "\n"
				  << std::left << std::setw(34) << "  \u251c\u2500 boundary_faces"
				  << std::right
				  << std::setw(20) << verticesBoundaryCount
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesBoundaryUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << verticesBoundaryCap  / 1048576.0
				  << "\n";

		std::cout << std::left << std::setw(30) << "  \u2514\u2500 octree structure"
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 tree vertices (all)"
				  << std::right
				  << std::setw(20) << nOctreeNodes
				  << std::setw(20) << std::fixed << std::setprecision(3) << octreeNodeMemory / 1048576.0
				  << std::setw(20) << "n/a"
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 tree leafs"
				  << std::right
				  << std::setw(20) << nLeafs 
				  << std::setw(20) << "n/a"
				  << std::setw(20) << "n/a"
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 data index storage"
				  << std::right;

		if (nOctreeIdx!=mesh._vertices.size()) {
			std::cout << std::setw(20) << "(W) "+std::to_string(nOctreeIdx);
		} else {
			std::cout << std::setw(20) << nOctreeIdx;
		}
		std::cout << std::setw(20) << std::fixed << std::setprecision(3) << octreeIndexMemoryUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << octreeIndexMemoryCap  / 1048576.0
				  << "\n"
				  << std::left << "      \u251c\u2500 maximum depth= " << maxDepth << "\n"
				  << std::left << "      \u2514\u2500 bounding box\n"
				  << std::left << "          \u251c\u2500 high= " << mesh._vertices.bbox().high() << "\n"
				  << std::left << "          \u2514\u2500 low= " << mesh._vertices.bbox().low()  << "\n"
				  << std::string(90, '-') << "\n";


		//_elements
		std::cout << std::left << std::setw(30) << "_elements"
				  << std::right
				  << std::setw(20) << mesh._elements.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << elementsUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << elementsCap  / 1048576.0
				  << "\n";
		if constexpr (HierarchicalMeshElement<Element_t>) {
			std::cout << std::left << std::setw(34) << "  \u251c\u2500 vertices";
		} else {
			std::cout << std::left << std::setw(34) << "  \u2514\u2500 vertices";
		}
		std::cout << std::right
				  << std::setw(20) << elementsNodesCount
				  << std::setw(20) << std::fixed << std::setprecision(3) << elementsNodesUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << elementsNodesCap  / 1048576.0
				  << "\n";
		if constexpr (HierarchicalMeshElement<Element_t>) {
			std::cout << std::left << std::setw(34) << "  \u2514\u2500 children"
					  << std::right
					  << std::setw(20) << elementsChildrenCount
					  << std::setw(20) << std::fixed << std::setprecision(3) << elementsChildrenUsed / 1048576.0
					  << std::setw(20) << std::fixed << std::setprecision(3) << elementsChildrenCap  / 1048576.0
					  << "\n";
		}
		std::cout << std::string(90, '-') << "\n";

		//_boundary
		std::cout << std::left << std::setw(30) << "_boundary"
				  << std::right
				  << std::setw(20) << mesh._boundary.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryCap  / 1048576.0
				  << "\n";
		std::cout << std::left << std::setw(34) << "  \u2514\u2500 vertices";
		std::cout << std::right
				  << std::setw(20) << boundaryNodesCount
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryNodesUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryNodesCap  / 1048576.0
				  << "\n";
		if constexpr (HierarchicalMeshElement<Face_t>) {
			std::cout << std::left << std::setw(34) << "  \u2514\u2500 children"
					  << std::right
					  << std::setw(20) << boundaryChildrenCount
					  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryChildrenUsed / 1048576.0
					  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryChildrenCap  / 1048576.0
					  << "\n";
		}
		std::cout << std::string(90, '-') << "\n";

		//_boundary_track
		std::cout << std::left << std::setw(30) << "_boundary_track"
				  << std::right
				  << std::setw(20) << mesh._boundary_track.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryTrackUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << boundaryTrackCap  / 1048576.0
				  << "\n";
		std::cout << std::string(90, '-') << "\n";


		//total
		double totalUsed = total_vertices_used + total_vertices_octree_used + total_elements_used + total_boundary_used + boundaryTrackUsed;
		double totalCap = total_vertices_cap + total_vertices_octree_cap + total_elements_cap + total_boundary_cap + boundaryTrackCap;
		std::cout << std::string(90, '-') << "\n";
		std::cout << std::left << std::setw(30) << "Total"
				  << std::right
				  << std::setw(20) << ""
				  << std::setw(20) << std::fixed << std::setprecision(3) << totalUsed / 1048576.0
				  << std::setw(20) << std::fixed << std::setprecision(3) << totalCap  / 1048576.0
				  << "\n";


		std::cout << std::string(90, '-') << "\n";
		//duplicate check in _vertices
		std::cout << "\n";
		mesh._vertices.duplicateCheck();
		mesh._vertices.findCheck();
	}
}

