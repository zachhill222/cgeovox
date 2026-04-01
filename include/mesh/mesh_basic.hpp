#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_iterator.hpp"
#include "mesh/mesh_vtk_out.hpp"
#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "gutil.hpp"

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
	/// @tparam space_dim        The dimension of the space that the mesh is embedded in. Usually 3. (Untested for 2).
	/// @tparam REF_DIM          The dimension of the space that the reference elements are embedded in. Usually 3. (2 for surface meshes).
	/// @tparam Scalar_t         The scalar type to emulate the real line. This should be robust under comparisions and arithmetic ordering. (e.g., FixedPrecision instead of float)
	/// @tparam Element_type     The type of element to use. This is usually set by the class that inherits from this class.
	/////////////////////////////////////////////////
	template<
			BasicMeshElement Element_type = BasicElement<VOXEL_VTK_ID>,
			Scalar           Scalar_type  = gutil::FixedPoint64<>
			>
	class BasicMesh	{
		/// Make the MeshView class a friend
		// template<int space_dim_u, int REF_DIM_u, Scalar Scalar_u, BasicMeshElement Element_u>
		// friend class MeshView;
	public:
		static constexpr int SPACE_DIM          = 3;
		static constexpr int REF_DIM            = vtk_ref_dim(Element_type::VTK_ID);
		static constexpr int ELEM_VTK_ID        = Element_type::VTK_ID;
		static constexpr int FACE_VTK_ID        = vtk_face_id(Element_type::VTK_ID);
		static constexpr int N_VERT_PER_ELEMENT = Element_type::N_VERTS;
		static constexpr int N_FACE_PER_ELEMENT = vtk_n_faces(Element_type::VTK_ID);		


		//set up element/face/vertex data types
		using Element_t  = Element_type;
		using HalfEdge_t = HalfEdge<N_FACE_PER_ELEMENT>;
		using Vertex_t   = BasicVertex<gutil::Point<SPACE_DIM,Scalar_type>>;
		
		//set up coordinate types
		using Scalar_t   = Scalar_type;
		using GeoPoint_t = gutil::Point<SPACE_DIM,Scalar_type>; //data type for computing spatial coordinates
		using RefPoint_t = gutil::Point<REF_DIM,double>; //data type for evaluating basis functions, computing jacobians, etc.

		//utility types
		using Mesh_t             = BasicMesh<Element_type,Scalar_type>; //type of this mesh
		using Index_t            = gutil::Point<REF_DIM,size_t>; //index for creating structured mesh in the constructor
		using GeoBox_t           = gutil::Box<SPACE_DIM, Scalar_type>; //boxes in the domain space
		using RefBox_t           = gutil::Box<REF_DIM, double>; //boxes in the reference space

		using VertexList_t       = VertexOctree<Vertex_t, 64, Scalar_type>;
		using ElementLogic_t     = VtkElementType_t<Mesh_t, Element_type::VTK_ID>; //type to handle logic of creating children, getting faces, etc.
		
		
	protected:
		/////////////////////////////////////////////////
		/// Container for the elements.
		/////////////////////////////////////////////////
		std::vector<Element_t> _elements;
		
		/////////////////////////////////////////////////
		/// Container for all faces
		/////////////////////////////////////////////////
		std::vector<HalfEdge_t> _halfedges;

		/////////////////////////////////////////////////
		/// Container to denote boundary vertices so that boundary elements/faces can be easily recovered.
		/////////////////////////////////////////////////
		std::vector<size_t> _boundary_vertices;

		/////////////////////////////////////////////////
		/// Container for the mesh vertices. This has an octree structure for quickly finding vertices from their location in space.
		/// This method has parallel read (_vertices.find) and serial write (_vertices.push_back) built in.
		/// If data can be guarenteed to not be duplicated, this also supports parallel write via _vertices.push_back_async.
		/////////////////////////////////////////////////
		VertexList_t _vertices;

	private:
		mutable std::shared_mutex _rw_mtx;

	public:
		BasicMesh() : _elements{}, _halfedges{}, _boundary_vertices{}, _vertices{} {}
		
		BasicMesh(const GeoBox_t &domain) : _elements{}, _halfedges{}, _boundary_vertices{}, _vertices(1.125*domain) {}

		BasicMesh(const RefBox_t &domain) requires(REF_DIM==2) : _elements{}, _halfedges{}, _boundary_vertices{}
		{
			Point_t low, high;
			int i;
			for (i=0; i<REF_DIM; i++) {
				low[i]  = domain.low()[i];
				high[i] = domain.high()[i];
			}

			for (i; i<SPACE_DIM; i++) {
				low[i]  = -1.0;
				high[i] =  1.0;
			}

			_vertices.set_bbox(1.125*GeoBox_t{low,high});
		}

		BasicMesh(const RefBox_t &domain, const Index_t &N)
			requires (VTK_ID==VOXEL_VTK_ID or VTK_ID==HEXAHEDRON_VTK_ID or VTK_ID==PIXEL_VTK_ID or VTK_ID==QUAD_VTK_ID)
		 	: BasicMesh(domain)
		{
			if constexpr (REF_DIM==3) {build_voxel_mesh(domain, N, useIsopar);}
			else if constexpr (REF_DIM==2) {build_pixel_mesh(domain, N, useIsopar);}
			else {throw std::runtime_error("BasicMesh: can't mesh domain");}
		}
		
		virtual ~BasicMesh() {}

		/////////////////////////////////////////////////
		/// Read elements and vertices externally
		/////////////////////////////////////////////////
		const Vertex_t&     get_vertex(const size_t idx)          const {assert(idx<_vertices.size()); return _vertices[idx];}
		const Element_t&    get_element(const size_t idx)         const {assert(idx<_elements.size()); return _elements[idx];}
		const HalfEdge_t&   get_halfedge(const size_t idx)        const {assert(idx<_halfedges.size()); return _halfedges[idx];}
		const Vertex_t&     get_boundary_vertex(const size_t idx) const {assert(idx<_boundary_vertices.size()); return _vertices[_boundary_vertices[idx]];}

		//////////////////////////////////////////////////
		/// Access containers externally
		//////////////////////////////////////////////////
		const VertexList_t&            get_vertices()          const {return _vertices;}
		const std::vector<Element_t>&  get_elements()          const {return _elements;}
		const std::vector<HalfEdge_t>& get_halfedges()         const {return _halfedges;}
		const std::vector<size_t>&     get_boundary_vertices() const {return _boundary_vertices;}

		//////////////////////////////////////////////////
		/// Utility methods
		//////////////////////////////////////////////////
		const size_t   closest_vertex(const GeoPoint_t& point) const {return _vertices.find_closest(point);}
		const GeoBox_t bbox()                                  const {return _vertices.bbox_tight();}

		size_t nElements() const {return _elements.size();}
		size_t nVertices() const {return _vertices.size();}
		
		/////////////////////////////////////////////////
		/// Allocate space
		/////////////////////////////////////////////////
		void element_reserve(const size_t n_elements) {
			_elements.reserve(n_elements);
			_halfedges.reserve(N_FACE_PER_ELEMENT*n_elements);
		}
		
		void element_resize(const size_t n_elements) {
			_elements.resize(n_elements);
			_halfedges.resize(N_FACE_PER_ELEMENT*n_elements);
		}

		void vertex_reserve(const size_t n_vertices) {
			_vertices.reserve(n_vertices);
		}

		void vertex_resize(const size_t n_vertices) {
			_vertices.resize(n_vertices);
		}

		void clear() {
			_elements.clear();
			_halfedges.clear();
			_vertices.clear();
		}

		/////////////////////////////////////////////////
		/// Mesh a 3D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void build_voxel_mesh(const GeoBox_t &domain, const Index_t& N, const bool useIsopar=false)
			requires (REF_DIM==3 and (VTK_ID==VOXEL_VTK_ID or VTK_ID==HEXAHEDRON_VTK_ID));


		/////////////////////////////////////////////////
		/// Mesh a 2D box using voxels of equal size. This can be used for testing or creating a simple initial mesh.
		///
		/// @param domain The domain to be meshed
		/// @param N The number of elements along each coordinate axis
		/////////////////////////////////////////////////
		void build_pixel_mesh(const GeoBox_t &domain, const Index_t& N, const bool useIsopar=false)
			requires (REF_DIM==2 and (VTK_ID==PIXEL_VTK_ID or VTK_ID==QUAD_VTK_ID));


		/////////////////////////////////////////////////
		/// A method to get the active elements that share a node with the specified element. This allows for the mesh to be refined and coarsened without changing the data structures.
		///
		/// This is a virtual method as it must be changed for a hierarchical mesh.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param neighbors A reference to an existing vector where the result will be stored (via neighbors.push_back()).
		/////////////////////////////////////////////////
		void get_element_neighbors(const size_t elem_idx, std::vector<size_t>& neighbors) const;

		
		/////////////////////////////////////////////////
		/// Get vertices at the specified coordinates. If there is no vertex already there, one is created.
		///
		/// @tparam N The number of vertices
		/// @tparam ASYNC When ASYNC is false, the vertices are found/inserted in the presented order and duplicates are ok.
		///               When ASYNC is true, the coordinates MUST be unique in all instances where _vertices is being altered.
		///               The caller should probably call _vertices.flush() to ensure that the structure is finished building as well.
		/// @param coords The coordinates of the vertices. Note that these resources will be moved and cannot be used after being sent to this method.
		///               If they are needed, use _vertices[result[i]].coord to get the coordinate that was at coords[i].
		/////////////////////////////////////////////////
		template<int N, bool ASYNC=false>
		std::array<size_t,N> prepare_vertices(std::array<GeoPoint_t,N>&& coords);


		/////////////////////////////////////////////////
		/// A method to construct a new element by its vertex coordinates. Vertex indices for the new element are
		/// obtained via prepare_vertices<N_VERT_PER_ELEMENT,ASYNC>.
		///
		/// @tparam ASYNC A flag to pass to prepare_vertices that determines if new vertices are constructed asynchronously.
		/// @param coords A reference to an existing vector of vertex coordinates.
		/////////////////////////////////////////////////
		template<bool ASYNC=false>
		Element_t construct_element(std::array<GeoPoint_t,N_VERT_PER_ELEMENT>&& coords);


		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The index where the element was moved to is returned.
		/// This method will populate the connectivity of the mesh, but the vertex indices of the element must be correctly set.
		///
		/// This method is marked as virtual so classes that inherit can populate special fields (e.g., color).
		/// 
		/// @tparam ASYNC A flag that should be set to true if multiple threads are inserting new elements.
		///               If ASYNC is true, then the index where the element is to be stored must be passed.
		///
		/// @param element The element to be inserted. The vertices must already be populated,
		///                but the other connectivities will be constructed from this element.
		///
		/// @param index If known, the index where the element is to be inserted can be passed. This must be done
		///              whenever ASYNC is true. If the size of index is larger than the size of _elements, then
		///              the new element is appended to the end of _elements via push_back(std::move(element)).
		/////////////////////////////////////////////////
		template<bool ASYNC=false>
		virtual size_t insert_element(Element_t&& element, size_t index = (size_t) -1);

		
		/////////////////////////////////////////////////
		/// Compute the halfedges of the (specified) elements.
		/////////////////////////////////////////////////
		void pair_halfedges();

		template<typename Container_type>
		void pair_halfedges_for_elements(const Container_type& element_indices)

		/////////////////////////////////////////////////
		/// Method to move a vertex in the mesh. If any of the elements associated with this node
		/// are not isoparametric, they are converted to their isoparametric versions.
		///
		/// @param vertex_idx  The index of the node that will be moved
		/// @param new_coord   The coordinate where the node will be moved to
		/////////////////////////////////////////////////
		// void moveVertex(const size_t vertex_idx, GeoPoint_t new_coord) {
		// 	Vertex_t VERTEX = _vertices[vertex_idx]; //make a new copy

		// 	for (size_t e_idx : VERTEX.elems) {makeIsoparametric(_elements[e_idx]);}
		// 	for (size_t f_idx : VERTEX.boundary_faces) {makeIsoparametric(_boundary[f_idx]);}
			
		// 	VERTEX.coord = new_coord;
		// 	_vertices.replace(std::move(VERTEX), vertex_idx);
		// }
		

		/////////////////////////////////////////////////
		/// Save the mesh to a file.
		///
		/// @param filename        The name of the file (including the path and extension) of the file to which the mesh will be written.
		/// @param include_details When set to true, the mesh details will be appended to the mesh topology.
		///                        This should usually be set to false if any additional data will be appended to the file.
		/////////////////////////////////////////////////
		void save_as(const std::string filename, const bool include_details=false, const bool use_ascii=false) const;

		/////////////////////////////////////////////////
		/// Iterators for _elements. These are the most common iterators and get the defualt begin() and end() methods.
		/////////////////////////////////////////////////
		auto element_begin()        {return _elements.begin();}
		auto element_end()          {return _elements.end();}
		auto element_begin()  const {return _elements.cbegin();}
		auto element_end()    const {return _elements.cend();}
		auto element_cbegin() const {return _elements.cbegin();}
		auto element_cend()   const {return _elements.cend();}

	
		/////////////////////////////////////////////////
		/// Iterators for _faces
		/////////////////////////////////////////////////
		auto face_begin()        {return _faces.begin();}
		auto face_end()          {return _faces.end();}
		auto face_begin()  const {return _faces.cbegin();}
		auto face_end()    const {return _faces.cend();}
		auto face_cbegin() const {return _faces.cbegin();}
		auto face_cend()   const {return _faces.cend();}


		/////////////////////////////////////////////////
		/// Iterators for _vertices
		/////////////////////////////////////////////////
		auto vertex_begin()        {return _vertices.begin();}
		auto vertex_end()          {return _vertices.end();}
		auto vertex_begin()  const {return _vertices.cbegin();}
		auto vertex_end()    const {return _vertices.cend();}
		auto vertex_cbegin() const {return _vertices.cbegin();}
		auto vertex_cend()   const {return _vertices.cend();}
	};


	template<BasicMeshElement Element_type, Scalar Scalar_type>
	void BasicMesh<Element_type, Scalar_type>::build_voxel_mesh (
			const GeoBox_t &domain,
			const Index_t &N)
			requires (REF_DIM==3 and (VTK_ID==VOXEL_VTK_ID or VTK_ID==HEXAHEDRON_VTK_ID))
		{

		//reserve space
		clear();
		vertices_resize((N[0]+1) * (N[1]+1) * (N[2]+1));
		element_resize(N[0]*N[1]*N[2]);
		
		//initialize the vertices
		const GeoPoint_t H = domain.sidelength() / GeoPoint_t(N);
		#pragma omp parallel for collapse(3)
		for (size_t i=0; i<=N[0]; i++) {
			for (size_t j=0; j<=N[1]; j++) {
				for (size_t k=0; k<=N[2]; k++) {
					GeoPoint_t vertex  = domain.low() + GeoPoint_t{i,j,k} * H;
					Vertex_t VERTEX(std::move(vertex));
					size_t idx = _vertices.push_back_async(std::move(VERTEX));
					_vertices[idx].index = idx;
				}
			}
		}
		_vertices.flush();


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
								const GeoPoint_t low  = domain.low() + GeoPoint_t{i,j,k} * H;
								const GeoPoint_t high = domain.low() + GeoPoint_t{i+1,j+1,k+1} * H;
								const GeoBox_t elem_box{low, high};
							
								//assemble the list of vertices
								std::array<GeoPoint_t, N_VERT_PER_ELEMENT> element_vertices;
								for (int l=0; l<N_VERT_PER_ELEMENT; l++) {
									if constexpr (VTK_ID == VOXEL_VTK_ID) {
										element_vertices[l] = elem.voxelvertex(l);
									}
									else if constexpr (VTK_ID == HEXAHEDRON_VTK_ID) {
										element_vertices[l] = elem.hexvertex(l);
									}
								}

								//put the element into the mesh
								const size_t idx = i + N[0]*(j + k*N[1]);
								Element_t ELEM(std::move(element_vertices));
								[[maybe_unused]] const size_t index = insert_element<true>(std::move(ELEM), idx);
								assert(index==idx);
								assert(index==_elements[index].index);
							}
						}
					}
				}
			}
		}

		//populate the connectivity of the half-edges
		pair_halfedges();
	}


	template<BasicMeshElement Element_type, Scalar Scalar_type>
	void BasicMesh<Element_type, Scalar_type>::setPixelMesh_Locked(
			const GeoBox_t &domain,
			const Index_t &N,
			const bool useIsopar) 
		requires (REF_DIM==2 and (VTK_ID==PIXEL_VTK_ID or VTK_ID==QUAD_VTK_ID))
	{
		//reserve space
		clear();

		vertices_reserve((N[0]+1) * (N[1]+1));
		elements_reserve(N[0]*N[1]);
		
		//construct the mesh
		const GeoPoint_t H(domain.sidelength() / GeoPoint_t(N));
		for (size_t i=0; i<N[0]; i++) {
			for (size_t j=0; j<N[1]; j++) {
				//define element extents
				GeoPoint_t low  = domain.low() + GeoPoint_t{i,j} * H;
				GeoPoint_t high = domain.low() + GeoPoint_t{i+1,j+1} * H;
				GeoBox_t   elem_box {low, high};
			
				//assemble the list of vertices
				std::array<GeoPoint_t, N_VERT_PER_ELEMENT> element_vertices;
				for (size_t l=0; l<N_VERT_PER_ELEMENT; l++) {
					if constexpr (ELEM_VTK_ID == QUAD_VTK_ID) {
						element_vertices[l] = static_cast<GeoPoint_t>(elem_box.hexvertex(l));
					}
					else if constexpr (ELEM_VTK_ID == PIXEL_VTK_ID) {
						element_vertices[l] = static_cast<GeoPoint_t>(elem_box.voxelvertex(l));
					}
				}

				//put the element into the mesh
				Element_t ELEM = construct_element(std::move(element_vertices));
				insert_element(std::move(ELEM));
			}
		}

		//finalize the faces
		pair_halfedges();
	}


	
	template<BasicMeshElement Element_type, Scalar Scalar_type>
	void BasicMesh<Element_type, Scalar_type>::get_element_neighbors(const size_t elem_idx, std::vector<size_t>& neighbors) const {
		const Element_t &ELEM = _elements[elem_idx];

		//use unordered set to ensure unique vertices
		std::unordered_set<size_t> neighbor_set;

		//loop through the vertices of the current element
		for (size_t n_idx : ELEM.vertices) {
			const Vertex_t& VERTEX = _vertices[n_idx];

			//loop through the elements of the current node
			for (size_t e_idx : VERTEX.elems) {
				if (e_idx!=elem_idx) {
					neighbor_set.insert(e_idx);
				}
			}
		}

		//convert the set to the vector
		neighbors.insert(neighbors.end(), neighbor_set.begin(), neighbor_set.end());
	}

	template<BasicMeshElement Element_type, Scalar Scalar_type>
	template<int N, bool ASYNC>
	std::array<size_t,N> BasicMesh<Element_type, Scalar_type>::prepare_vertices(std::array<GeoPoint_t,N>&& coords)
	{
		std::array<size_t,N> indices;

		//create new vertices as needed and aggregate their indices
		for (int i=0; i<N; ++i) {
			Vertex_t VERTEX(std::move(coords[i]));
			if constexpr (ASYNC) {
				indices[i] = _vertices.push_back_async(std::move(VERTEX));
			}
			else {
				indices[i] = _vertices.push_back(std::move(VERTEX));
			}
		}

		return indices;
	}

	template<BasicMeshElement Element_type, Scalar Scalar_type>
	template<bool ASYNC>
	Element_t BasicMesh<Element_type, Scalar_type>::construct_element(std::array<GeoPoint_t,N_VERT_PER_ELEMENT>&& coords)
	{
		//get indices of the vertices at the specified coordinates
		//create new vertices if necessary
		std::array<size_t, N_VERT_PER_ELEMENT> indices = prepare_vertices<ASYNC>(std::move(coords));

		//return the new element
		return Element_t{std::move(indices);}
	}
	

	template<BasicMeshElement Element_type, Scalar Scalar_type>
	template<bool ASYNC>
	size_t BasicMesh<Element_type, Scalar_type>::insert_element(Element_t&& element, size_t index)
	{
		if constexpr (ASYNC) {
			assert(index < _elements.size();)
			_elements[index] = std::move(element);
		}
		else {
			index = _elements.size();
			_elements.push_back(std::move(element));
		}


		//populate vertex connectivity
		for (size_t v_idx : _elements[index].vertices) {
			_vertices[v_idx].elems.push_back(index);
		}

		return index;
	}

	template<BasicMeshElement Element_type, Scalar Scalar_type>
	template<typename Container_type>
	void BasicMesh<Element_type, Scalar_type>::pair_halfedges_for_elements(const Container_type& element_indices)
	{
		ElementLogic_t THIS_ELEM_LOGIC, NEIGHBOR_ELEM_LOGIC;
		for (size_t e_idx : element_indices) {
			const Element_t& ELEM = _elements[e_idx];
			THIS_ELEM_LOGIC.set_element(*this, e_idx);

			//loop through the neighbors
			neighbors.clear();
			get_element_neighbors(e_idx, neighbors);

			//loop through ELEM local faces
			for (int f_idx=0; f_idx<N_FACE_PER_ELEMENT; ++f_idx) {
				auto face_indices = THIS_ELEM_LOGIC.get_face_vertices(f_idx);
				std::sort(face_indices.begin(), face_indices.end());

				//loop through neighbors to find the opposite face
				bool found = false;
				for (size_t n_idx : neighbors) {
					if constexpr (Element_t::HIERARCHICAL) {
						if (_elements[n_idx].depth != ELEM.depth) {continue;}
					}

					NEIGHBOR_ELEM_LOGIC.set_element(*this, n_idx);

					//loop through neighbor local faces
					for (int nf_idx=0; nf_idx<N_FACE_PER_ELEMENT; ++nf_idx) {
						auto neighbor_face_indices = NEIGHBOR_ELEM_LOGIC.get_face_vertices(nf_idx);
						std::sort(neighbor_face_indices.begin(), neighbor_face_indices.end());

						if (face_indices == neighbor_face_indices) {
							found = true;
							const size_t halfedge_idx = N_FACE_PER_ELEMENT*e_idx + static_cast<size_t>(f_idx);
							const size_t opposite_idx = N_FACE_PER_ELEMENT*n_idx + static_cast<size_t>(nf_idx);
							_halfedges[halfedge_idx].opposite = opposite_idx;
							_halfedges[opposite_idx].opposite = halfedge_idx; //race condition if not called in parallel and not carefully
							break;
						}
					}

					if (found) {break;}
				}
			}
		}
	}
	

	template<BasicMeshElement Element_type, Scalar Scalar_type>
	void BasicMesh<Element_type, Scalar_type>::pair_halfedges()
	{
		assert(_elements.size()*N_FACE_PER_ELEMENT == _halfedges.size());

		//for each element, attempt to pair each local face with another element
		#pragma omp parallel
		{
			ElementLogic_t THIS_ELEM_LOGIC;
			ElementLogic_t NEIGHBOR_ELEM_LOGIC;
			std::vector<size_t> neighbors;
			#pragma omp for
			for (size_t e_idx=0; e_idx<_elements.size(); ++e_idx) {
				const Element_t& ELEM = _elements[e_idx];
				THIS_ELEM_LOGIC.set_element(*this, e_idx);

				//loop through the neighbors
				neighbors.clear();
				get_element_neighbors(e_idx, neighbors);

				//loop through ELEM local faces
				for (int f_idx=0; f_idx<N_FACE_PER_ELEMENT; ++f_idx) {
					auto face_indices = THIS_ELEM_LOGIC.get_face_vertices(f_idx);
					std::sort(face_indices.begin(), face_indices.end());

					//loop through neighbors to find the opposite face
					bool found = false;
					for (size_t n_idx : neighbors) {
						NEIGHBOR_ELEM_LOGIC.set_element(*this, n_idx);

						//loop through neighbor local faces
						for (int nf_idx=0; nf_idx<N_FACE_PER_ELEMENT; ++nf_idx) {
							auto neighbor_face_indices = NEIGHBOR_ELEM_LOGIC.get_face_vertices(nf_idx);
							std::sort(neighbor_face_indices.begin(), neighbor_face_indices.end());

							if (face_indices == neighbor_face_indices) {
								found = true;
								const size_t halfedge_idx = N_FACE_PER_ELEMENT*e_idx + static_cast<size_t>(f_idx);
								const size_t opposite_idx = N_FACE_PER_ELEMENT*n_idx + static_cast<size_t>(nf_idx);
								_halfedges[halfedge_idx].opposite = opposite_idx;
								// _halfedges[opposite_idx].opposite = halfedge_idx; //race condition if we set this
								break;
							}
						}

						if (found) {break;}
					}
				}
			}	
		}
	}


	template<BasicMeshElement Element_type, Scalar Scalar_type>
	void BasicMesh<Element_type, Scalar_type>::save_as(const std::string filename, const bool include_details, const bool use_ascii) const {
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
	


	template<BasicMeshType Mesh_t>
	std::ostream& operator<<(std::ostream& os, const Mesh_t &mesh) {
		using Vertex_t  = typename Mesh_t::Vertex_t;
		using Element_t = typename Mesh_t::Element_t;

		os << "\n" << std::string(50, '=') << "\n"
		   << "Mesh Summary\n"
		   << std::string(50, '-') << "\n";

		os << std::left;
		os << "C++ types\n" << std::string(50, '-') << "\n";
		os << std::setw(20) << "ElementStruct_t " << std::setw(15) << elementTypeName<Element_t>() << "\n"
		   << std::setw(20) << "Vertex_t    " << std::setw(15) << vertexTypeName<Vertex_t>()       << "\n"
		   << std::string(50, '-') << "\n";

		os << std::left;
		os << "Feature Counts\n" << std::string(50, '-') << "\n"; 
		os << std::setw(15) << "Elements       " << std::setw(10) << std::right << mesh.nElements()         << "\n"
		   << std::setw(15) << "Boundary_Faces " << std::setw(10) << std::right << mesh.nBoundaryFaces() << "\n"
		   << std::setw(15) << "Nodes          " << std::setw(10) << std::right << mesh.nVertices()         << "\n"
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



	template<BasicMeshType Mesh_t>
	void memorySummary(const Mesh_t &mesh) {
		using Element_t  = typename Mesh_t::Element_t;
		using HalfEdge_t = typename Mesh_t::HalfEdge_t;
		using Vertex_t   = typename Mesh_t::Vertex_t;

		const auto vertices  = mesh.get_vertices();
		const auto elements  = mesh.get_elements();
		const auto halfedges = mesh.get_halfedges();

		constexpr double MiB = 1048576.0;

		//memory for vertices
		auto treeStats = vertices.get_tree_stats();
		double vertices_used = (double) sizeof(Vertex_t) * (double) vertices.size();
		double vertices_reserved = (double) sizeof(Vertex_t) * (double) vertices.capacity();

		double total_vertices_used = treeStats.memory_used_bytes + vertices_used;
		double total_vertices_reserved = treeStats.memory_reserved_bytes + vertices_reserved;

		//memory for elements
		double elements_used = (double) sizeof(Element_t) * (double) elements.size();
		double elements_reserved = (double) sizeof(Element_t) * (double) elements.capacity();
		
		//memory for halfedges
		double halfedge_used = (double) sizeof(HalfEdge_t) * (double) halfedges.size();
		double halfedge_reserved = (double) sizeof(HalfEdge_t) * (double) halfedges.capacity();

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
		std::cout << std::left << std::setw(30) << "vertices"
				  << std::right
				  << std::setw(20) << vertices.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << vertices_used / MiB;
				  << std::setw(20) << std::fixed << std::setprecision(3) << vertices_reserved / MiB;
				  << "\n";

		std::cout << std::left << std::setw(30) << "  \u2514\u2500 octree structure"
				  << std::setw(24) << ""
				  << std::right
				  << std::setw(20) << std::fixed << std::setprecision(3) << treeStats.memory_used_bytes / MiB
				  << std::setw(20) << std::fixed << std::setprecision(3) << treeStats.memory_reserved_bytes  / MiB
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 tree vertices (all)"
				  << std::right
				  << std::setw(20) << treeStats.n_nodes
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 tree leafs"
				  << std::right
				  << std::setw(20) << treeStats.n_leafs
				  << "\n"
				  << std::left << std::setw(34) << "      \u251c\u2500 data index storage"
				  << std::right;

		if (treeStats.n_used_indices!=mesh._vertices.size()) {
			std::cout << std::setw(20) << "(W) "+std::to_string(treeStats.n_used_indices);
		} else {
			std::cout << std::setw(20) << treeStats.n_used_indices;
		}
		std::cout << "\n"
				  << std::left << "      \u251c\u2500 maximum depth= " << treeStats.max_depth << "\n"
				  << std::left << "      \u2514\u2500 bounding box\n"
				  << std::left << "          \u251c\u2500 high= " << mesh._vertices.bbox().high() << "\n"
				  << std::left << "          \u2514\u2500 low= " << mesh._vertices.bbox().low()  << "\n"
				  << std::string(90, '-') << "\n";


		//_elements
		std::cout << std::left << std::setw(30) << "elements"
				  << std::right
				  << std::setw(20) << elements.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << elements_used / MiB
				  << std::setw(20) << std::fixed << std::setprecision(3) << elements_reserved / MiB
				  << "\n";

		//_halfedges
		std::cout << std::left << std::setw(30) << "halfedges"
				  << std::right
				  << std::setw(20) << halfedges.size()
				  << std::setw(20) << std::fixed << std::setprecision(3) << halfedge_used / MiB
				  << std::setw(20) << std::fixed << std::setprecision(3) << halfedge_reserved / MiB
				  << "\n";
		std::cout << std::string(90, '-') << "\n";


		//total
		double total_used = vertices_used + treeStats.memory_used_bytes + elements_used + halfedge_used;
		double total_reserved = vertices_used + treeStats.memory_reserved_bytes + elements_reserved + halfedge_reserved;
		std::cout << std::string(90, '-') << "\n";
		std::cout << std::left << std::setw(30) << "Total"
				  << std::right
				  << std::setw(20) << ""
				  << std::setw(20) << std::fixed << std::setprecision(3) << total_used / MiB
				  << std::setw(20) << std::fixed << std::setprecision(3) << total_reserved / MiB
				  << "\n";


		std::cout << std::string(90, '-') << "\n";
}

}