#pragma once

#include "gutil.hpp"
#include "mesh/vtk_defs.hpp"

#include <string>
#include <concepts>
#include <variant>
#include <vector>
#include <iostream>

namespace gv::mesh
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT STRUCT DEFINITIONS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	/////////////////////////////////////////////////
	/// Struct for tracking basic element information
	/////////////////////////////////////////////////
	template<int VTK_ID_, bool COLORABLE_, bool HIERARCHICAL_>
	struct MeshElement
	{
		//save parameters
		static constexpr int N_VERTS       = vtk_n_vertices(VTK_ID_);
		static constexpr int N_CHILDREN    = HIERARCHICAL_ ? vtk_n_children(VTK_ID_) : 0;
		static constexpr bool COLORABLE    = COLORABLE_;
		static constexpr bool HIERARCHICAL = HIERARCHICAL_;
		static constexpr int  VTK_ID       = VTK_ID_;

		//features that are always present
		std::array<size_t, N_VERTS> vertices;

		//colorable features
		[[no_unique_address]] std::conditional_t<COLORABLE,size_t,std::monostate> color;

		//hierarchical features
		[[no_unique_address]] std::conditional_t<HIERARCHICAL,size_t,std::monostate> parent;
		[[no_unique_address]] std::conditional_t<HIERARCHICAL,size_t,std::monostate> depth;
		[[no_unique_address]] std::conditional_t<HIERARCHICAL,bool,std::monostate> active;
		[[no_unique_address]] std::conditional_t<HIERARCHICAL,std::array<size_t,N_CHILDREN>,std::monostate> children;

		//constructors
		constexpr MeshElement() : vertices{} {
			if constexpr (COLORABLE) {color = (size_t) -1;}
			if constexpr (HIERARCHICAL) {
				parent   = (size_t) -1;
				depth    = 0;
				active   = false;
				children.fill((size_t) -1);
			}
		}

		MeshElement(std::array<size_t,N_VERTS>&& vertices_) : vertices{std::move(vertices_)} {
			if constexpr (COLORABLE) {color = (size_t) -1;}
			if constexpr (HIERARCHICAL) {
				parent   = (size_t) -1;
				depth    = 0;
				active   = false;
				children.fill((size_t) -1);
			}
		}
	};

	//standard types of mesh elements
	template<int VTK_ID>
	using BasicElement = MeshElement<VTK_ID, false, false>;

	template<int VTK_ID>
	using ColoredElement = MeshElement<VTK_ID, true, false>;

	template<int VTK_ID>
	using HierarchicalElement = MeshElement<VTK_ID, false, true>;

	template<int VTK_ID>
	using HierarchicalColoredElement = MeshElement<VTK_ID, true, true>;


	//element concepts
	template<typename T>
	concept BasicMeshElement = requires(T elem) {
		{ T::VTK_ID        } -> std::convertible_to<int>;
		{ T::N_VERTS       } -> std::convertible_to<int>;
		{ T::COLORABLE     } -> std::convertible_to<bool>;
		{ T::HIERARCHICAL  } -> std::convertible_to<bool>;
		{ elem.vertices[0] } -> std::convertible_to<size_t>;
	};

	template<typename T>
	concept ColorableMeshElement = BasicMeshElement<T> and T::COLORABLE;

	template<typename T>
	concept HierarchicalMeshElement = BasicMeshElement<T> and T::HIERARCHICAL;

	template<typename T>
	concept HierarchicalColorableMeshElement = BasicMeshElement<T> and T::COLORABLE and T::HIERARCHICAL;

	//struct for a Half-Edge data structure
	//half-edges represent a face of an element with outward unit normal. Interior faces have an "opposite" half-edge.
	//If the elements of the mesh have N faces each, then the half-edges for one element are stored in N contiguous addresses.
	template<int N_FACES>
	struct HalfEdge
	{
		size_t opposite = (size_t) -1; //index of the face opposite of this (i.e., opposite orientation)

		//index is the index of this half-edge in the vector/array
		static constexpr size_t element(const size_t index)    {return index / N_FACES;}
		static constexpr size_t local_face(const size_t index) {return index % N_FACES;}
		bool on_boundary() const {return opposite == (size_t) -1;}
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT HELPER FUNCTIONS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Check if two elements or faces are the same (up to orientation)
	template<BasicMeshElement Left_t, BasicMeshElement Right_t>
	bool operator==(const Left_t &A, const Right_t &B) {
		if constexpr (Left_t::VTK_ID != Right_t::VTK_ID) {return false;}
		else {
			auto a = A.vertices;
			auto b = B.vertices;
			std::sort(a.begin(), a.end());
			std::sort(b.begin(), b.end());

			return a == b;
		}
	};

	


	/// Element hashing function for use in unordered_set (for example). The order of the element vertices is irrelevent to the hash value.
	template<BasicMeshElement Element_t>
	struct ElemHashBitPack {
		size_t operator()(const Element_t& ELEM) const {
			//sort the vertices
			auto vertices = ELEM.vertices;
			std::sort(vertices.begin(), vertices.end());

			//initialize the hash by getting the last few bits from each node index
			size_t hash = 0;
			constexpr size_t bits_per_node = (sizeof(size_t)*8) / Element_t::N_VERTS;
			constexpr size_t mask = (((size_t) 1) << bits_per_node) - 1; //exactly the last bits_per_node bits are 1

			for (size_t i=0; i<vertices.size(); i++) {
				size_t node_bits = vertices[i] & mask;
				hash |= (node_bits << (i*bits_per_node));
			}

			//scramble the hash (MurmurHash3)
			if constexpr (sizeof(size_t)==8) {
				hash ^= hash >> 33;
				hash *= 0xff51afd7ed558ccdULL;
				hash ^= hash >> 33;
				hash *= 0xc4ceb9fe1a85ec53ULL;
				hash ^= hash >> 33;
			}
			else {
				hash ^= hash >> 16;
				hash *= 0x85ebca6b;
				hash ^= hash >> 16;
				hash *= 0xc2b2ae35;
				hash ^= hash >> 16;
			} 

			return hash;
		}
	};


	/// Element printing
	template<BasicMeshElement Element_t>
	std::ostream& operator<<(std::ostream& os, const Element_t &elem) {
		os << "vtkID= " << elem.VTK_ID << " (" << vtk_id_to_string(elem.VTK_ID) << ")\n";
		os << "index= " << elem.index << "\n";
		os << "vertices (" << elem.N_VERTS << ") : [ ";
		for (size_t n : elem.vertices) {os << n << " ";}
		os << "]\n";

		if constexpr (Element_t::COLORABLE) {
			os << "color= " << elem.color << "\n";
		}

		if constexpr (Element_t::HIERARCHICAL) {
			os << "active= " << elem.active << "\n";
			os << "depth=  " << elem.depth  << "\n";
			os << "parent= " << elem.parent << "\n";
			os << "children ("  << Element_t::N_CHILDREN << ") : [";
			for (size_t n : elem.children) {os << n << " ";}
			os << "]\n";
		}
		return os;
	}

	/// Element name printing
	template<BasicMeshElement Element_t>
	std::string elementTypeName() {
	    std::string name = static_cast<std::string>(vtk_id_to_string(Element_t::VTK_ID));
	    if constexpr (Element_t::HIERARCHICAL) name = "Hierarchical" + name;
	    if constexpr (Element_t::COLORABLE)    name = "Colored"      + name;
	    return name;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// VERTEX STRUCT DEFINITIONS AND OCTREE STORAGE
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	/////////////////////////////////////////////////
	/// A container for tracking the vertex information.
	/// Usually Point_t = gutil::Point<3,double>, but 2-D meshes or different precisions are allowed.
	/// This allows the vertices to be stored in a contiguous array as they all will require the same amount of memory. For a hexahedral mesh,=8.
	/////////////////////////////////////////////////
	template<gutil::pointlike Point_type = gutil::Point<3,gutil::FixedPoint<int64_t,0>>>
	struct BasicVertex {
		using Point_t  = Point_type;
		using Scalar_t = typename Point_t::scalar_type;
		static constexpr int DIM = Point_t::dim;

		Point_t coord; /// The location of this vertex in space.
		std::vector<size_t> elems; /// The elements that use this node
		
		BasicVertex(const Point_t& coord) : coord(coord), elems{} {}
		BasicVertex(Point_t&& coord)      : coord(std::move(coord)), elems{} {}
		BasicVertex() : coord(), elems{} {}
	};

	/////////////////////////////////////////////////
	/// Concept for a vertex
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshVertex = requires(T vertex) {
		typename T::Point_t;
		typename T::Scalar_t;
		requires gutil::pointlike<decltype(vertex.coord)>;
		{ vertex.elems } -> std::convertible_to<std::vector<size_t>>;
	};


	static_assert(BasicMeshVertex<BasicVertex<gutil::Point<3,double>>>, "BasicVertex<Point<3,double>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gutil::Point<2,double>>>, "BasicVertex<Point<2,double>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gutil::Point<3,float>>>,  "BasicVertex<Point<3,float>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gutil::Point<2,float>>>,  "BasicVertex<Point<2,float>> is not a BasicMeshVertex");


	template<BasicMeshVertex Vertex_t>
	bool operator==(const Vertex_t& left, const Vertex_t& right) {
		return left.coord == right.coord;
	}


	/// Vertex printing
	template<BasicMeshVertex Vertex_t>
	std::ostream& operator<<(std::ostream& os, const Vertex_t &vertex) {
		os << "index= " << vertex.index << "\n";
		os << "coord= " << vertex.coord << "\n";
		os << "elems (" << vertex.elems.size() << "): ";
		for (size_t n : vertex.elems) {
			os << n << " ";
		}
		os << "\n";
		os << "faces (" << vertex.faces.size() << "): ";
		for (size_t n : vertex.faces) {
			os << n << " ";
		}
		os << "\n";
		
		return os;
	}

	/// Vertex name printing
	template<BasicMeshVertex Vertex_t>
	std::string vertexTypeName() {
		if constexpr (std::same_as<Vertex_t,BasicVertex<typename Vertex_t::Point_t>>) {return "BasicVertex";}
		else {return "UNKNOWN";}
	}

	/////////////////////////////////////////////////
	/// A container for storing the vertices in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template<BasicMeshVertex Vertex_t, int N_DATA=64, typename T=double>
	class VertexOctree : public gutil::BasicParallelOctree<Vertex_t, true, Vertex_t::DIM, N_DATA, T>
	{
	public:
		using Parent_t = gutil::BasicParallelOctree<Vertex_t, true, Vertex_t::DIM, N_DATA, T>;
		using Data_t   = Vertex_t;
		using OctreePoint_t = gutil::Point<Vertex_t::DIM, T>;
		using Box_t          = gutil::Box<Vertex_t::DIM, T>;

		VertexOctree() : Parent_t() {}
		VertexOctree(const Box_t &bbox) : Parent_t(bbox) {}

	private:
		constexpr bool isValid(const Box_t &box, const Data_t &data) const override {return box.contains(static_cast<OctreePoint_t>(data.coord));}
		constexpr T dist2data(const Vertex_t::Point_t& point, const Data_t& data) const override {return gutil::squaredNorm(point-data.coord);}
	};




	/////////////////////////////////////////////////
	/// Concept for BasicMesh types
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshType = requires(T mesh) {
		// Must have relevant type aliases
		typename T::Element_t;
		typename T::HalfEdge_t;
		typename T::Vertex_t;
		typename T::GeoPoint_t;
		typename T::RefPoint_t;
		typename T::VertexList_t;
	};

	template<typename T>
	concept HierarchicalMeshType = BasicMeshType<T> and HierarchicalMeshElement<typename T::Element_t>;

	template<typename T>
	concept ColorableMeshType = BasicMeshType<T> and ColorableMeshElement<typename T::Element_t>;

	template<typename T>
	concept HierarchicalColorableMeshType = HierarchicalMeshType<T> and ColorableMeshElement<typename T::Element_t>;


}