#pragma once

#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree_parallel.hpp"

#include <concepts>
#include <vector>
#include <iostream>

namespace gv::mesh
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT CONCEPTS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////




	/////////////////////////////////////////////////
	/// Concept for an element
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshElement = requires(T elem) {
		{ elem.vertices } -> std::convertible_to<std::vector<size_t>>;
		{ elem.vtkID    } -> std::convertible_to<int>;
		{ elem.index    } -> std::convertible_to<size_t>;
	};

	/////////////////////////////////////////////////
	/// Concept for a colerable element
	/////////////////////////////////////////////////
	template<typename T>
	concept ColorableMeshElement = BasicMeshElement<T> and requires(T elem) {
		{ elem.color } -> std::convertible_to<size_t>;
	};

	/////////////////////////////////////////////////
	/// Concept for a hierarchical element
	/////////////////////////////////////////////////
	template<typename T>
	concept HierarchicalMeshElement = BasicMeshElement<T> and requires(T elem) {
		{ elem.is_active } -> std::convertible_to<bool>;
		{ elem.depth     } -> std::convertible_to<size_t>;
		{ elem.parent    } -> std::convertible_to<size_t>;
		{ elem.children  } -> std::convertible_to<std::vector<size_t>>;
	};

	/////////////////////////////////////////////////
	/// Concept for a hierarchical colorable element
	/////////////////////////////////////////////////
	template<typename T>
	concept HierarchicalColorableMeshElement = HierarchicalMeshElement<T> and ColorableMeshElement<T>;

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT STRUCT DEFINITIONS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////
	/// Struct for tracking boundary information
	/////////////////////////////////////////////////
	struct FaceTracker {
		size_t elem_idx;
		int elem_face;
	};

	/////////////////////////////////////////////////
	/// Struct for tracking basic element information
	/////////////////////////////////////////////////
	struct BasicElement {
		std::vector<size_t> vertices;
		int vtkID;
		size_t index = (size_t) -1;
		BasicElement() : vertices(), vtkID(0) {}
		BasicElement(const BasicElement& other) : vertices(other.vertices), vtkID(other.vtkID), index(other.index) {}
		BasicElement(const int vtkID) : vertices(vtk_n_vertices(vtkID)), vtkID(vtkID) {}
		BasicElement(const std::vector<size_t> &vertices, const int vtkID) : vertices(vertices), vtkID(vtkID) {
			assert(vertices.size()==vtk_n_vertices(vtkID));
		}
	};
	static_assert(BasicMeshElement<BasicElement>, "BasicElement is not a BasicMeshElement");

	/////////////////////////////////////////////////
	/// Struct for colorable elements
	/////////////////////////////////////////////////
	struct ColoredElement : BasicElement {
		size_t color = (size_t) -1;
		ColoredElement() : BasicElement() {}
		ColoredElement(const int vtkID) : BasicElement(vtkID) {}
		ColoredElement(const std::vector<size_t> &vertices, const int vtkID) : BasicElement(vertices, vtkID) {}
		ColoredElement(const BasicElement& other) : BasicElement(other) {}
		ColoredElement(const ColoredElement& other) : 
			BasicElement(other),
			color(other.color) {}
	};
	static_assert(BasicMeshElement<ColoredElement>, "ColoredElement is not a BasicMeshElement");
	static_assert(ColorableMeshElement<ColoredElement>, "ColoredElement is not a ColorableMeshElement");

	/////////////////////////////////////////////////
	/// Struct for hierarchical elements
	/////////////////////////////////////////////////
	struct HierarchicalElement : BasicElement {
		size_t parent  = (size_t) -1;
		size_t depth   = 0;
		bool is_active = false;
		std::vector<size_t> children;
		HierarchicalElement() : BasicElement(), children{} {}
		HierarchicalElement(const int vtkID) : BasicElement(vtkID), children{} {children.reserve(vtk_n_children(vtkID));}
		HierarchicalElement(const std::vector<size_t> &vertices, const int vtkID) : BasicElement(vertices, vtkID), children{} {children.reserve(vtk_n_children(vtkID));}
		HierarchicalElement(const BasicElement& other) : BasicElement(other), children{} {children.reserve(vtk_n_children(vtkID));}
		HierarchicalElement(const HierarchicalElement& other) : 
			BasicElement(other),
			parent(other.parent),
			depth(other.depth),
			is_active(other.is_active),
			children(other.children) {}
	};
	static_assert(BasicMeshElement<HierarchicalElement>, "HierarchicalElement is not a BasicMeshElement");
	static_assert(HierarchicalMeshElement<HierarchicalElement>, "HierarchicalElement is not a HierarchicalMeshElement");

	/////////////////////////////////////////////////
	/// Struct for hierarchical colerable elements
	/////////////////////////////////////////////////
	struct HierarchicalColoredElement : HierarchicalElement {
		size_t color = (size_t) -1;
		HierarchicalColoredElement() {}
		HierarchicalColoredElement(const int vtkID) : HierarchicalElement(vtkID) {}
		HierarchicalColoredElement(const std::vector<size_t> &vertices, const int vtkID) : HierarchicalElement(vertices, vtkID) {}
		HierarchicalColoredElement(const BasicElement& other) : HierarchicalElement(other) {}
		HierarchicalColoredElement(const HierarchicalColoredElement& other) : 
			HierarchicalElement(other),
			color(other.color) {}

	};
	static_assert(BasicMeshElement<HierarchicalColoredElement>, "HierarchicalColoredElement is not a BasicMeshElement");
	static_assert(ColorableMeshElement<HierarchicalColoredElement>, "HierarchicalColoredElement is not a ColorableMeshElement");
	static_assert(HierarchicalMeshElement<HierarchicalColoredElement>, "HierarchicalColoredElement is not a HierarchicalMeshElement");

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT HELPER FUNCTIONS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Check if two elements are the same (up to orientation)
	bool operator==(const BasicElement &A, const BasicElement &B) {
		if (A.vtkID!=B.vtkID) {return false;}
		if (A.vertices.size()!=B.vertices.size()) {return false;}

		std::vector<size_t> a = A.vertices;
		std::vector<size_t> b = B.vertices;
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());

		return a == b;
	};


	/// Element hashing function for use in unordered_set (for example). The order of the element vertices is irrelevent to the hash value.
	struct ElemHashBitPack {
		size_t operator()(const BasicElement& ELEM) const {
			//sort the vertices
			std::vector<size_t> vertices = ELEM.vertices;
			std::sort(vertices.begin(), vertices.end());

			//initialize the hash by getting the last few bits from each node index
			size_t hash = 0;
			size_t bits_per_node;
			if constexpr (sizeof(size_t)==4) {bits_per_node=32/vertices.size();} //32-bit
			else if constexpr (sizeof(size_t)==8) {bits_per_node=64/vertices.size();} //64-bit
			else {bits_per_node=1;}

			size_t mask = (((size_t) 1) << bits_per_node) - 1; //exactly the last bits_per_node bits are 1

			for (size_t i=0; i<vertices.size(); i++) {
				size_t node_bits = vertices[i] & mask;
				hash |= (node_bits << (i*bits_per_node));
			}

			//scramble the hash (MurmurHash3)
			if constexpr (sizeof(size_t)==4) {
				hash ^= hash >> 16;
				hash *= 0x85ebca6b;
				hash ^= hash >> 16;
				hash *= 0xc2b2ae35;
				hash ^= hash >> 16;
			} else if constexpr (sizeof(size_t)==8) {
				hash ^= hash >> 33;
				hash *= 0xff51afd7ed558ccdULL;
				hash ^= hash >> 33;
				hash *= 0xc4ceb9fe1a85ec53ULL;
				hash ^= hash >> 33;
			}

			return hash;
		}
	};


	/// Element printing
	template<BasicMeshElement Element_t>
	std::ostream& operator<<(std::ostream& os, const Element_t &elem) {
		os << "vtkID= " << elem.vtkID << " (" << vtk_id_to_string(elem.vtkID) << ")\n";
		os << "index= " << elem.index << "\n";
		os << "vertices (" << elem.vertices.size() << ") : [ ";
		for (size_t n : elem.vertices) {os << n << " ";}
		os << "]\n";

		if constexpr (ColorableMeshElement<Element_t>) {
			os << "color= " << elem.color << "\n";
		}

		if constexpr (HierarchicalMeshElement<Element_t>) {
			os << "is_active= " << elem.is_active << "\n";
			os << "depth= " << elem.depth << "\n";
			os << "parent= " << elem.parent << "\n";
			os << "children ("  << elem.children.size() << ") : [";
			for (size_t n : elem.children) {os << n << " ";}
			os << "]\n";
		}
		return os;
	}

	/// Element name printing
	template<BasicMeshElement Element_t>
	std::string elementTypeName() {
		if constexpr (std::same_as<Element_t,BasicElement>) {return "BasicElement";}
		if constexpr (std::same_as<Element_t,ColoredElement>) {return "ColoredElement";}
		if constexpr (std::same_as<Element_t,HierarchicalElement>) {return "HierarchicalElement";}
		if constexpr (std::same_as<Element_t,HierarchicalColoredElement>) {return "HierarchicalColoredElement";}
		else {return "UNKNOWN";}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// NODE STRUCT DEFINITIONS AND OCTREE STORAGE
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////
	/// Concept for a vertex
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshVertex = requires(T vertex) {
		typename T::Point_t;
		typename T::Scalar_t;
		requires gv::util::PointLike<decltype(vertex.coord)>;
		{ vertex.elems  } -> std::convertible_to<std::vector<size_t>>;
		{ vertex.boundary_faces  } -> std::convertible_to<std::vector<size_t>>;
	};


	/////////////////////////////////////////////////
	/// A container for tracking the node information.
	/// Usually Point_t = gv::util::Point<3,double>, but 2-D meshes or different precisions are allowed.
	/// If the mesh has more than one element type, should be the maximum number of vertices required to define any of the used element types.
	/// This allows the vertices to be stored in a contiguous array as they all will require the same amount of memory. For a hexahedral mesh,=8.
	/////////////////////////////////////////////////
	template <gv::util::PointLike Point_type>
	struct BasicVertex {
		using Point_t  = Point_type;
		using Scalar_t = typename Point_t::Scalar_t;
		static constexpr int dim = Point_t::dimension;

		Point_t coord; /// The location of this node in space.
		std::vector<size_t> elems; /// The elements that use this node
		std::vector<size_t> boundary_faces; /// The boundary faces/elements that use this node
		size_t index = (size_t) -1; /// The index of this node in _vertices. Sometimes helpful to have this recorded in the node.
		BasicVertex(const Point_t &coord) : coord(coord), elems(), boundary_faces(0) {}
		BasicVertex() : coord(), elems(), boundary_faces(0) {}
	};
	static_assert(BasicMeshVertex<BasicVertex<gv::util::Point<3,double>>>, "BasicVertex<Point<3,double>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gv::util::Point<2,double>>>, "BasicVertex<Point<2,double>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gv::util::Point<3,float>>>,  "BasicVertex<Point<3,float>> is not a BasicMeshVertex");
	static_assert(BasicMeshVertex<BasicVertex<gv::util::Point<2,float>>>,  "BasicVertex<Point<2,float>> is not a BasicMeshVertex");

	/// Equality check for mesh vertices for use in the octree.
	template <BasicMeshVertex Node_t>
	bool operator==(const Node_t &A, const Node_t &B) {
		return A.coord==B.coord;
	}

	/// Node printing
	template<BasicMeshVertex Node_t>
	std::ostream& operator<<(std::ostream& os, const Node_t &node) {
		os << "index= " << node.index << "\n";
		os << "coord= " << node.coord << "\n";
		os << "elems (" << node.elems.size() << "): ";
		for (size_t n : node.elems) {
			os << n << " ";
		}
		os << "\n";
		
		return os;
	}

	/// Node name printing
	template<BasicMeshVertex Node_t>
	std::string nodeTypeName() {
		if constexpr (std::same_as<Node_t,BasicVertex<typename Node_t::Point_t>>) {return "BasicVertex";}
		else {return "UNKNOWN";}
	}

	/////////////////////////////////////////////////
	/// A container for storing the vertices in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template<BasicMeshVertex Node_t, int N_DATA=64, Scalar T=double>
	class NodeOctree : public gv::util::BasicParallelOctree<Node_t, true, Node_t::dim, N_DATA, T>
	{
	public:
		using Parent_t = gv::util::BasicParallelOctree<Node_t, true, Node_t::dim, N_DATA, T>;
		using Data_t = Node_t;
		using Box_t  = gv::util::Box<Node_t::dim, T>;


		NodeOctree() : Parent_t() {}
		NodeOctree(const Box_t &bbox) : Parent_t(bbox) {}

	private:
		bool isValid(const Box_t &box, const Data_t &data) const override {return box.contains(data.coord);}
	};



	/////////////////////////////////////////////////
	/// Concept for BasicMesh types
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshType = requires(T mesh) {
		// Must have relevant type aliases
		typename T::element_type;
		typename T::face_type;
		typename T::node_type;
		typename T::Point_t;
		typename T::ElementIterator_t;
		typename T::BoundaryIterator_t;
	};


}