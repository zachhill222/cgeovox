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
		{ elem.nodes } -> std::convertible_to<std::vector<size_t>>;
		{ elem.vtkID } -> std::convertible_to<int>;
		{ elem.index } -> std::convertible_to<size_t>;
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
		std::vector<size_t> nodes;
		int vtkID;
		size_t index = (size_t) -1;
		BasicElement() : nodes(), vtkID(0) {}
		BasicElement(const BasicElement& other) : nodes(other.nodes), vtkID(other.vtkID), index(other.index) {}
		BasicElement(const int vtkID) : nodes(vtk_n_nodes(vtkID)), vtkID(vtkID) {}
		BasicElement(const std::vector<size_t> &nodes, const int vtkID) : nodes(nodes), vtkID(vtkID) {
			assert(nodes.size()==vtk_n_nodes(vtkID));
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
		ColoredElement(const std::vector<size_t> &nodes, const int vtkID) : BasicElement(nodes, vtkID) {}
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
		HierarchicalElement(const std::vector<size_t> &nodes, const int vtkID) : BasicElement(nodes, vtkID), children{} {children.reserve(vtk_n_children(vtkID));}
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
		HierarchicalColoredElement(const std::vector<size_t> &nodes, const int vtkID) : HierarchicalElement(nodes, vtkID) {}
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
		if (A.nodes.size()!=B.nodes.size()) {return false;}

		std::vector<size_t> a = A.nodes;
		std::vector<size_t> b = B.nodes;
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());

		return a == b;
	};


	/// Element hashing function for use in unordered_set (for example). The order of the element nodes is irrelevent to the hash value.
	struct ElemHashBitPack {
		size_t operator()(const BasicElement& ELEM) const {
			//sort the nodes
			std::vector<size_t> nodes = ELEM.nodes;
			std::sort(nodes.begin(), nodes.end());

			//initialize the hash by getting the last few bits from each node index
			size_t hash = 0;
			size_t bits_per_node;
			if constexpr (sizeof(size_t)==4) {bits_per_node=32/nodes.size();} //32-bit
			else if constexpr (sizeof(size_t)==8) {bits_per_node=64/nodes.size();} //64-bit
			else {bits_per_node=1;}

			size_t mask = (((size_t) 1) << bits_per_node) - 1; //exactly the last bits_per_node bits are 1

			for (size_t i=0; i<nodes.size(); i++) {
				size_t node_bits = nodes[i] & mask;
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
		os << "nodes (" << elem.nodes.size() << ") : [ ";
		for (size_t n : elem.nodes) {os << n << " ";}
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
	/// Concept for a node
	/////////////////////////////////////////////////
	template<typename T>
	concept BasicMeshNode = requires(T node) {
		typename T::Vertex_t;
		typename T::Scalar_t;
		requires gv::util::PointLike<decltype(node.vertex)>;
		{ node.elems  } -> std::convertible_to<std::vector<size_t>>;
		{ node.boundary_faces  } -> std::convertible_to<std::vector<size_t>>;
	};


	/////////////////////////////////////////////////
	/// A container for tracking the node information.
	/// Usually Point_t = gv::util::Point<3,double>, but 2-D meshes or different precisions are allowed.
	/// If the mesh has more than one element type, should be the maximum number of nodes required to define any of the used element types.
	/// This allows the nodes to be stored in a contiguous array as they all will require the same amount of memory. For a hexahedral mesh,=8.
	/////////////////////////////////////////////////
	template <gv::util::PointLike Point_type>
	struct BasicNode {
		using Vertex_t = Point_type;
		using Scalar_t = typename Vertex_t::Scalar_t;
		static constexpr int dim = Vertex_t::dimension;

		Vertex_t vertex; /// The location of this node in space.
		std::vector<size_t> elems; /// The elements that use this node
		std::vector<size_t> boundary_faces; /// The boundary faces/elements that use this node
		size_t index = (size_t) -1; /// The index of this node in _nodes. Sometimes helpful to have this recorded in the node.
		BasicNode(const Vertex_t &coord) : vertex(coord), elems(), boundary_faces(0) {}
		BasicNode() : vertex(), elems(), boundary_faces(0) {}
	};
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<3,double>>>, "BasicNode<Point<3,double>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<2,double>>>, "BasicNode<Point<2,double>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<3,float>>>,  "BasicNode<Point<3,float>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<2,float>>>,  "BasicNode<Point<2,float>> is not a BasicMeshNode");

	/// Equality check for mesh nodes for use in the octree.
	template <BasicMeshNode Node_t>
	bool operator==(const Node_t &A, const Node_t &B) {
		return A.vertex==B.vertex;
	}

	/// Node printing
	template<BasicMeshNode Node_t>
	std::ostream& operator<<(std::ostream& os, const Node_t &node) {
		os << "index= " << node.index << "\n";
		os << "vertex= " << node.vertex << "\n";
		os << "elems (" << node.elems.size() << "): ";
		for (size_t n : node.elems) {
			os << n << " ";
		}
		os << "\n";
		
		return os;
	}

	/// Node name printing
	template<BasicMeshNode Node_t>
	std::string nodeTypeName() {
		if constexpr (std::same_as<Node_t,BasicNode<typename Node_t::Vertex_t>>) {return "BasicNode";}
		else {return "UNKNOWN";}
	}

	/////////////////////////////////////////////////
	/// A container for storing the nodes in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template<BasicMeshNode Node_t, int N_DATA=64>
	class NodeOctree : public gv::util::BasicParallelOctree<Node_t, true, Node_t::dim, N_DATA, typename Node_t::Scalar_t>
	{
	public:
		using Parent_t = gv::util::BasicParallelOctree<Node_t, true, Node_t::dim, N_DATA, typename Node_t::Scalar_t>;
		using Data_t = Node_t;
		using Box_t  = gv::util::Box<Node_t::dim, typename Node_t::Scalar_t>;


		NodeOctree() : Parent_t() {}
		NodeOctree(const Box_t &bbox) : Parent_t(bbox) {}

	private:
		bool isValid(const Box_t &box, const Data_t &data) const override {return box.contains(data.vertex);}
		// bool isValid(const Box_t &box, const Data_t &data) const override {return gv::util::distance_squared(box,data.vertex) < 1e-2;}
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
		typename T::Vertex_t;
		typename T::ElementIterator_t;
		typename T::BoundaryIterator_t;
	};


}