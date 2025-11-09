#pragma once

#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

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
		{ elem.parent    } -> std::convertible_to<size_t>;
		{ elem.children  } -> std::convertible_to<std::vector<size_t>>;
	};


	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// ELEMENT STRUCT DEFINITIONS
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////
	/// Struct for tracking basic element information
	/////////////////////////////////////////////////
	struct BasicElement {
		std::vector<size_t> nodes;
		int vtkID;
		BasicElement() {}
		BasicElement(const int vtkID) : nodes(vtk_n_nodes(vtkID)), vtkID(vtkID) {}
		BasicElement(const std::vector<size_t> &nodes, const int vtkID) : nodes(nodes), vtkID(vtkID) {
			assert(nodes.size()==vtk_n_nodes(vtkID));
		}
	};
	static_assert(BasicMeshElement<BasicElement>, "BasicElement is not a BasicMeshElement");

	/////////////////////////////////////////////////
	/// Struct for colorable elements
	/////////////////////////////////////////////////
	struct ColorableElement : BasicElement {
		using BasicElement::BasicElement;
		size_t color = (size_t) -1;
	};
	static_assert(BasicMeshElement<ColorableElement>, "ColorableElement is not a BasicMeshElement");
	static_assert(ColorableMeshElement<ColorableElement>, "ColorableElement is not a ColorableMeshElement");

	/////////////////////////////////////////////////
	/// Struct for hierarchical elements
	/////////////////////////////////////////////////
	struct HierarchicalElement : BasicElement {
		size_t parent = (size_t) -1;
		std::vector<size_t> children;
		bool is_active = true;
		HierarchicalElement() {}
		HierarchicalElement(const int vtkID) : BasicElement(vtkID) {children.reserve(vtk_n_children(vtkID));}
		HierarchicalElement(const std::vector<size_t> &nodes, const int vtkID) : BasicElement(nodes, vtkID) {children.reserve(vtk_n_children(vtkID));}
	};
	static_assert(BasicMeshElement<HierarchicalElement>, "HierarchicalElement is not a BasicMeshElement");
	static_assert(HierarchicalMeshElement<HierarchicalElement>, "HierarchicalElement is not a HierarchicalMeshElement");

	/////////////////////////////////////////////////
	/// Struct for hierarchical colerable elements
	/////////////////////////////////////////////////
	struct HierarchicalColerableElement : BasicElement {
		size_t color = (size_t) -1;
		size_t parent = (size_t) -1;
		std::vector<size_t> children;
		bool is_active = true;
		HierarchicalColerableElement() {}
		HierarchicalColerableElement(const int vtkID) : BasicElement(vtkID) {children.reserve(vtk_n_children(vtkID));}
		HierarchicalColerableElement(const std::vector<size_t> &nodes, const int vtkID) : BasicElement(nodes, vtkID) {children.reserve(vtk_n_children(vtkID));}
	};
	static_assert(BasicMeshElement<HierarchicalColerableElement>, "HierarchicalColerableElement is not a BasicMeshElement");
	static_assert(ColorableMeshElement<HierarchicalColerableElement>, "HierarchicalColerableElement is not a ColorableMeshElement");
	static_assert(HierarchicalMeshElement<HierarchicalColerableElement>, "HierarchicalColerableElement is not a HierarchicalMeshElement");

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

	template<BasicMeshElement Element_t>
	std::ostream& operator<<(std::ostream& os, const Element_t &elem) {
		os << "vtkID= " << elem.vtkID << "\n";
		os << "nodes (" << elem.nodes.size() << "): ";
		for (size_t n : elem.nodes) {
			os << n << " ";
		}
		os << "\n";
		return os;
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
		using Scalar_t = typename Vertex_t::data_type;
		static constexpr int dim = Vertex_t::dimension;

		Vertex_t vertex; /// The location of this node in space.
		std::vector<size_t> elems; /// The elements that use this node
		std::vector<size_t> boundary_faces; /// The boundary faces/elements that use this node

		BasicNode(const Vertex_t &coord) : vertex(coord), elems(), boundary_faces(0) {}
		BasicNode() : vertex(), elems(), boundary_faces(0) {}
	};
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<3,double>>>, "BasicNode<Point<3,double>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<2,double>>>, "BasicNode<Point<2,double>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<3,float>>>, "BasicNode<Point<3,float>> is not a BasicMeshNode");
	static_assert(BasicMeshNode<BasicNode<gv::util::Point<2,float>>>, "BasicNode<Point<2,float>> is not a BasicMeshNode");

	/// Equality check for mesh nodes for use in the octree.
	template <BasicMeshNode Node_t>
	bool operator==(const Node_t &A, const Node_t &B) {return A.vertex==B.vertex;}


	template<BasicMeshNode Node_t>
	std::ostream& operator<<(std::ostream& os, const Node_t &node) {
		os << "vertex= " << node.vertex << "\n";
		os << "elems (" << node.elems.size() << "): ";
		for (size_t n : node.elems) {
			os << n << " ";
		}
		os << "\n";
		return os;
	}



	/////////////////////////////////////////////////
	/// A container for storing the nodes in an octree for more efficeint lookup. This is important as we must query if a node already exists in the mesh.
	/// @todo Determine if a kd-tree is better.
	/////////////////////////////////////////////////
	template<BasicMeshNode Node_t, int n_data=64>
	class NodeOctree : public gv::util::BasicOctree_Point<Node_t, Node_t::dim, n_data, typename Node_t::Scalar_t>
	{
	public:
		using Data_t = Node_t;
		using Box_t  = gv::util::Box<Node_t::dim, typename Node_t::Scalar_t>;

		NodeOctree() : //if bounding box is unknown ahead of time
			gv::util::BasicOctree_Point<Node_t, Node_t::dim, n_data, typename Node_t::Scalar_t>(1024) {}

		NodeOctree(const Box_t &bbox, const size_t capacity=1024) :
			gv::util::BasicOctree_Point<Node_t, Node_t::dim, n_data, typename Node_t::Scalar_t>(bbox, capacity) {}

	private:
		bool is_data_valid(const Box_t &box, const Data_t &data) const override {return box.contains(data.vertex);}
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