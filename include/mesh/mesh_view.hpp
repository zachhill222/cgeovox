#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"

namespace gv::mesh {


	/////////////////////////////////////////////////
	/// This class create a view into another mesh by linking referencences to the elements and nodes.
	///
	/// @tparam Mesh_t The type of mesh that a view is being created for.
	/////////////////////////////////////////////////
	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	class MeshView : public BasicMesh<Node_t,Element_t,Face_t> {
		/////////////////////////////////////////////////
		/// Friend functions to print the mesh information
		/////////////////////////////////////////////////
		// template <BasicMeshNode U, BasicMeshElement Element_u, BasicMeshElement Face_u>
		// friend std::ostream& operator<<(std::ostream& os, const MeshView<U,Element_u,Face_u> &mesh);



	public:
		///aliases to satisfy BasicMeshType concept
		using Mesh_t = BasicMesh<Node_t,Element_t,Face_t>;
		using element_type = Element_t;
		using face_type = Face_t;
		using node_type = Node_t;
		using Vertex_t = typename Mesh_t::Vertex_t;
		using ElementIterator_t = typename Mesh_t::ElementIterator_t;
		using BoundaryIterator_t = typename Mesh_t::BoundaryIterator_t;
		using NodeList_t = typename Mesh_t::NodeList_t;

	protected:
		const Mesh_t     &_parent;
		std::vector<bool> _element_mask;
		std::vector<bool> _boundary_mask;
		std::vector<bool> _node_mask;

	public:
		MeshView(const Mesh_t &mesh) : Mesh_t(),
			_parent(mesh),
			_element_mask(mesh._elements.size(), true),
			_boundary_mask(mesh._boundary.size(), true),
			_node_mask(mesh._nodes.size(), true)
			{
				// this->_nodes.set_bbox(_parent._nodes.bbox());
				// for (size_t n=0; n<_parent._nodes.size(); n++) {this->_nodes.push_back(_parent._nodes[n]);}
			}

		// Count methods
	    size_t nNodes() const override {
	        return std::count(_node_mask.begin(), _node_mask.end(), true);
	    }

	    size_t nElems() const override {
	        return std::count(_element_mask.begin(), _element_mask.end(), true);
	    }

	    size_t nBoundaryFaces() const override  {
	        return std::count(_boundary_mask.begin(), _boundary_mask.end(), true);
	    }

	    // Validation methods (used by iterators)
	    bool isElementValid(const Element_t& ELEM) const override  {
	        return ELEM.index < _element_mask.size() && _element_mask[ELEM.index] && _parent.isElementValid(ELEM);
	    }
	    
	    bool isFaceValid(const Face_t& FACE) const override  {
	        return FACE.index < _boundary_mask.size() && _boundary_mask[FACE.index] && _parent.isFaceValid(FACE);
	    }

	    /////////////////////////////////////////////////
		/// Iterators for _elements. These are the most common iterators and get the defualt begin() and end() methods.
		/// Marked as virtual so they can be pointed at other arrays for views into the mesh. (i.e., the MeshView class).
		/////////////////////////////////////////////////
		ElementIterator_t begin() const override {
			return ElementIterator_t(const_cast<MeshView<Node_t,Element_t,Face_t>*>(this),
				const_cast<std::vector<Element_t>*>(&_parent._elements), 0);
		}
		ElementIterator_t end()   const override {
			return ElementIterator_t(const_cast<MeshView<Node_t,Element_t,Face_t>*>(this),
				const_cast<std::vector<Element_t>*>(&_parent._elements), _parent._elements.size());
		}

	
		/////////////////////////////////////////////////
		/// Iterators for _boundary
		/////////////////////////////////////////////////
		BoundaryIterator_t boundaryBegin() const override {
			return BoundaryIterator_t(const_cast<MeshView<Node_t,Element_t,Face_t>*>(this),
				const_cast<std::vector<Face_t>*>(&_parent._boundary), 0);
		}
		BoundaryIterator_t boundaryEnd()   const override {
			return BoundaryIterator_t(const_cast<MeshView<Node_t,Element_t,Face_t>*>(this),
				const_cast<std::vector<Face_t>*>(&_parent._boundary), _parent._boundary.size());
		}


		/////////////////////////////////////////////////
		/// Iterators for _node
		/////////////////////////////////////////////////
		std::vector<Node_t>::const_iterator nodeBegin() const override {
			return _parent._nodes.cbegin();
		}
		std::vector<Node_t>::const_iterator nodeEnd()   const override {
			return _parent._nodes.cend();
		}
	};

	static_assert(BasicMeshType< MeshView<BasicNode<gv::util::Point<3,double>>, BasicElement, BasicElement >>,
		"MeshView is not a BasicMeshType with default template parameters.");
}