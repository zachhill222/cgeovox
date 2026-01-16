#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"

namespace gv::mesh {
	/////////////////////////////////////////////////
	/// This class create a view into another mesh by linking referencences to the elements and nodes.
	///
	/// @tparam Mesh_t The type of mesh that a view is being created for.
	/////////////////////////////////////////////////
	template<
			int              space_dim,
			int              ref_dim,
			Scalar           Scalar_t,
			BasicMeshElement ElementStruct_t = BasicElement
			>
	class MeshView : public BasicMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t> {
		/////////////////////////////////////////////////
		/// Friend functions to print the mesh information
		/////////////////////////////////////////////////
		// template <BasicMeshVertex U, BasicMeshElement Element_u, BasicMeshElement Face_u>
		// friend std::ostream& operator<<(std::ostream& os, const MeshView<U,Element_u,Face_u> &mesh);



	public:
		///aliases to satisfy BasicMeshType concept
		using Mesh_t = BasicMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t>;
		
		//aliases
		using typename Mesh_t::Index_t;
		using typename Mesh_t::DomainBox_t;
		using typename Mesh_t::RefBox_t;
		using typename Mesh_t::RefPoint_t;
		using typename Mesh_t::Point_t;
		using typename Mesh_t::Vertex_t;
		using typename Mesh_t::VertexList_t;
		using typename Mesh_t::ElementIterator_t;
		using typename Mesh_t::BoundaryIterator_t;

		//elements and faces have the same storage struct type, but it's nice to see the distinction in the code
		using typename Mesh_t::Element_t;
		using typename Mesh_t::Face_t;

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
	    size_t nVertices() const override {
	        return std::count(_node_mask.begin(), _node_mask.end(), true);
	    }

	    size_t nElements() const override {
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
			return ElementIterator_t(const_cast<MeshView<space_dim,ref_dim,Scalar_t,Element_t>*>(this),
				const_cast<std::vector<Element_t>*>(&_parent._elements), 0);
		}
		ElementIterator_t end()   const override {
			return ElementIterator_t(const_cast<MeshView<space_dim,ref_dim,Scalar_t,Element_t>*>(this),
				const_cast<std::vector<Element_t>*>(&_parent._elements), _parent._elements.size());
		}

	
		/////////////////////////////////////////////////
		/// Iterators for _boundary
		/////////////////////////////////////////////////
		BoundaryIterator_t boundaryBegin() const override {
			return BoundaryIterator_t(const_cast<MeshView<space_dim,ref_dim,Scalar_t,Element_t>*>(this),
				const_cast<std::vector<Face_t>*>(&_parent._boundary), 0);
		}
		BoundaryIterator_t boundaryEnd()   const override {
			return BoundaryIterator_t(const_cast<MeshView<space_dim,ref_dim,Scalar_t,Element_t>*>(this),
				const_cast<std::vector<Face_t>*>(&_parent._boundary), _parent._boundary.size());
		}


		/////////////////////////////////////////////////
		/// Iterators for _node
		/////////////////////////////////////////////////
		std::vector<Vertex_t>::const_iterator vertexBegin() const override {
			return _parent._vertices.cbegin();
		}
		std::vector<Vertex_t>::const_iterator vertexEnd()   const override {
			return _parent._vertices.cend();
		}
	};

	static_assert(BasicMeshType< MeshView<3,3,double,BasicElement >>,
		"MeshView is not a BasicMeshType with default template parameters.");
}