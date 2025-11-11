#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/basic_mesh.hpp"
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
	/// This class defines a topological mesh that supports different element types. This class only tracks the topology of the mesh and topological operations.
	/// Points can be stored using various precisions (e.g., T=float, double, long double). 
	/// There is no requirement or guarentee that the elements are conforming or that they do not overlap.
	/// Be aware that overlapping elements (in space) are not topologically overlapping if they do not share a node and could have the same color.
	///
	/// @tparam T The precision that the vertices are stored in. It may be completely unnecessary to store the vertices in double precision for some meshes.
	///
	/// @todo Add data and types to this description.
	/////////////////////////////////////////////////
	template<BasicMeshNode        Node_t       = BasicNode<gv::util::Point<3,double>>,
			 ColorableMeshElement Element_t    = ColoredElement,
			 BasicMeshElement     Face_t       = BasicElement,
			 ColorMethod          COLOR_METHOD = ColorMethod::GREEDY>
	class ColoredMesh : public BasicMesh<Node_t,Element_t,Face_t>
	{
	public:
		//aliases
		template<int n=3>
		using Index_t            = gv::util::Point<n,size_t>;
		template<int n=3>
		using Box_t              = gv::util::Box<n, typename Node_t::Scalar_t>;
		using Vertex_t           = Node_t::Vertex_t;

	protected:	
		MeshColorManager<COLOR_METHOD, Element_t, 1024> _color_manager;   //used to manage the color of the elements

	public:
		ColoredMesh() : 
			BasicMesh<Node_t,Element_t,Face_t>(),
			_color_manager(this->_elements) {}
		
		ColoredMesh(const Box_t<3> &domain, const Index_t<3> &N) :
			BasicMesh<Node_t,Element_t,Face_t>(domain),
			_color_manager(this->_elements) {this->setVoxelMesh_Locked(domain, N);}
		
		ColoredMesh(const Box_t<2> &domain, const Index_t<2> &N) :
			BasicMesh<Node_t,Element_t,Face_t>(domain),
			_color_manager(this->_elements) {this->setPixelMesh_Locked(domain, N);}


		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		void insertElement_Locked(Element_t &ELEM) override {
			const size_t elem_idx = this->_elements.size();
			BasicMesh<Node_t,Element_t,Face_t>::insertElement_Locked(ELEM);
			color_Locked(elem_idx);
		}


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
		void insertElement_Unlocked(Element_t &ELEM, const size_t elem_idx) override {
			BasicMesh<Node_t,Element_t,Face_t>::insertElement_Unlocked(ELEM, elem_idx);
			color_Unlocked(elem_idx);
		}


		/////////////////////////////////////////////////
		/// Color the specified element. Locked to a single thread.
		///
		/// @param elem_idx The index of the element to color
		/////////////////////////////////////////////////
		void color_Locked(const size_t elem_idx) {
			std::vector<size_t> neighbors;
			this->getElementNeighbors_Locked(elem_idx, neighbors);
			_color_manager.setColor_Locked(elem_idx, neighbors);
		}


		/////////////////////////////////////////////////
		/// Color the specified element. Not locked to a single thread.
		/// The method that calls this must ensure tht _elements[elem_idx] is writable and
		///     _elements[k] is readible for any element k that is a neigbor of element elem_idx.
		///
		/// @param elem_idx The index of the element to color
		/////////////////////////////////////////////////
		void color_Unlocked(const size_t elem_idx) {
			std::vector<size_t> neighbors;
			this->getElementNeighbors_Unlocked(elem_idx, neighbors);
			_color_manager.setColor_Unlocked(elem_idx, neighbors);
		}

		/////////////////////////////////////////////////
		/// Check if the coloring is valid
		/////////////////////////////////////////////////
		bool colorsValid_Unlocked() const;


		/// Friend function to print the mesh information
		template <BasicMeshNode U, ColorableMeshElement Element_u, BasicMeshElement Face_u, ColorMethod COLORMETHOD>
		friend std::ostream& operator<<(std::ostream& os, const ColoredMesh<U,Element_u,Face_u,COLORMETHOD> &mesh);
	};


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, BasicMeshElement Face_t, ColorMethod COLOR_METHOD>
	bool ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD>::colorsValid_Unlocked() const {
		for (auto it=this->begin(); it!=this->end(); ++it) {
			std::vector<size_t> neighbors;
			this->getElementNeighbors_Unlocked(it->index, neighbors);
			for (size_t n_idx: neighbors) {
				if (it->color == this->_elements[n_idx].color) {
					std::cout << "elements " << it->index << " and " << n_idx << " color (" << it->color << ") colision" << std::endl;
					return false;
				}
			}
		}
		return true;
	}


	template<BasicMeshNode Node_t, ColorableMeshElement Element_t, BasicMeshElement Face_t, ColorMethod COLOR_METHOD>
	std::ostream& operator<<(std::ostream& os, const ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD> &mesh) {
		const BasicMesh<Node_t,Element_t,Face_t> &base_mesh = mesh;
		os << base_mesh;
		os << mesh._color_manager;
		return os;
	}
}

