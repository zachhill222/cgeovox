#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_color_manager.hpp"

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include <vector>
#include <cassert>
#include <iostream>
#include <omp.h>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// This class extends the BasicMesh class to color the elements. The coloring is handled by the _color_manager. Each element has
	/// an ELEM.color field of type size_t. Any two elements in the mesh with the same color are guaranteed to have no nodes in common.
	/// When the ColorMethod::GREEDY is used, each element will recieve the first (lowest) valid color value. When the ColorMethod::BALANCED
	/// is used, each element will recieve the valid color with associated with the least number of elements.
	/// 
	/// @tparam space_dim        The dimension of the space that the mesh is embedded in. Usually 3. (Untested for 2).
	/// @tparam ref_dim          The dimension of the space that the reference elements are embedded in. Usually 3. (2 for surface meshes).
	/// @tparam Scalar_t         The scalar type to emulate the real line. This should be robust under comparisions and arithmetic ordering. (e.g., FixedPrecision instead of float)
	/// @tparam ElementStruct_t  The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam COLOR_METHOD     The method used to color the elements. Either greedy (ColorMethod::GREEDY) or balanced (ColorMethod::BALANCED).
	/// @tparam MAX_COLORS       The maximum number of colors that the mesh can have. Colors are stored in an std::array<std::atomic<size_t>> structure that is not resized.
	/////////////////////////////////////////////////
	template<
			int              space_dim,
			int              ref_dim,
			Scalar           Scalar_t,
			BasicMeshElement ElementStruct_t = BasicElement,
			ColorMethod      COLOR_METHOD    = ColorMethod::BALANCED,
			size_t           MAX_COLORS      = 64
			>
	class ColoredMesh : public BasicMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t>
	{
		using BaseClass = BasicMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t>;
	public:
		//aliases
		using typename BaseClass::Index_t;
		using typename BaseClass::DomainBox_t;
		using typename BaseClass::RefBox_t;
		using typename BaseClass::Point_t;
		using typename BaseClass::RefPoint_t;
		using typename BaseClass::Vertex_t;
		using typename BaseClass::VertexList_t;
		using typename BaseClass::ElementIterator_t;
		using typename BaseClass::BoundaryIterator_t;

		//elements and faces have the same storage struct type, but it's nice to see the distinction in the code
		using typename BaseClass::Element_t;
		using typename BaseClass::Face_t;

	protected:	
		MeshColorManager<COLOR_METHOD, Element_t, MAX_COLORS> _color_manager;   //used to manage the color of the elements

	public:
		ColoredMesh() : 
			BaseClass(),
			_color_manager(this->_elements) {}

		ColoredMesh(const DomainBox_t &domain) : 
			BaseClass(domain),
			_color_manager(this->_elements) {}

		ColoredMesh(const RefBox_t &domain) requires (ref_dim<space_dim) : 
			BaseClass(domain),
			_color_manager(this->_elements) {}
		
		ColoredMesh(const RefBox_t &domain, const Index_t &N, const bool useIsopar=false) : BaseClass(domain), _color_manager(this->_elements) {
			if constexpr (ref_dim==3) {this->setVoxelMesh_Locked(domain, N, useIsopar);}
			else if constexpr (ref_dim==2) {this->setPixelMesh_Locked(domain, N, useIsopar);}
			else {throw std::runtime_error("ColoredMesh: can't mesh domain");}
		}

		virtual ~ColoredMesh() {}

		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		///
		/// @param ELEM The element to be inserted. The nodes must already be populated. The element will be appended to _elements via _elements.push_back(std::move(ELEM)).
		/////////////////////////////////////////////////
		void insertElement_Locked(Element_t &ELEM) override {
			const size_t elem_idx = this->_elements.size();
			BaseClass::insertElement_Locked(ELEM);
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
			BaseClass::insertElement_Unlocked(ELEM, elem_idx);
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
		bool colorsValid_Unlocked() const {
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


		/// Friend function to print the mesh information
		template<int space_dim_u, int ref_dim_u, Scalar Scalar_u, ColorableMeshElement Element_u, ColorMethod COLORMETHOD>
		friend std::ostream& operator<<(std::ostream& os, const ColoredMesh<space_dim_u,ref_dim_u,Scalar_u,Element_u,COLORMETHOD> &mesh);
	};


	template<int space_dim, int ref_dim, Scalar Scalar_t, ColorableMeshElement Element_t, ColorMethod COLOR_METHOD>
	std::ostream& operator<<(std::ostream& os, const ColoredMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD> &mesh) {
		const BasicMesh<space_dim,ref_dim,Scalar_t,Element_t> &base_mesh = mesh;
		os << base_mesh;
		os << mesh._color_manager;
		return os;
	}
}

