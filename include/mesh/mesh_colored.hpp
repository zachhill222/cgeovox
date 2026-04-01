#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_color_manager.hpp"

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

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
			ColorableMeshElement Element_type = ColorableElement<VOXEL_VTK_ID>,
			Scalar               Scalar_type  = gutil::FixedPoint64<>,
			ColorMethod          COLOR_METHOD = ColorMethod::BALANCED,
			size_t               MAX_COLORS   = 64
			>
	class ColoredMesh : public BasicMesh<Element_type,Scalar_type>
	{
		using BASE = BasicMesh<Element_type,Scalar_type>;
	public:
		//import aliases
		using typename BASE::Element_t;
		using typename BASE::HalfEdge_t;
		using typename BASE::Vertex_t;
		
		using typename BASE::Scalar_t;
		using typename BASE::GeoPoint_t; //data type for computing spatial coordinates
		using typename BASE::RefPoint_t; //data type for evaluating basis functions, computing jacobians, etc.
		
		using typename BASE::Mesh_t = ColoredMesh<Element_type,Scalar_type,COLOR_METHOD,MAX_COLORS>; //type of this mesh
		using typename BASE::Index_t; //index for creating structured mesh in the constructor
		using typename BASE::GeoBox_t; //boxes in the domain space
		using typename BASE::RefBox_t; //boxes in the reference space

		using typename BASE::VertexList_t;
		using typename BASE::ElementLogic_t; //type to handle logic of creating children, getting faces, etc.

	protected:	
		MeshColorManager<COLOR_METHOD, Element_t, MAX_COLORS> _color_manager;   //used to manage the color of the elements

	public:
		ColoredMesh() : BASE(),	_color_manager(this->_elements) {}

		ColoredMesh(const GeoBox_t &domain) : BASE(domain), _color_manager(this->_elements) {}

		ColoredMesh(const RefBox_t &domain) requires(BASE::REF_DIM==2) : 
			BASE(domain),
			_color_manager(this->_elements) {}
		
		ColoredMesh(const GeoBox_t &domain, const Index_t &N) 
			requires (VTK_ID==VOXEL_VTK_ID or VTK_ID==HEXAHEDRON_VTK_ID or VTK_ID==PIXEL_VTK_ID or VTK_ID==QUAD_VTK_ID)
			: BASE(domain), _color_manager(this->_elements)
		{
			if constexpr (BASE::REF_DIM==3) {this->build_voxel_mesh(domain, N);}
			else if constexpr (BASE::REF_DIM==2) {this->build_pixel_mesh(domain, N);}
			else {throw std::runtime_error("ColoredMesh: can't mesh domain");}
		}

		virtual ~ColoredMesh() {}

		/////////////////////////////////////////////////
		/// A method to insert a new element into the mesh. The element must be constructed from specified existing nodes.
		/// The existing nodes will be updated but no new nodes will be created.
		/////////////////////////////////////////////////
		template<bool ASYNC=false>
		virtual size_t insert_element(Element_t&& element, size_t index = (size_t) -1) override
		{
			index = BASE::insert_element<ASYNC>(std::move(element), index);
			color_element<ASYNC>(index);
		}

		/////////////////////////////////////////////////
		/// Methods to get information about the coloring
		/////////////////////////////////////////////////
		size_t nColors() const {return _color_manager.nColors();}
		size_t colorCount(const size_t c) const {return _color_manager.colorCount(c);}


		/////////////////////////////////////////////////
		/// Color the specified element. Locked to a single thread.
		///
		/// @param e_idx The index of the element to color
		/////////////////////////////////////////////////
		template<bool ASYNC=false>
		void color_element(const size_t e_idx)
		{
			std::vector<size_t> neighbors;
			this->get_element_neighbors(e_idx, neighbors);
			if constexpr (ASYNC) {
				_color_manager.set_color_unlocked(e_idx, neighbors);
			}
			else {
				_color_manager.set_color_locked(e_idx, neighbors);
			}
		}


		/////////////////////////////////////////////////
		/// Check if the coloring is valid
		/////////////////////////////////////////////////
		bool are_colors_valid() const
		{
			for (size_t e_idx=0; e_idx<this->_elements.size(); ++e_idx) {
				const Element_t& ELEM = this->_elements[e_idx];

				std::vector<size_t> neighbors;
				this->get_element_neighbors(e_idx, neighbors);
				for (size_t n_idx: neighbors) {
					if (ELEM.color == this->_elements[n_idx].color) {
						std::cout << "elements " << e_idx << " and " << n_idx << " color (" << ELEM.color << ") colision" << std::endl;
						return false;
					}
				}
			}
			return true;
		}


		/// Friend function to print the mesh information
		template<ColorableMeshType Mesh_type>
		friend std::ostream& operator<<(std::ostream& os, const Mesh_type& mesh);
	};


	template<ColorableMeshType Mesh_t>
	std::ostream& operator<<(std::ostream& os, const Mesh_t& mesh) {
		const BasicMesh<typename Mesh_t::Element_t, typename Mesh_t::Scalar_t>& base_mesh = mesh;
		os << base_mesh;
		os << mesh._color_manager;
		return os;
	}
}

