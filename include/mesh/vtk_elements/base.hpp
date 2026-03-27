#pragma once

#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "concepts.hpp"

#include <array>
#include <vector>
#include <string_view>
#include <cassert>

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// Interface for VTK element types
	///
	/// This uses CRTP for when the element type is known at compile time
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t, typename DERIVED, int VTK_ID_>
	struct VTK_ELEMENT
	{
		static constexpr int VTK_ID            = VTK_ID_;
		static constexpr std::string_view NAME = vtk_id_to_string(VTK_ID);
		static constexpr int N_VERTICES        = vtk_n_vertices(VTK_ID);
		static constexpr int N_FACES           = vtk_n_faces(VTK_ID);
		static constexpr int FACE_VTK_ID       = vtk_face_id(VTK_ID);
		static constexpr int N_CHILDREN        = vtk_n_children(VTK_ID);
		static constexpr int N_VERT_ON_SPLIT   = vtk_n_vertices_when_split(VTK_ID);
		static constexpr int REF_DIM           = vtk_ref_dim(VTK_ID);

		using GeoPoint_t = typename Mesh_t::Point_t;      //type of point in space (e.g., mesh vertex coordinates)
		using Scalar_t   = typename GeoPoint_t::scalar_type; //likely a fixed precision type
		using RefPoint_t = gutil::Point<3,double>; //always use 3 so the polymorphic wrapper has a consistent return type with std::variant
		using Jac_t      = gutil::Matrix<GeoPoint_t::dim, RefPoint_t::dim, double, false>; //type of jacobian matrix

		static_assert(std::is_same_v<typename RefPoint_t::scalar_type, double>);

		size_t last_element = (size_t) -1; //the last element that was set
		std::array<size_t, N_VERTICES> vertices;
		std::array<GeoPoint_t, N_VERTICES> vertex_coords;
		std::array<GeoPoint_t, N_VERT_ON_SPLIT> child_vertex_coords;

		void set_element(const Mesh_t& mesh, const size_t element_index)
		{
			last_element = element_index;

			const auto& ELEM = mesh.getElement(element_index);
			assert(DERIVED::VTK_ID == ELEM.vtkID);

			for (int i=0; i<N_VERTICES; ++i) {
				const size_t v_idx = ELEM.vertices[i];
				vertices[i] = v_idx;
				vertex_coords[i] = mesh.getVertex(v_idx).coord;
			}
		}

		inline void set_child_vertices()
		{
			static_cast<DERIVED*>(this)->set_child_vertices_impl();
		}

		//return indices (in the correct order) that define the specified child
		//these indices point into this->child_vertex_coords
		//a child element has the same element type as the parent
		inline std::array<size_t, N_VERTICES> get_child_local_vertices(const int child_number) const
		{
			assert(0 <= child_number and child_number < N_CHILDREN);
			return static_cast<const DERIVED*>(this)->get_child_local_vertices_impl(child_number);
		}

		//return indices (in the correct order of the face element type)
		//that define the specified face. These indices are elements of this->vertices and point
		//into mesh->vertices
		inline std::array<size_t, vtk_n_vertices(FACE_VTK_ID)> get_face_vertices(const int face_number) const
		{
			assert(0 <= face_number and face_number < N_FACES);
			return static_cast<const DERIVED*>(this)->get_face_vertices_impl(face_number);	
		}

		//return indices (in the correct order of the face element type)
		//that define a child of a face. These indices point into this->child_vertices.
		//this must be compatible with creating the face element type and then getting that face's child vertices
		inline std::array<size_t, vtk_n_vertices_when_split(FACE_VTK_ID)> get_face_child_local_vertices(const int face_number) const
		{
			assert(0 <= face_number and face_number < N_FACES);
			return static_cast<const DERIVED*>(this)->get_face_child_local_vertices_impl(face_number);	
		}

		inline bool contains(const GeoPoint_t& point) const
		{
			return static_cast<const DERIVED*>(this)->contains_impl(point);
		}

		inline RefPoint_t geo2ref(const GeoPoint_t& point) const
		{
			return static_cast<const DERIVED*>(this)->geo2ref_impl(point);
		}

		inline GeoPoint_t ref2geo(const RefPoint_t& point) const
		{
			return static_cast<const DERIVED*>(this)->ref2geo_impl(point);
		}

		inline Jac_t jacobian(const RefPoint_t& point) const
		{
			return static_cast<const DERIVED*>(this)->jacobian_impl(point);
		}
	};
}