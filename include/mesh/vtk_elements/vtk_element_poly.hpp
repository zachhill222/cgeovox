#pragma once

#include "gutil.hpp"

#include "mesh/mesh_util.hpp"

#include "mesh/vtk_elements/base.hpp"
#include "mesh/vtk_elements/voxel.hpp"
#include "mesh/vtk_elements/hexahedron.hpp"
#include "mesh/vtk_elements/pixel.hpp"
#include "mesh/vtk_elements/quad.hpp"

#include "concepts.hpp"

#include <array>
#include <vector>
#include <cassert>
#include <variant>

namespace gv::mesh
{
	/////////////////////////////////////////////////////////////////////
	/// This is a polymorphic wrapper for VTK_ELEMENT that can be used when
	/// the element type is unknown at compile time.
	/////////////////////////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	struct VTK_ELEMENT_POLY
	{
		using GeoPoint_t = typename Mesh_t::Point_t;
		using Scalar_t   = typename GeoPoint_t::scalar_type;
		using RefPoint_t = typename Mesh_t::RefPoint_t;
		using Jac_t      = gutil::Matrix<GeoPoint_t::dim, RefPoint_t::dim, double, false>;

		//use variant to track all possible element types.
		//all supported element types must be listed here.
		using Variant_t  = std::variant<
			VTK_VOXEL<Mesh_t>,
			VTK_HEXAHEDRON<Mesh_t>,
			VTK_PIXEL<Mesh_t>,
			VTK_QUAD<Mesh_t>
		>; 
		Variant_t elem;

		//constructor chooses which element type this behaves as at runtime
		explicit VTK_ELEMENT_POLY(const int vtkID) {
			switch (vtkID) {
				case VOXEL_VTK_ID: elem = VTK_VOXEL<Mesh_t>{}; break;
				case HEXAHEDRON_VTK_ID: elem = VTK_HEXAHEDRON<Mesh_t>{}; break;
				case PIXEL_VTK_ID: elem = VTK_PIXEL<Mesh_t>{}; break;
				case QUAD_VTK_ID: elem = VTK_QUAD<Mesh_t>{}; break;
				default: throw std::runtime_error("Unknown vtkID: " + std::to_string(vtkID));
			}
		}

		//access arrays and data
		inline size_t last_element() const
		{
			return std::visit([](const auto& e){return e.last_element;}, elem);
		}

		inline size_t vertices(const int i) const
		{
			assert(0<= i and i < this->n_vertices());
			return std::visit([i](const auto& e){return e.vertices[i];}, elem);
		}

		inline GeoPoint_t vertex_coords(const int i)
		{
			assert(0<= i and i < this->n_vertices());
			return std::visit([i](auto& e){return e.vertex_coords[i];}, elem);
		}

		inline GeoPoint_t child_vertex_coords(const int i) const
		{
			assert(0<= i and i < this->n_vert_on_split());
			return std::visit([i](const auto& e){return e.child_vertex_coords[i];}, elem);
		}




		//implement VTK_ELEMENT interface by forwarding the argument to the correct variant
		inline int n_vertices() const
		{
			return std::visit([](const auto& e){return e.N_VERTICES;}, elem);
		}

		inline int n_faces() const
		{
			return std::visit([](const auto& e){return e.N_FACES;}, elem);
		}

		inline int face_vtk_id() const
		{
			return std::visit([](const auto& e){return e.FACE_VTK_ID;}, elem);
		}

		inline int n_children() const
		{
			return std::visit([](const auto& e){return e.N_CHILDREN;}, elem);
		}

		inline int n_vert_on_split() const
		{
			return std::visit([](const auto& e){return e.N_VERT_ON_SPLIT;}, elem);
		}

		inline void set_element(const Mesh_t& mesh, const size_t element_index)
		{
			std::visit([&mesh, element_index](auto& e){e.set_element(mesh, element_index);}, elem);
		}

		inline void set_child_vertices()
		{
			std::visit([](auto& e){e.set_child_vertices();}, elem);
		}

		inline std::vector<size_t> get_child_local_vertices(const int child_number) const
		{
			return std::visit([child_number](const auto& e) -> std::vector<size_t> {
				auto arr = e.get_child_local_vertices(child_number);
				return std::vector<size_t>(arr.begin(), arr.end());
			}, elem);	
		}

		inline std::vector<size_t> get_face_vertices(const int face_number) const
		{
			return std::visit([face_number](const auto& e) -> std::vector<size_t> {
				auto arr = e.get_face_vertices(face_number);
				return std::vector<size_t>(arr.begin(), arr.end());
			}, elem);
		}

		inline std::vector<size_t> get_face_child_local_vertices(const int face_number) const
		{
			return std::visit([face_number](const auto& e) -> std::vector<size_t> {
				auto arr = e.get_face_child_local_vertices(face_number);
				return std::vector<size_t>(arr.begin(), arr.end());
			}, elem);
			
		}

		inline bool contains(const GeoPoint_t& point) const
		{
			return std::visit([&point](const auto& e){return e.contains(point);}, elem);
		}

		inline RefPoint_t geo2ref(const GeoPoint_t& point) const
		{
			return std::visit([&point](const auto& e){return e.geo2ref(point);}, elem);
		}

		inline GeoPoint_t ref2geo(const RefPoint_t& point) const
		{
			return std::visit([&point](const auto& e){return e.ref2geo(point);}, elem);
		}

		inline Jac_t jacobian(const RefPoint_t& point) const
		{
			return std::visit([&point](const auto& e){return e.jacobian(point);}, elem);
		}
	};
}