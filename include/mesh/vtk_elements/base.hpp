#pragma once

// #include "util/point.hpp"
// #include "util/box.hpp"
// #include "util/matrix.hpp"
#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "concepts.hpp"

#include <vector>
#include <cassert>

namespace gv::mesh {
	/////////////////////////////////////////////////
	/// Interface for VTK element types
	///
	/// @tparam space_dim       The dimension of the space that the geometric elmenent is embedded in.
	/// @tparam ref_dim         The dimension of the space that the reference element is embedded in.
	/// @tparam VertexScalar_t  The type that is emulating the real line that the mesh uses. This should 
	///                              be robust for comparisons (e.g., uses exact or fixed precision arithmetic rather than floating point).
	/// @tparam MapScalar_t     The type that is emulating the real line that the FEM computations use. This should
	///                              be accurate and fast (e.g., use floating point rather than fixed precision)
	/////////////////////////////////////////////////
	template<int space_dim, int ref_dim, Scalar VertexScalar_t, Scalar MapScalar_t>
	struct VTK_ELEMENT {
		static_assert(0<ref_dim and ref_dim<=3,     "VTK_ELEMENT: the reference element must be in dimensions 1, 2, or 3.");
		static_assert(0<space_dim and space_dim<=3, "VTK_ELEMENT: the geometric element must be in dimensions 1, 2, or 3.");
		static_assert(ref_dim<=space_dim,           "VTK_ELEMENT: you cannot have the dimension of the refernece element larger than the geometric element.");

		VTK_ELEMENT(const BasicElement &elem) : ELEM(elem) {}
		virtual ~VTK_ELEMENT() {}
		
		using Point_t    = gutil::Point<space_dim, VertexScalar_t>;    //type of point in space (e.g., mesh vertex coordinates)
		using RefPoint_t = gutil::Point<ref_dim, MapScalar_t>;   //type of point in the reference domain
		using Jac_t      = gutil::Matrix<space_dim, ref_dim, MapScalar_t, false>; //type of jacobian matrix

		const BasicElement &ELEM;
		virtual void split(std::vector<Point_t>& vertex_coords) const = 0;
		virtual void getChildVertices(std::vector<size_t>& child_nodes, const int child_number, const std::vector<size_t>& split_node_numbers) const = 0;
		virtual void getFaceVertices(std::vector<size_t>& face_nodes, const int face_number) const = 0;
		virtual void getSplitFaceVertices(std::vector<size_t>& split_face_nodes, const int face_number, const std::vector<size_t>& split_node_numbers) const = 0;
		BasicElement getFace(const int face_number) const {
			BasicElement face(vtk_face_id(this->ELEM.vtkID));
			getFaceVertices(face.vertices, face_number);
			return face;
		}

		//evaluate the local shape functions that are used to map the reference element to the actual element
		virtual inline constexpr MapScalar_t eval_local_geo_shape_fun(const int i, const RefPoint_t& ref_coord) const noexcept = 0;

		//evaluate the gradient of the shape functions that are used to map the referent to the actual element
		virtual inline constexpr RefPoint_t  eval_local_geo_shape_grad(const int i, const RefPoint_t& ref_coord) const noexcept = 0;

		
		//evaluate the geometric mapping from the reference element to the actual element
		virtual constexpr Point_t reference_to_geometric(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept = 0;

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		virtual constexpr RefPoint_t geometric_to_reference(const std::vector<Point_t>& vertex_coords, const Point_t& coord) const noexcept = 0;

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		virtual constexpr Jac_t   eval_geo_shape_jac(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept = 0;
	};
}