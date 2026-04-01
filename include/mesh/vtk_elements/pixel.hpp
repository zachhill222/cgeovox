#pragma once

// #include "util/point.hpp"
// #include "util/box.hpp"
// #include "util/matrix.hpp"
#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>
#include <functional>


namespace gv::mesh {
	/* Pixel element vertex labels
	
	 			2 ----- 3
				|		|
				|		|
				0 ----- 1
	*/

	/////////////////////////////////////////////////
	/// Pixel element
	/// Note that this is not an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// and the center of the element can be found by averaging opposite vertices. Pixel elements must always be a rectangular
	/// subset of a plane that is parallel to a coordinate axis with sides parallel to the remaining two coordinate axes (i.e., faces 
	/// of an axis-aligned bounding box).
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1].
	/// 
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1) is T{0.5}(1-e0)*0.5(1+e1) where e0 and e1 are the cartesian coordinates
	/// in the reference element.
	///
	/// The mapping from the reference element to the actual/mesh element is of the form:
	///
	/// F(e0,e1) = A*[e0, e1]^t + b
	/// 
	/// where A is a 3x2 matrix and b is the location of the center of the mesh element. Because the the map is affine, the Jacobian matrix (J=A) is constant.
	/// In fact sqrt(J^t J) = 0.25*h1*h2 where h1 and h2 are the side-lengths of the mesh element.
	/////////////////////////////////////////////////
	template<typename Mesh_t>
	struct VTK_PIXEL : public VTK_ELEMENT<Mesh_t, VTK_PIXEL<Mesh_t>, PIXEL_VTK_ID> {
		//define types
		using BASE = VTK_ELEMENT<Mesh_t, VTK_PIXEL<Mesh_t>, PIXEL_VTK_ID>;
		using typename BASE::GeoPoint_t;
		using typename BASE::Scalar_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		using BASE::BASE;

		void set_child_vertices_impl()
		{
			//copy over the parent vertices
			for (int i=0; i<BASE::N_VERTICES; ++i) {
				this->child_vertex_coords[i] = this->vertex_coords[i];
			}

			assert(BASE::N_VERTICES==4);
			using T = Scalar_t;

			//edge midpoints
			this->child_vertex_coords[4] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1]}); //4 - bottom
			this->child_vertex_coords[5] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[3]}); //5 - right
			this->child_vertex_coords[6] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[3]}); //6 - top
			this->child_vertex_coords[7] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[2]}); //7 - left

			//center
			this->child_vertex_coords[8] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[3]}); //8 - center
		}

		std::array<size_t,4> get_child_local_vertices_impl(const int child_number) const
		{
			std::array<size_t,4> child_vertices;
			switch (child_number) {
				case (0):
					child_vertices[0] = 0; //0
					child_vertices[1] = 4; //0-1
					child_vertices[2] = 7; //0-2
					child_vertices[3] = 8; //0-3
					break;
				case (1):
					child_vertices[0] = 4; //0-1
					child_vertices[1] = 1; //1
					child_vertices[2] = 8; //0-3
					child_vertices[3] = 5; //1-3
					break;
				case (2):
					child_vertices[0] = 7; //0-2
					child_vertices[1] = 8; //0-3
					child_vertices[2] = 2; //2
					child_vertices[3] = 6; //2-3
					break;
				case (3):
					child_vertices[0] = 8; //0-3
					child_vertices[1] = 5; //1-3
					child_vertices[2] = 6; //2-3
					child_vertices[3] = 3; //3
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}

			return child_vertices;
		}

		std::array<size_t,2> get_face_vertices_impl(const int face_number) const
		{
			std::array<size_t,2> face_vertices;
			switch (face_number) {
			case (0):
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[1];
				break;
			case (1):
				face_vertices[0] = this->vertices[1];
				face_vertices[1] = this->vertices[3];
				break;
			case (2):
				face_vertices[0] = this->vertices[3];
				face_vertices[1] = this->vertices[2];
				break;
			case (3):
				face_vertices[0] = this->vertices[2];
				face_vertices[1] = this->vertices[0];
				break;
			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}

			return face_vertices;
		}

		std::array<size_t,3> get_face_child_local_vertices_impl(const int face_number) const
		{
			std::array<size_t,3> split_face_vertices;
			switch (face_number) {
				case (0): // Bottom [0, 1]
				split_face_vertices[0] = 0; //0
				split_face_vertices[1] = 1; //1
				split_face_vertices[2] = 4; //0-1
				break;

			case (1): // Right [1, 3]
				split_face_vertices[0] = 1; //1
				split_face_vertices[1] = 3; //3
				split_face_vertices[2] = 5; //1-3
				break;

			case (2): // Top [3, 2]
				split_face_vertices[0] = 3; //3
				split_face_vertices[1] = 2; //2
				split_face_vertices[2] = 6; //2-3
				break;

			case (3): // Left [2, 0]
				split_face_vertices[0] = 2; //2
				split_face_vertices[1] = 0; //0
				split_face_vertices[2] = 7; //0-2
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}

			return split_face_vertices;
		}

		
		//evaluate the geometric mapping from the reference element to the actual element
		GeoPoint_t ref2geo_impl(const RefPoint_t& point) const
		{
			//GeoPoint may only work at a very large or very small scale.
			//do the arithmetic in the reference (double) type
			const auto center = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[3] + this->vertex_coords[0]));
			const auto H      = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[3] - this->vertex_coords[0]));
			return static_cast<GeoPoint_t>(center + H*point);
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		RefPoint_t geo2ref_impl(const GeoPoint_t& point) const
		{
			assert(contains_impl(point));
			//GeoPoint may only work at a very large or very small scale.
			//do the arithmetic in the reference (double) type
			const auto center = Scalar_t{0.5} * (this->vertex_coords[3] + this->vertex_coords[0]);
			const auto H      = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[3] - this->vertex_coords[0]));
			
			return static_cast<RefPoint_t>(point - center) / static_cast<RefPoint_t>(H);
		}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		Jac_t jacobian_impl(const RefPoint_t& point) const
		{
			//TODO: this may be wrong if the orientation doesn't have the z-coordinate at 0
			Jac_t jacobian{};
			GeoPoint_t H = this->vertex_coords[3]-this->vertex_coords[0];
			for (int i=0; i<2; i++) {
				jacobian(i,i) = 0.5*static_cast<double>(H[i]);
			}

			return jacobian;
		}

		//determine if a point in space is interior to the element
		bool contains_impl(const GeoPoint_t& point) const
		{
			return (this->vertex_coords[0] <= point) and (point <= this->vertex_coords[3]);
		}
	};
}