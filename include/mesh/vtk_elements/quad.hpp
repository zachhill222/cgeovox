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
	/* Quad element vertex labels
	
	 			3 ----- 2
				|		|
				|		|
				0 ----- 1
	*/

	/////////////////////////////////////////////////
	/// Quad element
	/// Note that this is an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// but the center of the element must be found by evaluating the shape functions.
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1].
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1) is 0.5(1-e0)*0.5(1+e1) where e0 and e1 are the cartesian coordinates
	/// in the reference element.
	///
	/// Let N0,...N3 be the shape functions at each vertex. Then the mapping from the reference element to the actual element is given by
	///
	/// F(e0,e1) = v0*N0 + v1*N1 + v2*N2 + v3*N3
	///
	/// Where v0,...,v3 are the vertices of the actual/mesh element in R3.
	/// Note e0=e1=0 at the center of the reference element so that N0=N1=N2=N3=0.25 at the center. Mapping into the mesh element gives us the center
	/// 0.25*(v0+v1+v2+v3), which cannot be simplified to the average of any two opposite vertices (v0 may be moved nearly arbitrarily, which changes the
	/// face center but does not change the evaluation of 0.5*(v1+v3).)
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	struct VTK_QUAD : public VTK_ELEMENT<Mesh_t, VTK_QUAD<Mesh_t>, QUAD_VTK_ID> {
		//define types
		using BASE = VTK_ELEMENT<Mesh_t, VTK_QUAD<Mesh_t>, QUAD_VTK_ID>;
		using typename BASE::GeoPoint_t;
		using typename BASE::Scalar_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		using BASE::BASE;

		//coordinates for the reference element. store in row-major to pull out rows easier.
		static constexpr double REF_COORDS[4][2] {
			{-1, -1},
			{ 1, -1},
			{-1,  1},
			{ 1,  1}
		};

		void set_child_vertices_impl()
		{
			//copy over parent vertices
			for (int i=0; i<BASE::N_VERTICES; ++i) {
				this->child_vertex_coords[i] = this->vertex_coords[i];
			}
			
			assert(BASE::N_VERTICES==4);
			using T = Scalar_t;
			
			//edge midpoints
			this->child_vertex_coords[4] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1]}); //4 - bottom (B)
			this->child_vertex_coords[5] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[2]}); //5 - right (R)
			this->child_vertex_coords[6] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[3]}); //6 - top (T)
			this->child_vertex_coords[7] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[3]}); //7 - left (L)

			//center
			this->child_vertex_coords[8] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1],this->vertex_coords[2],this->vertex_coords[3]}); //8 (C)
		}

		std::array<size_t,4> get_child_local_vertices_impl(const int child_number) const
		{
			std::array<size_t,4> child_vertices;

			switch (child_number) {
				case (0):
					child_vertices[0] = 0; //0
					child_vertices[1] = 4; //B
					child_vertices[2] = 8; //C
					child_vertices[3] = 7; //L
					break;
				case (1):
					child_vertices[0] = 4; //B
					child_vertices[1] = 1; //1
					child_vertices[2] = 5; //R
					child_vertices[3] = 8; //C
					break;
				case (2):
					child_vertices[0] = 8; //C
					child_vertices[1] = 5; //R
					child_vertices[2] = 2; //2
					child_vertices[3] = 6; //T
					break;
				case (3):
					child_vertices[0] = 7; //L
					child_vertices[1] = 8; //C
					child_vertices[2] = 6; //T
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
				face_vertices[1] = this->vertices[2];
				break;
			case (2):
				face_vertices[0] = this->vertices[2];
				face_vertices[1] = this->vertices[3];
				break;
			case (3):
				face_vertices[0] = this->vertices[3];
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

			case (1): // Right [1, 2]
				split_face_vertices[0] = 1; //1
				split_face_vertices[1] = 2; //2
				split_face_vertices[2] = 5; //1-2
				break;

			case (2): // Top [2, 3]
				split_face_vertices[0] = 2; //2
				split_face_vertices[1] = 3; //3
				split_face_vertices[2] = 6; //2-3
				break;

			case (3): // Left [3, 0]
				split_face_vertices[0] = 3; //3
				split_face_vertices[1] = 0; //0
				split_face_vertices[2] = 7; //0-3
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}

			return split_face_vertices;
		}


		//evaluate the bi-linear shape function associated with vertex i on the reference element
		inline constexpr double eval_local_geo_shape_fun(const int i, const RefPoint_t& point) const noexcept
		{
			assert(0<= i and i<BASE::N_VERTICES);
			return 0.25*(1.0+REF_COORDS[i][0]*point[0])*(1.0+REF_COORDS[i][1]*point[1]);
		}

		inline constexpr RefPoint_t  eval_local_geo_shape_grad(const int i, const RefPoint_t& point) const noexcept
		{
			assert(0<= i and i<BASE::N_VERTICES);
			
			RefPoint_t result{};
			result[0] = 0.25 * REF_COORDS[i][0]                               * (1.0+REF_COORDS[i][1]*point[1]);
			result[1] = 0.25 * (1.0+REF_COORDS[i][0]*point[0]) * REF_COORDS[i][1];
			return result;
		}

		
		//evaluate the geometric mapping from the reference element to the actual element
		GeoPoint_t ref2geo_impl(const RefPoint_t& point) const
		{
			GeoPoint_t result{}; //zero
			for (int i=0; i<BASE::N_VERTICES; i++) {
				result += eval_local_geo_shape_fun(i,point) * this->vertex_coords[i];
			}
			return result;
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		RefPoint_t geo2ref_impl(const GeoPoint_t& point) const {return RefPoint_t{};}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		Jac_t  jacobian_impl(const RefPoint_t& point) const {return Jac_t{};};

		//determine if a point in space is interior to the element
		bool contains_impl(const GeoPoint_t& point) const
		{
			assert(false);
			return true;
		}
	};
}