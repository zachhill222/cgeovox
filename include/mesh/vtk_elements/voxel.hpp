#pragma once

#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"
#include "mesh/vtk_elements/base.hpp"

#include <vector>
#include <cassert>

namespace gv::mesh {
	/* Voxel element vertex labels
	
	 			2 ------- 3
				|\	      |\
				| \		  | \
				0 -\----- 1  \
				 \	\      \  \
				  \	 6 ------- 7
				   \ |		 \ |
					\|        \|
					 4 ------- 5
	*/

	/////////////////////////////////////////////////
	/// Voxel element
	/// Note that this is not an iso-parametric element.
	///
	/// The reference element for a VTK_QUAD is [-1,1]x[-1,1]x[-1,1].
	/// The basis/shape function located at each vertex is a product of the coresponding linear basis functions on the connecting edges.
	/// For example, the basis function located at the reference vertex (-1,+1,+1) is 0.5(1-e0)*0.5(1+e1)*0.5(1+e2) where e0, e1, and e2
	/// are the cartesian coordinates in the reference element.
	///
	/// Let N0,...,N7 be the shape functions at each vertex. Then the mapping from the reference element to the actual element is given by
	///
	/// F(e0,e1) = v0*N0 + v1*N1 + ... + v7*N7
	///
	/// Where v0,...,v7 are the vertices of the actual/mesh element in R3.
	/// Note e0=e1=e2=0 at the center of the reference element so that N0=N1=N2=N3=0.125 at the center. Mapping into the mesh element gives us the center
	/// 0.125*(v0+...+v7), which cannot be simplified to the average of any two opposite vertices.
	/// Note by a similar line of reasoning, the center of each face must be found by averaging all four vertices.
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	struct VTK_VOXEL : public VTK_ELEMENT<Mesh_t, VTK_VOXEL<Mesh_t>, VOXEL_VTK_ID>
	{
		//define types
		using BASE = VTK_ELEMENT<Mesh_t, VTK_VOXEL<Mesh_t>, VOXEL_VTK_ID>;
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

			assert(BASE::N_VERTICES==8);
			using T = Scalar_t;

			//edge midpoints
			this->child_vertex_coords[ 8] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1]}); //8  - back face
			this->child_vertex_coords[ 9] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[3]}); //9  - back face
			this->child_vertex_coords[10] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[3]}); //10 - back face
			this->child_vertex_coords[11] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[2]}); //11 - back face

			this->child_vertex_coords[12] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[4]}); //12 - connecting edge
			this->child_vertex_coords[13] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[6]}); //13 - connecting edge
			this->child_vertex_coords[14] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[3],this->vertex_coords[7]}); //14 - connecting edge
			this->child_vertex_coords[15] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[5]}); //15 - connecting edge
			
			this->child_vertex_coords[16] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[5]}); //16 - front face
			this->child_vertex_coords[17] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[5],this->vertex_coords[7]}); //17 - front face
			this->child_vertex_coords[18] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[6],this->vertex_coords[7]}); //18 - front face
			this->child_vertex_coords[19] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[6]}); //19 - front face

			//face midpoints
			this->child_vertex_coords[20] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[6]}); //20 - left face
			this->child_vertex_coords[21] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[7]}); //21 - right face
			this->child_vertex_coords[22] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[7]}); //22 - top face
			this->child_vertex_coords[23] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[5]}); //23 - bottom face
			this->child_vertex_coords[24] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[3]}); //24 - back face
			this->child_vertex_coords[25] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[7]}); //25 - front face

			//center
			this->child_vertex_coords[26] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[7]}); //26
		}

		std::array<size_t,8> get_child_local_vertices_impl(const int child_number) const
		{
			std::array<size_t,8> child_vertices;
			switch (child_number) {
				case (0): //voxel element containing original vertex 0
					child_vertices[0] =  0; //0
					child_vertices[1] =  8; //0-1
					child_vertices[2] = 11; //0-2
					child_vertices[3] = 24; //0-3
					child_vertices[4] = 12; //0-4
					child_vertices[5] = 23; //0-5
					child_vertices[6] = 20; //0-6
					child_vertices[7] = 26; //0-7
					break;

				case (1): //voxel element containing original vertex 1
					child_vertices[0] =  8; //0-1
					child_vertices[1] =  1; //1
					child_vertices[2] = 24; //0-3
					child_vertices[3] =  9; //1-3
					child_vertices[4] = 23; //0-5
					child_vertices[5] = 15; //1-5
					child_vertices[6] = 26; //0-7
					child_vertices[7] = 21; //1-7
					break;

				case (2): //voxel element containing original vertex 2
					child_vertices[0] = 11; //0-2
					child_vertices[1] = 24; //0-3
					child_vertices[2] =  2; //2
					child_vertices[3] = 10; //2-3
					child_vertices[4] = 20; //0-6
					child_vertices[5] = 26; //0-7
					child_vertices[6] = 13; //2-6
					child_vertices[7] = 22; //2-7
					break;

				case (3): //voxel element containing original vertex 3
					child_vertices[0] = 24; //0-3
					child_vertices[1] =  9; //1-3
					child_vertices[2] = 10; //2-3
					child_vertices[3] =  3; //3
					child_vertices[4] = 26; //0-7
					child_vertices[5] = 21; //1-7
					child_vertices[6] = 22; //2-7
					child_vertices[7] = 14; //3-7
					break;

				case (4): //voxel element containing original vertex 4
					child_vertices[0] = 12; //0-4
					child_vertices[1] = 23; //0-5
					child_vertices[2] = 20; //0-6
					child_vertices[3] = 26; //0-7
					child_vertices[4] =  4; //4
					child_vertices[5] = 16; //4-5
					child_vertices[6] = 19; //4-6
					child_vertices[7] = 25; //4-7
					break;

				case (5): //voxel element containing original vertex 5
					child_vertices[0] = 23; //0-5
					child_vertices[1] = 15; //1-5
					child_vertices[2] = 26; //0-7
					child_vertices[3] = 21; //1-7
					child_vertices[4] = 16; //4-5
					child_vertices[5] =  5; //5
					child_vertices[6] = 25; //4-7
					child_vertices[7] = 17; //5-7
					break;

				case (6): //voxel element containing original vertex 6
					child_vertices[0] = 20; //0-6
					child_vertices[1] = 26; //0-7
					child_vertices[2] = 13; //2-6
					child_vertices[3] = 22; //2-7
					child_vertices[4] = 19; //4-6
					child_vertices[5] = 25; //4-7
					child_vertices[6] =  6; //6
					child_vertices[7] = 18; //6-7
					break;

				case (7): //voxel element containing original vertex 7
					child_vertices[0] = 26; //0-7
					child_vertices[1] = 21; //1-7
					child_vertices[2] = 22; //2-7
					child_vertices[3] = 14; //3-7
					child_vertices[4] = 25; //4-7
					child_vertices[5] = 17; //5-7
					child_vertices[6] = 18; //6-7
					child_vertices[7] =  7; //7
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}

			return child_vertices;
		}

		std::array<size_t,4> get_face_vertices_impl(const int face_number) const {
			std::array<size_t,4> face_vertices;
			switch (face_number) {
			case (0): //left
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[4];
				face_vertices[2] = this->vertices[2];
				face_vertices[3] = this->vertices[6];
				break;
			case (1): //right
				face_vertices[0] = this->vertices[1];
				face_vertices[1] = this->vertices[3];
				face_vertices[2] = this->vertices[5];
				face_vertices[3] = this->vertices[7];
				break;
			case (2): //top
				face_vertices[0] = this->vertices[2];
				face_vertices[1] = this->vertices[6];
				face_vertices[2] = this->vertices[3];
				face_vertices[3] = this->vertices[7];
				break;
			case (3): //bottom
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[1];
				face_vertices[2] = this->vertices[4];
				face_vertices[3] = this->vertices[5];
				break;
			case (4): //back
				face_vertices[0] = this->vertices[1];
				face_vertices[1] = this->vertices[0];
				face_vertices[2] = this->vertices[3];
				face_vertices[3] = this->vertices[2];
				break;
			case (5): //front
				face_vertices[0] = this->vertices[4];
				face_vertices[1] = this->vertices[5];
				face_vertices[2] = this->vertices[6];
				face_vertices[3] = this->vertices[7];
				break;
			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}

			return face_vertices;
		}

		std::array<size_t,9> get_face_child_local_vertices_impl(const int face_number) const
		{
			std::array<size_t,9> split_face_vertices;
			switch (face_number) {
				case (0): // Left face [0, 4, 2, 6]
				split_face_vertices[0] =  0; //0
				split_face_vertices[1] =  4; //4
				split_face_vertices[2] =  2; //2
				split_face_vertices[3] =  6; //6
				split_face_vertices[4] = 12; //0-4
				split_face_vertices[5] = 19; //4-6
				split_face_vertices[6] = 13; //2-6
				split_face_vertices[7] = 11; //0-2
				split_face_vertices[8] = 20; //left face center
				break;

			case (1): // Right face [1, 3, 5, 7]
				split_face_vertices[0] =  1; //1
				split_face_vertices[1] =  3; //3
				split_face_vertices[2] =  5; //5
				split_face_vertices[3] =  7; //7
				split_face_vertices[4] =  9; //1-3
				split_face_vertices[5] = 14; //3-7
				split_face_vertices[6] = 17; //5-7
				split_face_vertices[7] = 15; //1-5
				split_face_vertices[8] = 21; //right face center
				break;

			case (2): // Top face [2, 6, 3, 7]
				split_face_vertices[0] =  2; //2
				split_face_vertices[1] =  6; //6
				split_face_vertices[2] =  3; //3
				split_face_vertices[3] =  7; //7
				split_face_vertices[4] = 13; //2-6
				split_face_vertices[5] = 18; //6-7
				split_face_vertices[6] = 14; //3-7
				split_face_vertices[7] = 10; //2-3
				split_face_vertices[8] = 22; //top face center
				break;

			case (3): // Bottom face [0, 1, 4, 5]
				split_face_vertices[0] =  0; //0
				split_face_vertices[1] =  1; //1
				split_face_vertices[2] =  4; //4
				split_face_vertices[3] =  5; //5
				split_face_vertices[4] =  8; //0-1
				split_face_vertices[5] = 15; //1-5
				split_face_vertices[6] = 16; //4-5
				split_face_vertices[7] = 12; //0-4
				split_face_vertices[8] = 23; //bottom face center
				break;

			case (4): // Back face [1, 0, 3, 2]
				split_face_vertices[0] =  1; //1
				split_face_vertices[1] =  0; //0
				split_face_vertices[2] =  3; //3
				split_face_vertices[3] =  2; //2
				split_face_vertices[4] =  8; //1-0
				split_face_vertices[5] = 11; //0-2
				split_face_vertices[6] = 10; //2-3
				split_face_vertices[7] =  9; //3-1
				split_face_vertices[8] = 24; //back face center
				break;

			case (5): // Front face [4, 5, 6, 7]
				split_face_vertices[0] =  4; //4
				split_face_vertices[1] =  5; //5
				split_face_vertices[2] =  6; //6
				split_face_vertices[3] =  7; //7
				split_face_vertices[4] = 16; //4-5
				split_face_vertices[5] = 17; //5-7
				split_face_vertices[6] = 18; //6-7
				split_face_vertices[7] = 19; //4-6
				split_face_vertices[8] = 25; //front face center
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}

			return split_face_vertices;
		}

		//determine if a point in space is interior to the element
		bool contains_impl(const GeoPoint_t& point) const
		{
			return (this->vertex_coords[0] <= point) and (point <= this->vertex_coords[7]);
		}
		
		//evaluate the geometric mapping from the reference element to the actual element
		GeoPoint_t ref2geo_impl(const RefPoint_t& point) const
		{
			//GeoPoint may only work at a very large or very small scale.
			//do the arithmetic in the reference (double) type
			const auto center = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[7] + this->vertex_coords[0]));
			const auto H      = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[7] - this->vertex_coords[0]));
			return static_cast<GeoPoint_t>(center + H*point);
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		RefPoint_t geo2ref_impl(const GeoPoint_t& point) const
		{
			assert(contains_impl(point));
			//GeoPoint may only work at a very large or very small scale.
			//do the arithmetic in the reference (double) type
			const auto center = Scalar_t{0.5} * (this->vertex_coords[7] + this->vertex_coords[0]);
			const auto H      = static_cast<RefPoint_t>(Scalar_t{0.5} * (this->vertex_coords[7] - this->vertex_coords[0]));
			
			return static_cast<RefPoint_t>(point - center) / static_cast<RefPoint_t>(H);
		}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		Jac_t jacobian_impl(const RefPoint_t& ref_coord) const
		{
			Jac_t jacobian{};
			GeoPoint_t H = this->vertex_coords[7]-this->vertex_coords[0];
			for (int i=0; i<3; i++) {
				jacobian(i,i) = 0.5*static_cast<double>(H[i]);
			}

			return jacobian;
		}
	};
}