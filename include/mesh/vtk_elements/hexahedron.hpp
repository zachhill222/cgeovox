#pragma once

#include "gutil.hpp"

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"
#include "mesh/vtk_elements/base.hpp"

#include <vector>
#include <cassert>

namespace gv::mesh {
	/* Hexahedron element vertices labels
	
	 			3 ------- 2
				|\	      |\
				| \		  | \
				0 -\----- 1  \
				 \	\      \  \
				  \	 7 ------- 6
				   \ |		 \ |
					\|        \|
					 4 ------- 5
	*/

	/////////////////////////////////////////////////
	/// Hexahedral element
	/// Note that this is an iso-parametric element. The edge-midpoints can be found via an average of the endpoints,
	/// but the center of the element must be found by evaluating the shape functions.
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
	template<typename Mesh_t>
	struct VTK_HEXAHEDRON : public VTK_ELEMENT<Mesh_t, VTK_HEXAHEDRON<Mesh_t>, HEXAHEDRON_VTK_ID>
	{
		//define types
		using BASE = VTK_ELEMENT<Mesh_t, VTK_HEXAHEDRON<Mesh_t>, HEXAHEDRON_VTK_ID>;
		using typename BASE::GeoPoint_t;
		using typename BASE::Scalar_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		using BASE::BASE;
		
		//coordinates for the reference element. store in row-major to pull out rows easier.
		static constexpr double REF_COORDS[8][3] {
			{-1, -1, -1},
			{ 1, -1, -1},
			{ 1,  1, -1},
			{-1,  1, -1},
			{-1, -1,  1},
			{ 1, -1,  1},
			{ 1,  1,  1},
			{-1,  1,  1},
		};

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
			this->child_vertex_coords[ 9] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[2]}); //9  - back face
			this->child_vertex_coords[10] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[3]}); //10 - back face
			this->child_vertex_coords[11] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[3]}); //11 - back face

			this->child_vertex_coords[12] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[4]}); //12 - connecting edge
			this->child_vertex_coords[13] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[3],this->vertex_coords[7]}); //13 - connecting edge
			this->child_vertex_coords[14] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[6]}); //14 - connecting edge
			this->child_vertex_coords[15] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[5]}); //15 - connecting edge
			
			this->child_vertex_coords[16] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[5]}); //16 - front face
			this->child_vertex_coords[17] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[5],this->vertex_coords[6]}); //17 - front face
			this->child_vertex_coords[18] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[6],this->vertex_coords[7]}); //18 - front face
			this->child_vertex_coords[19] = T{0.5}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[7]}); //19 - front face

			//face midpoints
			this->child_vertex_coords[20] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[3],this->vertex_coords[4],this->vertex_coords[7]}); //20 - left face  (L)
			this->child_vertex_coords[21] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[1],this->vertex_coords[2],this->vertex_coords[5],this->vertex_coords[6]}); //21 - right face (R)
			this->child_vertex_coords[22] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[2],this->vertex_coords[3],this->vertex_coords[6],this->vertex_coords[7]}); //22 - up face    (U)
			this->child_vertex_coords[23] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1],this->vertex_coords[4],this->vertex_coords[5]}); //23 - down face  (D)
			this->child_vertex_coords[24] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],this->vertex_coords[1],this->vertex_coords[2],this->vertex_coords[3]}); //24 - back face  (B)
			this->child_vertex_coords[25] = T{0.25}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[4],this->vertex_coords[5],this->vertex_coords[6],this->vertex_coords[7]}); //25 - front face (F)

			//center
			this->child_vertex_coords[26] = T{0.125}*gutil::sorted_sum<3,T,T,T>({this->vertex_coords[0],
																				this->vertex_coords[1],
																				this->vertex_coords[2],
																				this->vertex_coords[3],
																				this->vertex_coords[4],
																				this->vertex_coords[5],
																				this->vertex_coords[6],
																				this->vertex_coords[7]}
																				); //26
		}

		std::array<size_t,8> get_child_local_vertices_impl(const int child_number) const
		{
			std::array<size_t,8> child_vertices;

			switch (child_number) {
				case (0): //hex element containing original vertex 0
					child_vertices[0] =  0; //0
					child_vertices[1] =  8; //0-1
					child_vertices[2] = 24; //B
					child_vertices[3] = 11; //0-3
					child_vertices[4] = 12; //0-4
					child_vertices[5] = 23; //D
					child_vertices[6] = 26; //C
					child_vertices[7] = 20; //L
					break;

				case (1): //hex element containing original vertex 1
					child_vertices[0] =  8; //0-1
					child_vertices[1] =  1; //1
					child_vertices[2] =  9; //1-2
					child_vertices[3] = 24; //B
					child_vertices[4] = 23; //D
					child_vertices[5] = 15; //1-5
					child_vertices[6] = 21; //R
					child_vertices[7] = 26; //C
					break;

				case (2): //hex element containing original vertex 2
					child_vertices[0] = 24; //B
					child_vertices[1] =  9; //1-2
					child_vertices[2] =  2; //2
					child_vertices[3] = 10; //2-3
					child_vertices[4] = 26; //C
					child_vertices[5] = 21; //R
					child_vertices[6] = 14; //2-6
					child_vertices[7] = 22; //U
					break;

				case (3): //hex element containing original vertex 3
					child_vertices[0] = 11; //0-3
					child_vertices[1] = 24; //B
					child_vertices[2] = 10; //2-3
					child_vertices[3] =  3; //3
					child_vertices[4] = 20; //L
					child_vertices[5] = 26; //C
					child_vertices[6] = 22; //U
					child_vertices[7] = 13; //3-7
					break;

				case (4): //hex element containing original vertex 4
					child_vertices[0] = 12; //0-4
					child_vertices[1] = 23; //D
					child_vertices[2] = 26; //C
					child_vertices[3] = 20; //L
					child_vertices[4] =  4; //4
					child_vertices[5] = 16; //4-5
					child_vertices[6] = 25; //F
					child_vertices[7] = 19; //4-7
					break;

				case (5): //hex element containing original vertex 5
					child_vertices[0] = 23; //D
					child_vertices[1] = 15; //1-5
					child_vertices[2] = 21; //R
					child_vertices[3] = 26; //C
					child_vertices[4] = 16; //4-5
					child_vertices[5] =  5; //5
					child_vertices[6] = 17; //5-6
					child_vertices[7] = 25; //F
					break;

				case (6): //hex element containing original vertex 6
					child_vertices[0] = 26; //C
					child_vertices[1] = 21; //R
					child_vertices[2] = 14; //2-6
					child_vertices[3] = 22; //U
					child_vertices[4] = 25; //F
					child_vertices[5] = 17; //5-6
					child_vertices[6] =  6; //6
					child_vertices[7] = 18; //6-7
					break;

				case (7): //hex element containing original vertex 7
					child_vertices[0] = 20; //L
					child_vertices[1] = 26; //C
					child_vertices[2] = 22; //U
					child_vertices[3] = 13; //3-7
					child_vertices[4] = 19; //4-7
					child_vertices[5] = 25; //F
					child_vertices[6] = 18; //6-7
					child_vertices[7] =  7; //7
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}

			return child_vertices;
		}

		std::array<size_t,4> get_face_vertices_impl(const int face_number) const
		{
			std::array<size_t,4> face_vertices;
			switch (face_number) {
			case (0): //left
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[4];
				face_vertices[2] = this->vertices[7];
				face_vertices[3] = this->vertices[3];
				break;
			case (1): //right
				face_vertices[0] = this->vertices[1];
				face_vertices[1] = this->vertices[2];
				face_vertices[2] = this->vertices[6];
				face_vertices[3] = this->vertices[5];
				break;
			case (2): //top
				face_vertices[0] = this->vertices[2];
				face_vertices[1] = this->vertices[3];
				face_vertices[2] = this->vertices[7];
				face_vertices[3] = this->vertices[6];
				break;
			case (3): //bottom
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[1];
				face_vertices[2] = this->vertices[5];
				face_vertices[3] = this->vertices[4];
				break;
			case (4): //back
				face_vertices[0] = this->vertices[0];
				face_vertices[1] = this->vertices[3];
				face_vertices[2] = this->vertices[2];
				face_vertices[3] = this->vertices[1];
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
			case (0) : //Left face [0, 4, 7, 3]
				split_face_vertices[0] =  0; //0
				split_face_vertices[1] =  4; //4
				split_face_vertices[2] =  7; //7
				split_face_vertices[3] =  3; //3
				split_face_vertices[4] = 12; //0-4
				split_face_vertices[5] = 19; //4-7
				split_face_vertices[6] = 13; //7-3
				split_face_vertices[7] = 11; //3-0
				split_face_vertices[8] = 20; //L
				break;
			case (1): // Right face [1,2,6,5]
				split_face_vertices[0] =  1; //1
				split_face_vertices[1] =  2; //2
				split_face_vertices[2] =  6; //6
				split_face_vertices[3] =  5; //5
				split_face_vertices[4] =  9; //1-2
				split_face_vertices[5] = 14; //2-6
				split_face_vertices[6] = 17; //6-5
				split_face_vertices[7] = 15; //5-1
				split_face_vertices[8] = 21; //R
				break;
		        
		    case (2): // Top face [2,3,7,6]
				split_face_vertices[0] =  2; //2
				split_face_vertices[1] =  3; //3
				split_face_vertices[2] =  7; //7
				split_face_vertices[3] =  6; //6
				split_face_vertices[4] = 10; //2-3
				split_face_vertices[5] = 13; //3-7
				split_face_vertices[6] = 18; //7-6
				split_face_vertices[7] = 14; //6-2
				split_face_vertices[8] = 22; //U
				break;
		        
		    case (3): // Bottom face [0,1,5,4]
				split_face_vertices[0] =  0; //0
				split_face_vertices[1] =  1; //1
				split_face_vertices[2] =  5; //5
				split_face_vertices[3] =  4; //4
				split_face_vertices[4] =  8; //0-1
				split_face_vertices[5] = 15; //1-5
				split_face_vertices[6] = 16; //5-4
				split_face_vertices[7] = 12; //4-0
				split_face_vertices[8] = 23; //D
				break;
		        
		    case (4): // Back face [0,3,2,1]
				split_face_vertices[0] =  0; //0
				split_face_vertices[1] =  3; //3
				split_face_vertices[2] =  2; //2
				split_face_vertices[3] =  1; //1
				split_face_vertices[4] = 11; //0-3
				split_face_vertices[5] = 10; //3-2
				split_face_vertices[6] =  9; //2-1
				split_face_vertices[7] =  8; //1-0
				split_face_vertices[8] = 24; //B
				break;
		        
		    case (5): // Front face [4,5,6,7]
				split_face_vertices[0] =  4; //4
				split_face_vertices[1] =  5; //5
				split_face_vertices[2] =  6; //6
				split_face_vertices[3] =  7; //7
				split_face_vertices[4] = 16; //4-5
				split_face_vertices[5] = 17; //5-6
				split_face_vertices[6] = 18; //6-7
				split_face_vertices[7] = 19; //7-4
				split_face_vertices[8] = 25; //F
				break;
		        
		    default:
		        throw std::out_of_range("face number out of bounds");
		        break;
		    }

		    return split_face_vertices;
		}
		
		//evaluate the tri-linear shape function associated with vertex i on the reference element
		//this is used for iso-parametric mapping
		inline constexpr double eval_local_geo_shape_fun(const int i, const RefPoint_t& ref_coord) const noexcept
		{
			assert(0<= i and i<BASE::N_VERTICES);
			return 0.125*(1.0+REF_COORDS[i][0]*ref_coord[0])*(1.0+REF_COORDS[i][1]*ref_coord[1])*(1.0+REF_COORDS[i][2]*ref_coord[2]);
		}

		inline constexpr RefPoint_t eval_local_geo_shape_grad(const int i, const RefPoint_t& ref_coord) const noexcept
		{
			assert(0<=i and i<BASE::N_VERTICES);
			RefPoint_t result{};

			result[0] = 0.125 * REF_COORDS[i][0]                               * (1.0+REF_COORDS[i][1]*ref_coord[1]) * (1.0+REF_COORDS[i][2]*ref_coord[2]);
			result[1] = 0.125 * (1.0+REF_COORDS[i][0]*ref_coord[0]) * REF_COORDS[i][1]                               * (1.0+REF_COORDS[i][2]*ref_coord[2]);
			result[2] = 0.125 * (1.0+REF_COORDS[i][0]*ref_coord[0]) * (1.0+REF_COORDS[i][1]*ref_coord[1]) * REF_COORDS[i][2];
			return result;
		}

		
		//evaluate the geometric mapping from the reference element to the actual element
		GeoPoint_t ref2geo_impl(const RefPoint_t& ref_coord) const
		{
			GeoPoint_t result{}; //zero
			for (int i=0; i<BASE::N_VERTICES; i++) {
				result += eval_local_geo_shape_fun(i,ref_coord) * this->vertex_coords[i];
			}
			return result;
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		RefPoint_t geo2ref_impl(const GeoPoint_t& coord) const {return RefPoint_t{};}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		Jac_t jacobian_impl(const RefPoint_t& ref_coord) const {return Jac_t{};};

		//determine if a point in space is interior to the element
		bool contains_impl(const GeoPoint_t& coord) const
		{
			assert(false);
			return false;
		}
	};
}