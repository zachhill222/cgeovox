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
	template<Scalar VertexScalar_t, Scalar MapScalar_t>
	struct VTK_VOXEL : public VTK_ELEMENT<3,3,VertexScalar_t, MapScalar_t> {
		//define types
		using BASE = VTK_ELEMENT<3,3,VertexScalar_t,MapScalar_t>;
		using typename BASE::Point_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		VTK_VOXEL(const BasicElement &elem) : BASE(elem) {assert(elem.vtkID==VTK_ID); assert(elem.vertices.size()==vtk_n_vertices(elem.vtkID));}
		
		//vtk element type
		static constexpr int VTK_ID  = VOXEL_VTK_ID;
		static constexpr int N_VERTICES = vtk_n_vertices(VTK_ID);

		//coordinates for the reference element. store in row-major to pull out rows easier.
		static constexpr gutil::Matrix<8,3,MapScalar_t,false> REF_COORDS {
			{-1, -1, -1},
			{ 1, -1, -1},
			{-1,  1, -1},
			{ 1,  1, -1},
			{-1, -1,  1},
			{ 1, -1,  1},
			{-1,  1,  1},
			{ 1,  1,  1},
		};

		void split(std::vector<Point_t>& vertex_coords) const override {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));
			vertex_coords.reserve(vtk_n_vertices_when_split(VTK_ID));

			using T = VertexScalar_t;
			//edge midpoints
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[1]})); //8  - back face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[1],vertex_coords[3]})); //9  - back face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[2],vertex_coords[3]})); //10 - back face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[2]})); //11 - back face

			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[4]})); //12 - connecting edge
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[2],vertex_coords[6]})); //13 - connecting edge
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[3],vertex_coords[7]})); //14 - connecting edge
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[1],vertex_coords[5]})); //15 - connecting edge
			
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[4],vertex_coords[5]})); //16 - front face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[5],vertex_coords[7]})); //17 - front face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[6],vertex_coords[7]})); //18 - front face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[4],vertex_coords[6]})); //19 - front face

			//face midpoints
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[6]})); //20 - left face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[1],vertex_coords[7]})); //21 - right face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[2],vertex_coords[7]})); //22 - top face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[5]})); //23 - bottom face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[3]})); //24 - back face
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[4],vertex_coords[7]})); //25 - front face

			//center
			vertex_coords.emplace_back(T{0.5}*gutil::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[7]})); //26
		}

		void getChildVertices(std::vector<size_t> &child_vertices, const int child_number, const std::vector<size_t> &split_vertex_numbers) const override {
			assert(split_vertex_numbers.size()==vtk_n_vertices_when_split(VTK_ID));
			child_vertices.resize(vtk_n_vertices(VTK_ID));

			switch (child_number) {
				case (0): //voxel element containing original vertex 0
					child_vertices[0] = split_vertex_numbers[ 0]; //0
					child_vertices[1] = split_vertex_numbers[ 8]; //0-1
					child_vertices[2] = split_vertex_numbers[11]; //0-2
					child_vertices[3] = split_vertex_numbers[24]; //0-3
					child_vertices[4] = split_vertex_numbers[12]; //0-4
					child_vertices[5] = split_vertex_numbers[23]; //0-5
					child_vertices[6] = split_vertex_numbers[20]; //0-6
					child_vertices[7] = split_vertex_numbers[26]; //0-7
					break;

				case (1): //voxel element containing original vertex 1
					child_vertices[0] = split_vertex_numbers[ 8]; //0-1
					child_vertices[1] = split_vertex_numbers[ 1]; //1
					child_vertices[2] = split_vertex_numbers[24]; //0-3
					child_vertices[3] = split_vertex_numbers[ 9]; //1-3
					child_vertices[4] = split_vertex_numbers[23]; //0-5
					child_vertices[5] = split_vertex_numbers[15]; //1-5
					child_vertices[6] = split_vertex_numbers[26]; //0-7
					child_vertices[7] = split_vertex_numbers[21]; //1-7
					break;

				case (2): //voxel element containing original vertex 2
					child_vertices[0] = split_vertex_numbers[11]; //0-2
					child_vertices[1] = split_vertex_numbers[24]; //0-3
					child_vertices[2] = split_vertex_numbers[ 2]; //2
					child_vertices[3] = split_vertex_numbers[10]; //2-3
					child_vertices[4] = split_vertex_numbers[20]; //0-6
					child_vertices[5] = split_vertex_numbers[26]; //0-7
					child_vertices[6] = split_vertex_numbers[13]; //2-6
					child_vertices[7] = split_vertex_numbers[22]; //2-7
					break;

				case (3): //voxel element containing original vertex 3
					child_vertices[0] = split_vertex_numbers[24]; //0-3
					child_vertices[1] = split_vertex_numbers[ 9]; //1-3
					child_vertices[2] = split_vertex_numbers[10]; //2-3
					child_vertices[3] = split_vertex_numbers[ 3]; //3
					child_vertices[4] = split_vertex_numbers[26]; //0-7
					child_vertices[5] = split_vertex_numbers[21]; //1-7
					child_vertices[6] = split_vertex_numbers[22]; //2-7
					child_vertices[7] = split_vertex_numbers[14]; //3-7
					break;

				case (4): //voxel element containing original vertex 4
					child_vertices[0] = split_vertex_numbers[12]; //0-4
					child_vertices[1] = split_vertex_numbers[23]; //0-5
					child_vertices[2] = split_vertex_numbers[20]; //0-6
					child_vertices[3] = split_vertex_numbers[26]; //0-7
					child_vertices[4] = split_vertex_numbers[ 4]; //4
					child_vertices[5] = split_vertex_numbers[16]; //4-5
					child_vertices[6] = split_vertex_numbers[19]; //4-6
					child_vertices[7] = split_vertex_numbers[25]; //4-7
					break;

				case (5): //voxel element containing original vertex 5
					child_vertices[0] = split_vertex_numbers[23]; //0-5
					child_vertices[1] = split_vertex_numbers[15]; //1-5
					child_vertices[2] = split_vertex_numbers[26]; //0-7
					child_vertices[3] = split_vertex_numbers[21]; //1-7
					child_vertices[4] = split_vertex_numbers[16]; //4-5
					child_vertices[5] = split_vertex_numbers[ 5]; //5
					child_vertices[6] = split_vertex_numbers[25]; //4-7
					child_vertices[7] = split_vertex_numbers[17]; //5-7
					break;

				case (6): //voxel element containing original vertex 6
					child_vertices[0] = split_vertex_numbers[20]; //0-6
					child_vertices[1] = split_vertex_numbers[26]; //0-7
					child_vertices[2] = split_vertex_numbers[13]; //2-6
					child_vertices[3] = split_vertex_numbers[22]; //2-7
					child_vertices[4] = split_vertex_numbers[19]; //4-6
					child_vertices[5] = split_vertex_numbers[25]; //4-7
					child_vertices[6] = split_vertex_numbers[ 6]; //6
					child_vertices[7] = split_vertex_numbers[18]; //6-7
					break;

				case (7): //voxel element containing original vertex 7
					child_vertices[0] = split_vertex_numbers[26]; //0-7
					child_vertices[1] = split_vertex_numbers[21]; //1-7
					child_vertices[2] = split_vertex_numbers[22]; //2-7
					child_vertices[3] = split_vertex_numbers[14]; //3-7
					child_vertices[4] = split_vertex_numbers[25]; //4-7
					child_vertices[5] = split_vertex_numbers[17]; //5-7
					child_vertices[6] = split_vertex_numbers[18]; //6-7
					child_vertices[7] = split_vertex_numbers[ 7]; //7
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}
		}

		void getFaceVertices(std::vector<size_t> &face_vertices, const int face_number) const override {
			face_vertices.resize(4);
			switch (face_number) {
			case (0): //left
				face_vertices[0] = this->ELEM.vertices[0];
				face_vertices[1] = this->ELEM.vertices[4];
				face_vertices[2] = this->ELEM.vertices[2];
				face_vertices[3] = this->ELEM.vertices[6];
				break;
			case (1): //right
				face_vertices[0] = this->ELEM.vertices[1];
				face_vertices[1] = this->ELEM.vertices[3];
				face_vertices[2] = this->ELEM.vertices[5];
				face_vertices[3] = this->ELEM.vertices[7];
				break;
			case (2): //top
				face_vertices[0] = this->ELEM.vertices[2];
				face_vertices[1] = this->ELEM.vertices[6];
				face_vertices[2] = this->ELEM.vertices[3];
				face_vertices[3] = this->ELEM.vertices[7];
				break;
			case (3): //bottom
				face_vertices[0] = this->ELEM.vertices[0];
				face_vertices[1] = this->ELEM.vertices[1];
				face_vertices[2] = this->ELEM.vertices[4];
				face_vertices[3] = this->ELEM.vertices[5];
				break;
			case (4): //back
				face_vertices[0] = this->ELEM.vertices[1];
				face_vertices[1] = this->ELEM.vertices[0];
				face_vertices[2] = this->ELEM.vertices[3];
				face_vertices[3] = this->ELEM.vertices[2];
				break;
			case (5): //front
				face_vertices[0] = this->ELEM.vertices[4];
				face_vertices[1] = this->ELEM.vertices[5];
				face_vertices[2] = this->ELEM.vertices[6];
				face_vertices[3] = this->ELEM.vertices[7];
				break;
			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}

		void getSplitFaceVertices(std::vector<size_t> &split_face_vertices, const int face_number, const std::vector<size_t> &split_vertex_numbers) const override {
			split_face_vertices.resize(vtk_n_vertices_when_split(vtk_face_id(VTK_ID)));
			assert(split_vertex_numbers.size()==vtk_n_vertices_when_split(VTK_ID));

			switch (face_number) {
				case (0): // Left face [0, 4, 2, 6]
				split_face_vertices[0] = split_vertex_numbers[ 0]; //0
				split_face_vertices[1] = split_vertex_numbers[ 4]; //4
				split_face_vertices[2] = split_vertex_numbers[ 2]; //2
				split_face_vertices[3] = split_vertex_numbers[ 6]; //6
				split_face_vertices[4] = split_vertex_numbers[12]; //0-4
				split_face_vertices[5] = split_vertex_numbers[19]; //4-6
				split_face_vertices[6] = split_vertex_numbers[13]; //2-6
				split_face_vertices[7] = split_vertex_numbers[11]; //0-2
				split_face_vertices[8] = split_vertex_numbers[20]; //left face center
				break;

			case (1): // Right face [1, 3, 5, 7]
				split_face_vertices[0] = split_vertex_numbers[ 1]; //1
				split_face_vertices[1] = split_vertex_numbers[ 3]; //3
				split_face_vertices[2] = split_vertex_numbers[ 5]; //5
				split_face_vertices[3] = split_vertex_numbers[ 7]; //7
				split_face_vertices[4] = split_vertex_numbers[ 9]; //1-3
				split_face_vertices[5] = split_vertex_numbers[14]; //3-7
				split_face_vertices[6] = split_vertex_numbers[17]; //5-7
				split_face_vertices[7] = split_vertex_numbers[15]; //1-5
				split_face_vertices[8] = split_vertex_numbers[21]; //right face center
				break;

			case (2): // Top face [2, 6, 3, 7]
				split_face_vertices[0] = split_vertex_numbers[ 2]; //2
				split_face_vertices[1] = split_vertex_numbers[ 6]; //6
				split_face_vertices[2] = split_vertex_numbers[ 3]; //3
				split_face_vertices[3] = split_vertex_numbers[ 7]; //7
				split_face_vertices[4] = split_vertex_numbers[13]; //2-6
				split_face_vertices[5] = split_vertex_numbers[18]; //6-7
				split_face_vertices[6] = split_vertex_numbers[14]; //3-7
				split_face_vertices[7] = split_vertex_numbers[10]; //2-3
				split_face_vertices[8] = split_vertex_numbers[22]; //top face center
				break;

			case (3): // Bottom face [0, 1, 4, 5]
				split_face_vertices[0] = split_vertex_numbers[ 0]; //0
				split_face_vertices[1] = split_vertex_numbers[ 1]; //1
				split_face_vertices[2] = split_vertex_numbers[ 4]; //4
				split_face_vertices[3] = split_vertex_numbers[ 5]; //5
				split_face_vertices[4] = split_vertex_numbers[ 8]; //0-1
				split_face_vertices[5] = split_vertex_numbers[15]; //1-5
				split_face_vertices[6] = split_vertex_numbers[16]; //4-5
				split_face_vertices[7] = split_vertex_numbers[12]; //0-4
				split_face_vertices[8] = split_vertex_numbers[23]; //bottom face center
				break;

			case (4): // Back face [1, 0, 3, 2]
				split_face_vertices[0] = split_vertex_numbers[ 1]; //1
				split_face_vertices[1] = split_vertex_numbers[ 0]; //0
				split_face_vertices[2] = split_vertex_numbers[ 3]; //3
				split_face_vertices[3] = split_vertex_numbers[ 2]; //2
				split_face_vertices[4] = split_vertex_numbers[ 8]; //1-0
				split_face_vertices[5] = split_vertex_numbers[11]; //0-2
				split_face_vertices[6] = split_vertex_numbers[10]; //2-3
				split_face_vertices[7] = split_vertex_numbers[ 9]; //3-1
				split_face_vertices[8] = split_vertex_numbers[24]; //back face center
				break;

			case (5): // Front face [4, 5, 6, 7]
				split_face_vertices[0] = split_vertex_numbers[ 4]; //4
				split_face_vertices[1] = split_vertex_numbers[ 5]; //5
				split_face_vertices[2] = split_vertex_numbers[ 6]; //6
				split_face_vertices[3] = split_vertex_numbers[ 7]; //7
				split_face_vertices[4] = split_vertex_numbers[16]; //4-5
				split_face_vertices[5] = split_vertex_numbers[17]; //5-7
				split_face_vertices[6] = split_vertex_numbers[18]; //6-7
				split_face_vertices[7] = split_vertex_numbers[19]; //4-6
				split_face_vertices[8] = split_vertex_numbers[25]; //front face center
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}

		//evaluate the tri-linear shape function associated with vertex i on the reference element
		inline constexpr MapScalar_t eval_local_geo_shape_fun(const int i, const RefPoint_t& ref_coord) const noexcept override {
			assert(0<= i and i<N_VERTICES);
			return MapScalar_t(0.125)*(MapScalar_t(1)+REF_COORDS(i,0)*ref_coord[0])*(MapScalar_t(1)+REF_COORDS(i,1)*ref_coord[1])*(MapScalar_t(1)+REF_COORDS(i,2)*ref_coord[2]);
		}

		inline constexpr RefPoint_t  eval_local_geo_shape_grad(const int i, const RefPoint_t& ref_coord) const noexcept override {
			assert(0<=i and i<N_VERTICES);
			RefPoint_t result{};

			result[0] = MapScalar_t(0.125) * REF_COORDS(i,0)                               * (MapScalar_t(1)+REF_COORDS(i,1)*ref_coord[1]) * (MapScalar_t(1)+REF_COORDS(i,2)*ref_coord[2]);
			result[1] = MapScalar_t(0.125) * (MapScalar_t(1)+REF_COORDS(i,0)*ref_coord[0]) * REF_COORDS(i,1)                               * (MapScalar_t(1)+REF_COORDS(i,2)*ref_coord[2]);
			result[2] = MapScalar_t(0.125) * (MapScalar_t(1)+REF_COORDS(i,0)*ref_coord[0]) * (MapScalar_t(1)+REF_COORDS(i,1)*ref_coord[1]) * REF_COORDS(i,2);
			return result;
		}

		
		//evaluate the geometric mapping from the reference element to the actual element
		constexpr Point_t reference_to_geometric(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept override {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));

			Point_t result{}; //zero
			for (int i=0; i<N_VERTICES; i++) {
				result += eval_local_geo_shape_fun(i,ref_coord) * vertex_coords[i];
			}
			return result;
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		constexpr RefPoint_t geometric_to_reference(const std::vector<Point_t>& vertex_coords, const Point_t& coord) const noexcept override {return RefPoint_t{};}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		constexpr Jac_t eval_geo_shape_jac(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept override {
			Jac_t jacobian{};
			Point_t H = vertex_coords[7]-vertex_coords[0];
			for (int i=0; i<3; i++) {
				jacobian(i,i) = 0.5*H[i];
			}

			return jacobian;
		}
	};
}