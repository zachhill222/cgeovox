#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/matrix.hpp"

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
	template<Scalar VertexScalar_t, Scalar MapScalar_t>
	class VTK_QUAD : public VTK_ELEMENT<3,2,VertexScalar_t,MapScalar_t> {
	public:
		//define types
		using BASE = VTK_ELEMENT<3,2,VertexScalar_t,MapScalar_t>;
		using typename BASE::Point_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		VTK_QUAD(const BasicElement &elem) : BASE(elem) {assert(elem.vtkID==VTK_ID); assert(elem.vertices.size()==vtk_n_vertices(elem.vtkID));}
		
		//vtk element type
		static constexpr int VTK_ID  = QUAD_VTK_ID;
		static constexpr int N_VERTICES = vtk_n_vertices(VTK_ID);

		//coordinates for the reference element. store in row-major to pull out rows easier.
		static constexpr gv::util::Matrix<4,2,MapScalar_t,false> REF_COORDS {
			{-1, -1},
			{ 1, -1},
			{-1,  1},
			{ 1,  1}
		};

		void split(std::vector<Point_t>& vertex_coords) const override {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));
			vertex_coords.reserve(vtk_n_vertices_when_split(VTK_ID));
			using T = VertexScalar_t;
			
			//edge midpoints
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[1]})); //4 - bottom (B)
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[1],vertex_coords[2]})); //5 - right (R)
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[2],vertex_coords[3]})); //6 - top (T)
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[3]})); //7 - left (L)

			//center
			vertex_coords.emplace_back(T{0.25}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[1],vertex_coords[2],vertex_coords[3]})); //8 (C)
		}

		void getChildVertices(std::vector<size_t> &child_vertices, const int child_number, const std::vector<size_t> &split_vertex_numbers) const override {
			assert(split_vertex_numbers.size()==vtk_n_vertices_when_split(VTK_ID));
			child_vertices.resize(vtk_n_vertices(VTK_ID));

			switch (child_number) {
				case (0):
					child_vertices[0] = split_vertex_numbers[0]; //0
					child_vertices[1] = split_vertex_numbers[4]; //B
					child_vertices[2] = split_vertex_numbers[8]; //C
					child_vertices[3] = split_vertex_numbers[7]; //L
					break;
				case (1):
					child_vertices[0] = split_vertex_numbers[4]; //B
					child_vertices[1] = split_vertex_numbers[1]; //1
					child_vertices[2] = split_vertex_numbers[5]; //R
					child_vertices[3] = split_vertex_numbers[8]; //C
					break;
				case (2):
					child_vertices[0] = split_vertex_numbers[8]; //C
					child_vertices[1] = split_vertex_numbers[5]; //R
					child_vertices[2] = split_vertex_numbers[2]; //2
					child_vertices[3] = split_vertex_numbers[6]; //T
					break;
				case (3):
					child_vertices[0] = split_vertex_numbers[7]; //L
					child_vertices[1] = split_vertex_numbers[8]; //C
					child_vertices[2] = split_vertex_numbers[6]; //T
					child_vertices[3] = split_vertex_numbers[3]; //3
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
			}
		}

		void getFaceVertices(std::vector<size_t> &face_vertices, const int face_number) const override {
			face_vertices.resize(2);
			switch (face_number) {
			case (0):
				face_vertices[0] = this->ELEM.vertices[0];
				face_vertices[1] = this->ELEM.vertices[1];
				break;
			case (1):
				face_vertices[0] = this->ELEM.vertices[1];
				face_vertices[1] = this->ELEM.vertices[2];
				break;
			case (2):
				face_vertices[0] = this->ELEM.vertices[2];
				face_vertices[1] = this->ELEM.vertices[3];
				break;
			case (3):
				face_vertices[0] = this->ELEM.vertices[3];
				face_vertices[1] = this->ELEM.vertices[0];
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
				case (0): // Bottom [0, 1]
				split_face_vertices[0] = split_vertex_numbers[ 0]; //0
				split_face_vertices[1] = split_vertex_numbers[ 1]; //1
				split_face_vertices[2] = split_vertex_numbers[ 4]; //0-1
				break;

			case (1): // Right [1, 2]
				split_face_vertices[0] = split_vertex_numbers[ 1]; //1
				split_face_vertices[1] = split_vertex_numbers[ 2]; //2
				split_face_vertices[2] = split_vertex_numbers[ 5]; //1-2
				break;

			case (2): // Top [2, 3]
				split_face_vertices[0] = split_vertex_numbers[ 2]; //2
				split_face_vertices[1] = split_vertex_numbers[ 3]; //3
				split_face_vertices[2] = split_vertex_numbers[ 6]; //2-3
				break;

			case (3): // Left [3, 0]
				split_face_vertices[0] = split_vertex_numbers[ 3]; //3
				split_face_vertices[1] = split_vertex_numbers[ 0]; //0
				split_face_vertices[2] = split_vertex_numbers[ 7]; //0-3
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}


		//evaluate the bi-linear shape function associated with vertex i on the reference element
		inline constexpr MapScalar_t eval_local_geo_shape_fun(const int i, const RefPoint_t& ref_coord) const noexcept override {
			assert(0<= i and i<N_VERTICES);
			return MapScalar_t(0.25)*(MapScalar_t(1)+REF_COORDS(i,0)*ref_coord[0])*(MapScalar_t(1)+REF_COORDS(i,1)*ref_coord[1]);
		}

		inline constexpr RefPoint_t  eval_local_geo_shape_grad(const int i, const RefPoint_t& ref_coord) const noexcept override {
			assert(0<= i and i<N_VERTICES);
			
			RefPoint_t result{};
			result[0] = MapScalar_t(0.25) * REF_COORDS(i,0)                               * (MapScalar_t(1)+REF_COORDS(i,1)*ref_coord[1]);
			result[1] = MapScalar_t(0.25) * (MapScalar_t(1)+REF_COORDS(i,0)*ref_coord[0]) * REF_COORDS(i,1);
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
		constexpr Jac_t   eval_geo_shape_jac(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept override {return Jac_t{};};
	};
}