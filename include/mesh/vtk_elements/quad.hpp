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
	template<typename Point_t>
	class VTK_QUAD : public VTK_ELEMENT<Point_t> {
	public:
		VTK_QUAD(const BasicElement &elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.vertices.size()==vtk_n_vertices(elem.vtkID));}
		static constexpr int VTK_ID  = QUAD_VTK_ID;
		static constexpr int REF_DIM = 2; //dimension of the reference element
		using Scalar_t    = typename Point_t::Scalar_t;
		using RefPoint_t  = gv::util::Point<REF_DIM, Scalar_t>; //type of point in the reference element
		using Matrix_t    = gv::util::Matrix<3,REF_DIM,Scalar_t>; //dimensions of the jacobian matrix (output space is always R3)

		using ScalarFun_t = std::function<Scalar_t(RefPoint_t)>; //function type to evaluate a basis in the element
		using VectorFun_t = std::function<Point_t(RefPoint_t)>; //function type to evaluate the gradient of a basis function
		using MatrixFun_t = std::function<Matrix_t(RefPoint_t)>; //function type to evaluate the jacobian of the isoparametric mapping

		void split(std::vector<Point_t>& vertex_coords) const override {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));
			vertex_coords.reserve(vtk_n_vertices_when_split(VTK_ID));
			using T = Scalar_t;
			
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

		bool isInterior(const std::vector<Point_t>& vertices, const Point_t& coord) const override {
			assert(false);
			return false;
		}
	};
}