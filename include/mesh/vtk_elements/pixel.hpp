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
	///
	/////////////////////////////////////////////////
	template<typename Point_t>
	class VTK_PIXEL : public VTK_ELEMENT<Point_t> {
	public:
		VTK_PIXEL(const BasicElement& elem) : VTK_ELEMENT<Point_t>(elem) {assert(elem.vtkID==VTK_ID); assert(elem.vertices.size()==vtk_n_vertices(elem.vtkID));}
		static constexpr int VTK_ID  = PIXEL_VTK_ID;
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
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[1]})); //4 - bottom
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[1],vertex_coords[3]})); //5 - right
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[2],vertex_coords[3]})); //6 - top
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[2]})); //7 - left

			//center
			vertex_coords.emplace_back(T{0.5}*(vertex_coords[0]+vertex_coords[3])); //8
		}

		void getChildVertices(std::vector<size_t> &child_vertices, const int child_number, const std::vector<size_t> &split_vertex_numbers) const override {
			assert(split_vertex_numbers.size()==vtk_n_vertices_when_split(VTK_ID));
			child_vertices.resize(vtk_n_vertices(VTK_ID));

			switch (child_number) {
				case (0):
					child_vertices[0] = split_vertex_numbers[0]; //0
					child_vertices[1] = split_vertex_numbers[4]; //0-1
					child_vertices[2] = split_vertex_numbers[7]; //0-2
					child_vertices[3] = split_vertex_numbers[8]; //0-3
					break;
				case (1):
					child_vertices[0] = split_vertex_numbers[4]; //0-1
					child_vertices[1] = split_vertex_numbers[1]; //1
					child_vertices[2] = split_vertex_numbers[8]; //0-3
					child_vertices[3] = split_vertex_numbers[5]; //1-3
					break;
				case (2):
					child_vertices[0] = split_vertex_numbers[7]; //0-2
					child_vertices[1] = split_vertex_numbers[8]; //0-3
					child_vertices[2] = split_vertex_numbers[2]; //2
					child_vertices[3] = split_vertex_numbers[6]; //2-3
					break;
				case (3):
					child_vertices[0] = split_vertex_numbers[8]; //0-3
					child_vertices[1] = split_vertex_numbers[5]; //1-3
					child_vertices[2] = split_vertex_numbers[6]; //2-3
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
				face_vertices[1] = this->ELEM.vertices[3];
				break;
			case (2):
				face_vertices[0] = this->ELEM.vertices[3];
				face_vertices[1] = this->ELEM.vertices[2];
				break;
			case (3):
				face_vertices[0] = this->ELEM.vertices[2];
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

			case (1): // Right [1, 3]
				split_face_vertices[0] = split_vertex_numbers[ 1]; //1
				split_face_vertices[1] = split_vertex_numbers[ 3]; //3
				split_face_vertices[2] = split_vertex_numbers[ 5]; //1-3
				break;

			case (2): // Top [3, 2]
				split_face_vertices[0] = split_vertex_numbers[ 3]; //3
				split_face_vertices[1] = split_vertex_numbers[ 2]; //2
				split_face_vertices[2] = split_vertex_numbers[ 6]; //2-3
				break;

			case (3): // Left [2, 0]
				split_face_vertices[0] = split_vertex_numbers[ 2]; //2
				split_face_vertices[1] = split_vertex_numbers[ 0]; //0
				split_face_vertices[2] = split_vertex_numbers[ 7]; //0-2
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