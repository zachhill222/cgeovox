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
	/* Voxel element vertex labels
		0 ------ 1  
	*/

	/////////////////////////////////////////////////
	/// Line element
	/////////////////////////////////////////////////
	template<Scalar VertexScalar_t, Scalar MapScalar_t>
	class VTK_LINE : public VTK_ELEMENT<3,1,VertexScalar_t,MapScalar_t>{
	public:
		//define types
		using BASE = VTK_ELEMENT<3,1,VertexScalar_t,MapScalar_t>;
		using typename BASE::Point_t;
		using typename BASE::RefPoint_t;
		using typename BASE::Jac_t;

		//constructor
		VTK_LINE(const BasicElement &elem) : BASE(elem) {assert(elem.vtkID==VTK_ID); assert(elem.vertices.size()==vtk_n_vertices(elem.vtkID));}
		
		//vtk element type
		static constexpr int VTK_ID  = LINE_VTK_ID;
		static constexpr int N_VERTICES = vtk_n_vertices(VTK_ID);

		//coordinates for the reference element
		static constexpr gv::util::Matrix<2,1,MapScalar_t> REF_COORDS {
			{-1}, {1}
		};

		void split(std::vector<Point_t> &vertex_coords) const override {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));
			using T = VertexScalar_t;
			vertex_coords.emplace_back(T{0.5}*gv::util::sorted_sum<3,T,T,T>({vertex_coords[0],vertex_coords[1]}));
		}

		void getChildVertices(std::vector<size_t> &child_vertices, const int child_number, const std::vector<size_t> &split_node_numbers) const override {
			assert(split_node_numbers.size()==vtk_n_vertices_when_split(VTK_ID));
			child_vertices.resize(vtk_n_vertices(VTK_ID));

			switch (child_number) {
				case (0):
					child_vertices[0] = split_node_numbers[0];
					child_vertices[1] = split_node_numbers[2];
					break;
				case (1):
					child_vertices[0] = split_node_numbers[2];
					child_vertices[1] = split_node_numbers[1];
					break;
				default:
					throw std::out_of_range("child number out of bounds");
					break;
				}
		}

		void getFaceVertices(std::vector<size_t> &face_vertices, const int face_number) const override {
			face_vertices.resize(1);
			face_vertices[0] = this->ELEM.vertices[face_number];
		}

		void getSplitFaceVertices(std::vector<size_t> &split_face_vertices, const int face_number, const std::vector<size_t> &split_node_numbers) const override {
			split_face_vertices.resize(vtk_n_vertices_when_split(vtk_face_id(VTK_ID)));
			assert(split_node_numbers.size()==vtk_n_vertices_when_split(VTK_ID));

			switch (face_number) {
				case (0): // Left [0, 2]
				split_face_vertices[0] = split_node_numbers[0];
				split_face_vertices[1] = split_node_numbers[2];
				break;

			case (1): // Right [2, 1]
				split_face_vertices[0] = split_node_numbers[2];
				split_face_vertices[1] = split_node_numbers[1];
				break;

			default:
				throw std::out_of_range("face number out of bounds");
				break;
			}
		}

		
		//evaluate the local shape functions that are used to map the reference element to the actual element
		inline constexpr MapScalar_t eval_local_geo_shape_fun(const int i, const RefPoint_t& ref_coord) const noexcept override {return MapScalar_t{};}
		inline constexpr RefPoint_t  eval_local_geo_shape_grad(const int i, const RefPoint_t& ref_coord) const noexcept override {return RefPoint_t{};}

		
		//evaluate the geometric mapping from the reference element to the actual element
		constexpr Point_t reference_to_geometric(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept {
			assert(vertex_coords.size()==vtk_n_vertices(VTK_ID));

			Point_t result{}; //zero
			for (int i=0; i<N_VERTICES; i++) {
				result += eval_local_geo_shape_fun(i,ref_coord) * vertex_coords[i];
			}
			return result;
		}

		//evaluate the geometric inverse mapping from the actual/geometric element to the reference element
		constexpr RefPoint_t geometric_to_reference(const std::vector<Point_t>& vertex_coords, const Point_t& coord) const noexcept {return RefPoint_t{};}

		//evaluate the jacobian matrix of the mapping from the reference element to the actual element
		constexpr Jac_t   eval_geo_shape_jac(const std::vector<Point_t>& vertex_coords, const RefPoint_t& ref_coord) const noexcept {return Jac_t{};};
	};
}