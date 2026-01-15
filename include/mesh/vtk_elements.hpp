#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "mesh/vtk_elements/base.hpp"
#include "mesh/vtk_elements/line.hpp"
#include "mesh/vtk_elements/pixel.hpp"
#include "mesh/vtk_elements/quad.hpp"
#include "mesh/vtk_elements/voxel.hpp"
#include "mesh/vtk_elements/hexahedron.hpp"

#include "concepts.hpp"

#include <vector>
#include <cassert>
#include <iostream>


namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// Factory to create the appropriate vtk_element type
	/// Note that whenever this function is called, the result must be deleted.
	/// For example auto* VE = VTK_ELEMENT_FACTORY(ELEM); delete VE;
	/////////////////////////////////////////////////
	template<int space_dim, int ref_dim, Scalar VertexScalar_t, Scalar MapScalar_t=double> 
	VTK_ELEMENT<space_dim, ref_dim, VertexScalar_t,MapScalar_t>* _VTK_ELEMENT_FACTORY(const BasicElement &ELEM) {
		assert(ref_dim == vtk_ref_dim(ELEM.vtkID));
		if constexpr (ref_dim == 1) {
			switch (ELEM.vtkID) {
				case LINE_VTK_ID:       return new VTK_LINE<VertexScalar_t, MapScalar_t>(ELEM);
				default: throw std::invalid_argument("unknown element type of reference dimension 1.");
			}
		}
		else if constexpr (ref_dim == 2) {
			switch (ELEM.vtkID) {
				case PIXEL_VTK_ID:      return new VTK_PIXEL<VertexScalar_t, MapScalar_t>(ELEM);
				case QUAD_VTK_ID:       return new VTK_QUAD<VertexScalar_t, MapScalar_t>(ELEM);
				default: throw std::invalid_argument("unknown element type of reference dimension 2.");
			}
		}
		else if constexpr (ref_dim == 3) {
			switch (ELEM.vtkID) {
				case VOXEL_VTK_ID:      return new VTK_VOXEL<VertexScalar_t, MapScalar_t>(ELEM);
				case HEXAHEDRON_VTK_ID: return new VTK_HEXAHEDRON<VertexScalar_t, MapScalar_t>(ELEM);
				default: throw std::invalid_argument("unknown element type of reference dimension 3.");
			}
		}
		else {throw std::invalid_argument("unknown element type.");}
	};
	
	/////////////////////////////////////////////////
	/// Function to change an element into its isoparametric variant (e.g. pixel to quad or voxel to hexahedron).
	/// Technically this is simply defining a non-affine geometric mapping. It is only isoparametric if the appropriate
	/// FEM DOFs are used.
	/////////////////////////////////////////////////
	void makeIsoparametric(BasicElement &ELEM) {
		switch (ELEM.vtkID) {
		case PIXEL_VTK_ID:
			std::swap(ELEM.vertices[2],ELEM.vertices[3]);
			ELEM.vtkID = QUAD_VTK_ID;
			return;
		case VOXEL_VTK_ID:
			std::swap(ELEM.vertices[2],ELEM.vertices[3]);
			std::swap(ELEM.vertices[6],ELEM.vertices[7]);
			ELEM.vtkID = HEXAHEDRON_VTK_ID;
			return;
		}
	}
}