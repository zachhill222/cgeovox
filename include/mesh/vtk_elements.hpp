#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "mesh/vtk_elements/base.hpp"
// #include "mesh/vtk_elements/line.hpp"
// #include "mesh/vtk_elements/pixel.hpp"
// #include "mesh/vtk_elements/quad.hpp"
#include "mesh/vtk_elements/voxel.hpp"
#include "mesh/vtk_elements/hexahedron.hpp"
#include "mesh/vtk_elements/vtk_element_poly.hpp"

#include "concepts.hpp"

#include <vector>
#include <cassert>
#include <iostream>


namespace gv::mesh
{	
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