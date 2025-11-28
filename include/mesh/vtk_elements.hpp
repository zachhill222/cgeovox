#pragma once

#include "util/point.hpp"
#include "util/box.hpp"


#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "mesh/vtk_elements/base.hpp"
#include "mesh/vtk_elements/line.hpp"
#include "mesh/vtk_elements/pixel.hpp"
#include "mesh/vtk_elements/quad.hpp"
#include "mesh/vtk_elements/voxel.hpp"
#include "mesh/vtk_elements/hexahedron.hpp"

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
	template <typename Point_t> 
	VTK_ELEMENT<Point_t>* _VTK_ELEMENT_FACTORY(const BasicElement &ELEM) {
		switch (ELEM.vtkID) {
			case LINE_VTK_ID:       return new VTK_LINE<Point_t>(ELEM);
			case PIXEL_VTK_ID:      return new VTK_PIXEL<Point_t>(ELEM);
			case QUAD_VTK_ID:       return new VTK_QUAD<Point_t>(ELEM);
			case VOXEL_VTK_ID:      return new VTK_VOXEL<Point_t>(ELEM);
			case HEXAHEDRON_VTK_ID: return new VTK_HEXAHEDRON<Point_t>(ELEM);
			default: throw std::invalid_argument("unknown element type.");
		}
	};
	
	/////////////////////////////////////////////////
	/// Function to change an element into its isoparametric variant (e.g. pixel to quad or voxel to hexahedron).
	/////////////////////////////////////////////////
	void makeIsoparametric(BasicElement &ELEM) {
		switch (ELEM.vtkID) {
		case PIXEL_VTK_ID:
			std::swap(ELEM.nodes[2],ELEM.nodes[3]);
			ELEM.vtkID = QUAD_VTK_ID;
			return;
		case VOXEL_VTK_ID:
			std::swap(ELEM.nodes[2],ELEM.nodes[3]);
			std::swap(ELEM.nodes[6],ELEM.nodes[7]);
			ELEM.vtkID = HEXAHEDRON_VTK_ID;
			return;
		}
	}
}