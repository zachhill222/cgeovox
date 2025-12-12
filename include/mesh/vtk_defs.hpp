#pragma once

#include <string>

//element types/ids
#define VERTEX_VTK_ID 1
#define LINE_VTK_ID 3
#define TRIANGLE_VTK_ID 5
#define PIXEL_VTK_ID 8
#define QUAD_VTK_ID 9
#define TETRA_VTK_ID 10
#define VOXEL_VTK_ID 11
#define HEXAHEDRON_VTK_ID 12
#define QUADRATIC_EDGE_VTK_ID 21
#define QUADRATIC_TRIANGLE_VTK_ID 22
#define QUADRATIC_TETRA_VTK_ID 24
#define BIQUADRATIC_QUAD_VTK_ID 28
#define TRIQUADRATIC_HEXAHEDRON_VTK_ID 29


namespace gv::mesh {

	std::string vtk_id_to_string(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: 				 return "VERTEX";
			case LINE_VTK_ID: 					 return "LINE";
			case TRIANGLE_VTK_ID: 				 return "TRIANGLE";
			case PIXEL_VTK_ID: 					 return "PIXEL";
			case QUAD_VTK_ID: 					 return "QUAD";
			case TETRA_VTK_ID: 				 	 return "TETRA";
			case VOXEL_VTK_ID: 					 return "VOXEL";
			case HEXAHEDRON_VTK_ID: 			 return "HEXAHEDRON";
			case QUADRATIC_EDGE_VTK_ID: 		 return "QUADRATIC_EDGE";
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return "QUADRATIC_TRIANGLE";
			case QUADRATIC_TETRA_VTK_ID: 		 return "QUADRATIC_TETRA";
			case BIQUADRATIC_QUAD_VTK_ID: 		 return "BIQUADRATIC_QUAD";
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return "TRIQUADRATIC_HEXAHEDRON";
			default: return "UNKNOWN";
		}
	}
	

	/// Helper function to pair vtkID to number of vertices
	constexpr size_t vtk_n_vertices(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: 				 return 1;
			case LINE_VTK_ID: 					 return 2;
			case TRIANGLE_VTK_ID: 				 return 3;
			case PIXEL_VTK_ID: 					 return 4;
			case QUAD_VTK_ID: 					 return 4;
			case TETRA_VTK_ID: 					 return 4;
			case VOXEL_VTK_ID: 					 return 8;
			case HEXAHEDRON_VTK_ID: 			 return 8;
			case QUADRATIC_EDGE_VTK_ID: 	 	 return 3;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return 6;
			case QUADRATIC_TETRA_VTK_ID: 		 return 10;
			case BIQUADRATIC_QUAD_VTK_ID: 		 return 9;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return 27;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}

	/// Helper function to pair vtkID the reference dimension
	constexpr int vtk_ref_dim(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: 				 return 0;
			case LINE_VTK_ID: 					 return 1;
			case TRIANGLE_VTK_ID: 				 return 2;
			case PIXEL_VTK_ID: 					 return 2;
			case QUAD_VTK_ID: 					 return 2;
			case TETRA_VTK_ID: 					 return 3;
			case VOXEL_VTK_ID: 					 return 3;
			case HEXAHEDRON_VTK_ID: 			 return 3;
			case QUADRATIC_EDGE_VTK_ID: 	 	 return 2;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return 2;
			case QUADRATIC_TETRA_VTK_ID: 		 return 3;
			case BIQUADRATIC_QUAD_VTK_ID: 		 return 3;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return 3;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}

	/// Helper function to pair vtkID to number of faces
	constexpr int vtk_n_faces(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: 				 return 0;
			case LINE_VTK_ID: 					 return 1;
			case TRIANGLE_VTK_ID: 				 return 3;
			case PIXEL_VTK_ID: 					 return 4;
			case QUAD_VTK_ID: 					 return 4;
			case TETRA_VTK_ID: 					 return 4;
			case VOXEL_VTK_ID: 					 return 6;
			case HEXAHEDRON_VTK_ID: 			 return 6;
			case QUADRATIC_EDGE_VTK_ID: 		 return 2;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return 3;
			case QUADRATIC_TETRA_VTK_ID: 		 return 4;
			case BIQUADRATIC_QUAD_VTK_ID: 		 return 4;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return 6;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}

	/// Helper function to pair vtkID of an element to the vtkID of its face
	constexpr int vtk_face_id(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: assert(false); return 0;
			case LINE_VTK_ID: 					 return VERTEX_VTK_ID;
			case TRIANGLE_VTK_ID: 				 return LINE_VTK_ID;
			case PIXEL_VTK_ID: 					 return LINE_VTK_ID;
			case QUAD_VTK_ID: 					 return LINE_VTK_ID;
			case TETRA_VTK_ID: 					 return TRIANGLE_VTK_ID;
			case VOXEL_VTK_ID: 					 return PIXEL_VTK_ID;
			case HEXAHEDRON_VTK_ID: 			 return QUAD_VTK_ID;
			case QUADRATIC_EDGE_VTK_ID: 		 return VERTEX_VTK_ID;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return QUADRATIC_EDGE_VTK_ID;
			case QUADRATIC_TETRA_VTK_ID: 		 return QUADRATIC_TRIANGLE_VTK_ID;
			case BIQUADRATIC_QUAD_VTK_ID: 		 return QUADRATIC_EDGE_VTK_ID;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return BIQUADRATIC_QUAD_VTK_ID;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}

	/// Helper function get the number of children
	constexpr size_t vtk_n_children(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: assert(false);   return 0;
			case LINE_VTK_ID: 					 return 2;
			case TRIANGLE_VTK_ID: 				 return 4;
			case PIXEL_VTK_ID: 					 return 4;
			case QUAD_VTK_ID: 					 return 4;
			case TETRA_VTK_ID: 					 return 4;
			case VOXEL_VTK_ID: 					 return 8;
			case HEXAHEDRON_VTK_ID: 			 return 8;
			case QUADRATIC_EDGE_VTK_ID: 		 return 2;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return 4;
			case QUADRATIC_TETRA_VTK_ID: 		 return 4;
			case BIQUADRATIC_QUAD_VTK_ID: 		 return 8;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return 8;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}

	/// Helper function get the number nodes after split
	constexpr int vtk_n_vertices_when_split(const int vtkID) {
		switch (vtkID) {
			case VERTEX_VTK_ID: assert(false); 	 return 0;
			case LINE_VTK_ID: 					 return 3;
			case TRIANGLE_VTK_ID: 				 return 6;
			case PIXEL_VTK_ID: 					 return 9;
			case QUAD_VTK_ID: 					 return 9;
			case TETRA_VTK_ID: 					 return 8; //?
			case VOXEL_VTK_ID: 					 return 27;
			case HEXAHEDRON_VTK_ID: 			 return 27;
			case QUADRATIC_EDGE_VTK_ID: 		 return 5;
			case QUADRATIC_TRIANGLE_VTK_ID: 	 return 15;
			case QUADRATIC_TETRA_VTK_ID: 		 return 20; //?
			case BIQUADRATIC_QUAD_VTK_ID: 		 return 25;
			case TRIQUADRATIC_HEXAHEDRON_VTK_ID: return 125;
			default:
				throw std::invalid_argument("Unknown element type.");
				return -1;
		}
	}
}