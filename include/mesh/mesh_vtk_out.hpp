#pragma once

#include "mesh/mesh_util.hpp"

#include <cstring>
#include <fstream>

#include <limits>

namespace gv::mesh {
	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void print_topology_ascii_vtk(const std::string &filename, const BasicMesh<Node_t,Element_t,Face_t> &mesh) const {
		//open and check file
		std::ofstream file(filename);
		if (not file.is_open()){
			throw std::runtime_error("Couldn't open " + filename);
			return;
		}

		//create buffer
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes_ThreadLocked() << " float\n";
		for (size_t i=0; i<nNodes_ThreadLocked(); i++) { buffer << mesh._nodes[i].vertex << "\n";}
		
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		
		//ELEMENTS
		//calculate the number of entries required (numberOfNodes + listOfNodes)
		size_t nEntries = 0;
		const size_t nElements = mesh.nElems_ThreadLocked();
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			nEntries  += 1 + _elements[e_idx].nodes.size();
		}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			buffer << ELEM.nodes.size();
			for (size_t n=0; n<ELEM.nodes.size(); n++) {
				buffer << " " << ELEM.nodes[n];
			}
			buffer << "\n";
		}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VTK_ID
		buffer << "CELL_TYPES " << nElements << "\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			buffer << ELEM.vtkID << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::print_topology_binary_vtk(const std::string& filename) const {
	    
	    
	    //only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(typename Node_t::Scalar_t)==4 or sizeof(typename Node_t::Scalar_t)==8, "Unsupported floating point size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many nodes.
	    //additionally, the integers are expected to be signed in the legacy format. uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_nodes = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (_nodes.size()-1 > max_legacy_vtk_nodes) {throw std::runtime_error("Node index " + std::to_string(_nodes.size()-1) + " exceeds legacy VTK format limit.");}


	    //open the file in binary write mode
	    std::ofstream file(filename, std::ios::binary);
	    if (!file.is_open()) {
	        throw std::runtime_error("Cannot open file: " + filename);
	    }
	    
	    // Helper lambda to write big-endian integers (cast to int32_t due to legacy vtk)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_size_t = [&file](const size_t value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };

	    auto write_be_int = [&file](const int value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
	    
	    // Helper lambda to write big-endian floats (handles float, double, etc.)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_float = [&file](Node_t::Scalar_t value) {
	        if constexpr (sizeof(typename Node_t::Scalar_t) == 4) {
	            // 32-bit float
	            uint32_t temp;
	            std::memcpy(&temp, &value, sizeof(typename Node_t::Scalar_t));
	            uint32_t be_value = ((temp & 0xFF000000) >> 24) |
	                                ((temp & 0x00FF0000) >> 8)  |
	                                ((temp & 0x0000FF00) << 8)  |
	                                ((temp & 0x000000FF) << 24);
	            file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	        } else if constexpr (sizeof(typename Node_t::Scalar_t) == 8) {
	            // 64-bit double
	            uint64_t temp;
	            std::memcpy(&temp, &value, sizeof(typename Node_t::Scalar_t));
	            uint64_t be_value = ((temp & 0xFF00000000000000ULL) >> 56) |
	                                ((temp & 0x00FF000000000000ULL) >> 40) |
	                                ((temp & 0x0000FF0000000000ULL) >> 24) |
	                                ((temp & 0x000000FF00000000ULL) >> 8)  |
	                                ((temp & 0x00000000FF000000ULL) << 8)  |
	                                ((temp & 0x0000000000FF0000ULL) << 24) |
	                                ((temp & 0x000000000000FF00ULL) << 40) |
	                                ((temp & 0x00000000000000FFULL) << 56);
	            file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	        }
	    };
	    
	    // HEADER (note legacy vtk can combine ascii and binary data)
	    file << "# vtk DataFile Version 2.0\n";
	    file << "Mesh Data\n";
	    file << "BINARY\n\n";
	    file << "DATASET UNSTRUCTURED_GRID\n";
	    
	    // POINTS (binary data)
	    if constexpr (sizeof(typename Node_t::Scalar_t)==4) {file << "POINTS " << nNodes_ThreadLocked() << " float\n";}
	    else if constexpr (sizeof(typename Node_t::Scalar_t)==8) {file << "POINTS " << nNodes_ThreadLocked() << " double\n";}
	    
	    for (size_t i = 0; i < nNodes_ThreadLocked(); i++) {
	        write_be_float(_nodes[i].vertex[0]);
	        write_be_float(_nodes[i].vertex[1]);
	        write_be_float(_nodes[i].vertex[2]);
	    }
	    file << "\n";
	    
	    // ELEMENTS - calculate counts
	    size_t nEntries = 0;
	    size_t nElements = nElems_ThreadLocked();
	    #pragma omp parallel for reduction(+:nEntries)
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			nEntries  += 1 + _elements[e_idx].nodes.size();
	    }
	    
	    // CELLS (binary data)
	    file << "CELLS " << nElements << " " << nEntries << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			const Element_t &ELEM = _elements[e_idx];
			write_be_size_t(ELEM.nodes.size());
			for (size_t n = 0; n < ELEM.nodes.size(); n++) {
				write_be_size_t(ELEM.nodes[n]);
			}
	    }
	    file << "\n";
	    
	    // CELL_TYPES (binary data)
	    file << "CELL_TYPES " << nElements << "\n";
	    for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
	        const Element_t &ELEM = _elements[e_idx];
	        write_be_int(ELEM.vtkID);
	    }
	    file << "\n";
	    
	    file.close();
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::print_mesh_details_ascii_vtk(std::ostream &os) const {
		
		
		std::stringstream buffer;

		//NODE DETAILS
		int n_node_fields = 2;
		buffer << "POINT_DATA " << _nodes.size() << "\n";
		buffer << "FIELD node_info " << n_node_fields << "\n";

		//boundary
		size_t max_boundary_faces=0;
		#pragma omp parallel for reduction(std::max:max_boundary_faces)
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			max_boundary_faces = std::max(max_boundary_faces, _nodes[n_idx].boundary_faces.size());
		}

		buffer << "boundary " << max_boundary_faces << " " << _nodes.size() << " integer\n";
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];

			size_t i;
			for (i=0; i<NODE.boundary_faces.size(); i++) { buffer << NODE.boundary_faces[i] << " ";	}
			for (;i<max_boundary_faces; i++) { buffer << "-1 ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
		

		//elements
		size_t max_elem=0;
		#pragma omp parallel for reduction(std::max:max_elem)
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			max_elem = std::max(max_elem, _nodes[n_idx].elems.size());
		}
		
		buffer << "elements " << max_elem << " " << _nodes.size() << " integer\n";
		for (size_t n_idx=0; n_idx<_nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];
			size_t i;
			for (i=0; i<NODE.elems.size(); i++) { buffer << NODE.elems[i] << " ";}
			for (; i<max_elem; i++) { buffer << "-1 ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


		//ELEMENT DETAILS
		size_t nElements = _elements.size();

		buffer << "CELL_DATA " << nElements << "\n";
		int n_fields = 2;
		buffer << "FIELD elem_info " << n_fields << "\n";

		
		//index
		buffer << "element_index 1 " << nElements << " integer\n";
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			buffer << e_idx << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		
		//neighbors
		size_t max_neighbors=0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			getElementNeighbors_ThreadLocked(e_idx, neighbors[n_idx]);
			max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
			n_idx+=1;
		}
		buffer << "neighbors " << max_neighbors << " " << nElements << " integer\n";
		
		n_idx=0;
		for (size_t e_idx=0; e_idx<_elements.size(); e_idx++) {
			size_t i;
			for (i=0; i<neighbors[n_idx].size(); i++) {buffer << neighbors[n_idx][i] << " ";}
			for (; i<max_neighbors; i++) {buffer << "-1 ";}
			n_idx+=1;
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}


	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	void BasicMesh<Node_t,Element_t,Face_t>::print_mesh_details_binary_vtk(const std::string& filename) const {
		//only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many nodes.
	    //additionally, the integers are expected to be signed in the legacy format. uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_nodes = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (_nodes.size()-1 > max_legacy_vtk_nodes) {throw std::runtime_error("Node index " + std::to_string(_nodes.size()-1) + " exceeds legacy VTK format limit.");}


		
		
		
		// Open file in append mode
		std::ofstream file(filename, std::ios::binary | std::ios::app);
		if (!file.is_open()) {
			throw std::runtime_error("Cannot open file for appending: " + filename);
		}
		
		// Helper lambda to write big-endian integers (cast to int32_t due to legacy vtk)
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_be_size_t = [&file](const size_t value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };

	    auto write_be_int = [&file](const int value) {
	    	int32_t legacy_vtk_value = static_cast<int32_t>(value);
	    	uint32_t be_value = ((legacy_vtk_value & 0xFF000000) >> 24) |
	    						((legacy_vtk_value & 0x00FF0000) >> 8)  |
	    						((legacy_vtk_value & 0x0000FF00) << 8)  |
	    						((legacy_vtk_value & 0x000000FF) << 24);
	    	file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
		
		//NODE DETAILS
		int n_node_fields = 2;
		
		file << "POINT_DATA " << _nodes.size() << "\n";
		file << "FIELD node_info " << n_node_fields << "\n";
		
		//boundary
		size_t max_boundary_faces = 0;
		#pragma omp parallel for reduction(std::max:max_boundary_faces)
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			max_boundary_faces = std::max(max_boundary_faces, _nodes[n_idx].boundary_faces.size());
		}
		
		file << "boundary " << max_boundary_faces << " " << _nodes.size() << " int\n";
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];
			
			size_t i;
			for (i = 0; i < NODE.boundary_faces.size(); i++) {
				write_be_size_t(NODE.boundary_faces[i]);
			}
			for (; i < max_boundary_faces; i++) {
				write_be_int(-1);
			}
		}
		file << "\n";
		
		//elements
		size_t max_elem = 0;
		#pragma omp parallel for reduction(std::max:max_elem)
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			max_elem = std::max(max_elem, _nodes[n_idx].elems.size());
		}
		
		file << "elements " << max_elem << " " << _nodes.size() << " int\n";
		for (size_t n_idx = 0; n_idx < _nodes.size(); n_idx++) {
			const Node_t &NODE = _nodes[n_idx];
			size_t i;
			for (i = 0; i < NODE.elems.size(); i++) {
				write_be_size_t(NODE.elems[i]);
			}
			for (; i < max_elem; i++) {
				write_be_int(-1);
			}
		}
		file << "\n";
		
		//ELEMENT DETAILS
		size_t nElements = nElems_ThreadLocked();
		file << "CELL_DATA " << nElements << "\n";
		int n_fields = 2;
		file << "FIELD elem_info " << n_fields << "\n";
		

		//index
		file << "element_index 1 " << nElements << " int\n";
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			write_be_size_t(e_idx);
		}
		file << "\n";
		
		
		//neighbors
		size_t max_neighbors = 0;
		std::vector<std::vector<size_t>> neighbors(nElements);
		size_t n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			getElementNeighbors_ThreadLocked(e_idx, neighbors[n_idx]);
			max_neighbors = std::max(max_neighbors, neighbors[n_idx].size());
			n_idx += 1;
		}
		file << "neighbors " << max_neighbors << " " << nElements << " int\n";
		
		n_idx = 0;
		for (size_t e_idx = 0; e_idx < _elements.size(); e_idx++) {
			size_t i;
			for (i = 0; i < neighbors[n_idx].size(); i++) {
				write_be_size_t(neighbors[n_idx][i]);
			}
			for (; i < max_neighbors; i++) {
				write_be_int(-1);
			}
			n_idx += 1;
		}
		file << "\n";
		
		file.close();
	}




	
}