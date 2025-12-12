#pragma once

#include "mesh/mesh_util.hpp"
#include "util/scalars/float_manipulation.hpp"

#include <cstring>
#include <fstream>
#include <cstdint>
#include <limits>

namespace gv::mesh {
	/////////////////////////////////////////////////
	/// Write the node locations and element connectivity to the specified file. Data can be appended to the stream after this is called.
	/// When saving information (e.g., a solution to a PDE define on this mesh), this will initialize the mesh and then the data can be appended.
	///
	/// @param file        A filestream into the file.
	/// @param mesh        The mesh whos topology is to be written into the file.
	/// @param description An optional short description to add to the header.
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	void print_topology_ascii_vtk(std::ofstream &file, const Mesh_t &mesh, const std::string description="Mesh Data") {
		if (!file.is_open()) {
			throw std::runtime_error("File is not open");
		}

		using Element_t = typename Mesh_t::Element_t;
		using PrintPoint_t = gv::util::Point<3,float>;

		//get number of vertices and elements
		const size_t nNodes    = mesh.nNodes();
		const size_t nElements = mesh.nElems();

		//create buffer
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << description + "\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes << " float\n";
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			buffer << static_cast<PrintPoint_t>(it->coord) << "\n";
		}
		buffer << "\n";
		file   << buffer.rdbuf();
		buffer.str("");

		
		//ELEMENTS
		//calculate the number of entries required (numberOfNodes + listOfNodes)
		size_t nEntries = 0;
		for (const Element_t &ELEM : mesh) {nEntries += 1 + ELEM.vertices.size();}

		buffer << "CELLS " << nElements << " " << nEntries << "\n";
		for (const Element_t &ELEM : mesh) {
			buffer << ELEM.vertices.size();
			for (size_t n : ELEM.vertices) {buffer << " " << n;}
			buffer << "\n";
		}
		buffer << "\n";
		file   << buffer.rdbuf();
		buffer.str("");


		//VTK_ID
		buffer << "CELL_TYPES " << nElements << "\n";
		for (const Element_t &ELEM : mesh) {buffer << ELEM.vtkID << " ";}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");
	}


	/////////////////////////////////////////////////
	/// Print the details of the vertices and elements to the output stream. This includes element colors and which elements each node belongs to.
	/// Due to the way that field data is stored in ASCII VTK format, it will be difficult to append any additional information to a file afterwards.
	///
	/// @param file  A filstream into the file to write to.
	/// @param mesh  The mesh whos topology is to be written into the file.
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	void print_mesh_details_ascii_vtk(std::ofstream &file, const Mesh_t &mesh) {
		if (!file.is_open()) {
			throw std::runtime_error("File is not open");
		}

		using Vertex_t  = typename Mesh_t::Vertex_t;
		using Element_t = typename Mesh_t::Element_t;
		
		//get number of vertices and elements
		const size_t nNodes    = mesh.nNodes();
		const size_t nElements = mesh.nElems();

		std::stringstream buffer;

		//NODE DETAILS
		int n_node_fields = 2; //elements and boundary are always tracked
		if constexpr (requires {Vertex_t::index;}) {n_node_fields++;}

		buffer << "POINT_DATA " << nNodes << "\n";
		buffer << "FIELD node_info " << n_node_fields << "\n";

		//boundary
		size_t max_boundary_faces=1;
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			max_boundary_faces = std::max(max_boundary_faces, it->boundary_faces.size());
		}

		buffer << "boundary " << max_boundary_faces << " " << nNodes << " integer\n";
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			const Vertex_t &NODE = *it;

			size_t i;
			for (i=0; i<NODE.boundary_faces.size(); i++) { buffer << NODE.boundary_faces[i] << " ";	}
			for (;i<max_boundary_faces; i++) { buffer << "-1 ";}
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");
		
		//elements
		size_t max_elem=0;
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			max_elem = std::max(max_elem, it->elems.size());
		}
		
		buffer << "elements " << max_elem << " " << nNodes << " integer\n";
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			const Vertex_t &NODE = *it;
			size_t i;
			for (i=0; i<NODE.elems.size(); i++) { buffer << NODE.elems[i] << " ";}
			for (; i<max_elem; i++) { buffer << "-1 ";}
		}
		buffer << "\n\n";
		file   << buffer.rdbuf();
		buffer.str("");

		if constexpr (requires {Vertex_t::index;}) {
			buffer << "index 1 " << nNodes << " integer\n";
			for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {buffer << it->index << " ";}
			buffer << "\n\n";
			file   << buffer.rdbuf();
			buffer.str("");
		}


		//ELEMENT DETAILS
		buffer << "CELL_DATA " << nElements << "\n";
		int n_elem_fields = 0;
		if constexpr (requires {Element_t::color;})    {n_elem_fields++;}
		if constexpr (requires {Element_t::parent;})   {n_elem_fields++;}
		if constexpr (requires {Element_t::children;}) {n_elem_fields++;}
		if constexpr (requires {Element_t::index;})    {n_elem_fields++;}
		buffer << "FIELD elem_info " << n_elem_fields << "\n";

		
		if constexpr (requires {Element_t::index;}) {
			buffer << "index 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {buffer << ELEM.index << " ";}
			buffer << "\n\n";
			file   << buffer.rdbuf();
			buffer.str("");
		}

		if constexpr (requires {Element_t::color;}) {
			buffer << "color 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {
				if (ELEM.color < (size_t)-1) {buffer << ELEM.color << " ";} //valid color
				else {buffer << "-1 ";} //invalid color
			}
			buffer << "\n\n";
			file   << buffer.rdbuf();
			buffer.str("");
		}

		if constexpr (requires {Element_t::parent;}) {
			buffer << "parent 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {
				if (ELEM.parent < (size_t)-1) {buffer << ELEM.parent << " ";} //valid parent
				else {buffer << "-1 ";} //invalid parent
			}
			buffer << "\n\n";
			file   << buffer.rdbuf();
			buffer.str("");
		}

		if constexpr (requires {Element_t::children;}) {
			size_t max_children = 0;
			for (const Element_t &ELEM : mesh) {
				max_children = std::max(max_children, ELEM.children.size());
			}

			if (max_children>0) {
				buffer << "children " << max_children << " " << nElements << " integer\n";
				for (const Element_t &ELEM : mesh) {
					size_t i;
					for (i=0; i<ELEM.children.size(); i++) {buffer << ELEM.children[i] << " ";}
					for (; i<max_children; i++) {buffer << "-1 ";}
				}	
			} else {
				buffer << "children " << 1 << " " << nElements << " integer\n";
				for (size_t i=0; i<nElements; i++) {buffer << " -1";}
			}
			
			buffer << "\n\n";
			file   << buffer.rdbuf();
			buffer.str("");
		}
	}





	/////////////////////////////////////////////////
	/// Print the node locations and element connectivity to the output file in binary format. Data can be appended to the file after this is called.
	/// When saving information (e.g., a solution to a PDE define on this mesh), this will initialize the mesh and then the data can be appended.
	///
	/// @param file  A filstream into the file to write to.
	/// @param mesh        The mesh whos topology is to be written into the file.
	/// @param description An optional short description to add to the header.
	/////////////////////////////////////////////////
	template<BasicMeshType Mesh_t>
	void print_topology_binary_vtk(std::ofstream &file, const Mesh_t &mesh, const std::string description="Mesh Data") {
		if (!file.is_open()) {
			throw std::runtime_error("File is not open");
		}

	    using Vertex_t  = typename Mesh_t::Vertex_t;
		using Element_t = typename Mesh_t::Element_t;

		//get number of vertices and elements
		const size_t nNodes    = mesh.nNodes();
		const size_t nElements = mesh.nElems();

	    //only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(typename Vertex_t::Scalar_t)==4 or sizeof(typename Vertex_t::Scalar_t)==8, "Unsupported floating point size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many vertices.
	    //additionally, the integers are expected to be signed in the legacy format.
	    //uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_vertices = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (nNodes > max_legacy_vtk_vertices) {
	    	throw std::runtime_error("Node index " + std::to_string(nNodes) + " exceeds legacy VTK format limit.");
	    }
	    
	    // Helper lambda to write numbers in big-endian format. gv::util::FloatingPointBits
	    // can be initialized with types convertible to integer or floating point types
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_big_endian = [&file](auto value) {
	        gv::util::FloatingPointBits<8*sizeof(value)> converter(value);
	        auto be_value = converter.big_endian();
	        file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
	    
	    // HEADER (note legacy vtk can combine ascii and binary data)
	    file << "# vtk DataFile Version 2.0\n";
	    file << description << "\n";
	    file << "BINARY\n\n";
	    file << "DATASET UNSTRUCTURED_GRID\n";
	    
	    // POINTS (binary data)
	    if      constexpr (sizeof(typename Vertex_t::Scalar_t)==4) {file << "POINTS " << nNodes << " float\n";}
	    else if constexpr (sizeof(typename Vertex_t::Scalar_t)==8) {file << "POINTS " << nNodes << " double\n";}
	    
	    for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			const Vertex_t &NODE = *it;
			if constexpr (sizeof(typename Vertex_t::Scalar_t)==4) {
				write_big_endian(static_cast<float>(NODE.coord[0]));
				write_big_endian(static_cast<float>(NODE.coord[1]));
				write_big_endian(static_cast<float>(NODE.coord[2]));
			}
		    else if constexpr (sizeof(typename Vertex_t::Scalar_t)==8) {
		    	write_big_endian(static_cast<double>(NODE.coord[0]));
				write_big_endian(static_cast<double>(NODE.coord[1]));
				write_big_endian(static_cast<double>(NODE.coord[2]));
		    }
	        
	    }
	    file << "\n";
	    
	    // ELEMENTS - calculate counts
	    size_t nEntries = 0;
	    for (const Element_t &ELEM : mesh) {nEntries  += 1 + ELEM.vertices.size();}
	    
	    // CELLS (binary data)
	    file << "CELLS " << nElements << " " << nEntries << "\n";
	    for (const Element_t &ELEM : mesh) {
			write_big_endian(static_cast<int>(ELEM.vertices.size()));
			for (size_t n_idx : ELEM.vertices) {write_big_endian(static_cast<int>(n_idx));}
	    }
	    file << "\n";
	    
	    // CELL_TYPES (binary data)
	    file << "CELL_TYPES " << nElements << "\n";
	    for (const Element_t &ELEM : mesh) {write_big_endian(static_cast<int>(ELEM.vtkID));}
	    file << "\n";
	}



	
		
	/////////////////////////////////////////////////
	/// Print the details of the vertices and elements to the output stream. This includes element colors and which elements each node belongs to.
	/// Due to the way that field data is stored in BINARY VTK format, it will be difficult to append any additional information to a file afterwards.
	///
	/// @param file  A filstream into the file to write to.
	/// @param mesh  The mesh whos topology is to be written into the file.
	/////////////////////////////////////////////////

	template<BasicMeshType Mesh_t>
	void print_mesh_details_binary_vtk(std::ofstream &file, const Mesh_t &mesh) {
		if (!file.is_open()) {
			throw std::runtime_error("File is not open");
		}


		using Vertex_t  = typename Mesh_t::Vertex_t;
		using Element_t = typename Mesh_t::Element_t;

		//get number of vertices and elements
		const size_t nNodes    = mesh.nNodes();
		const size_t nElements = mesh.nElems();

	    //only 32 and 64 bit data types are supported. can add more if necessary
	    static_assert(sizeof(size_t)==4 or sizeof(size_t)==8, "Unsupported size_t size");
	    static_assert(sizeof(typename Vertex_t::Scalar_t)==4 or sizeof(typename Vertex_t::Scalar_t)==8, "Unsupported floating point size");

	    //in this file format, the node indices must be 4 bytes. ensure that there are not too many vertices.
	    //additionally, the integers are expected to be signed in the legacy format.
	    //uint32_t **might** be possible, but likely we need xml files for meshes that large.
	    constexpr size_t max_legacy_vtk_vertices = static_cast<size_t>(std::numeric_limits<int32_t>::max());
	    if (nNodes > max_legacy_vtk_vertices) {
	    	throw std::runtime_error("Node index " + std::to_string(nNodes) + " exceeds legacy VTK format limit.");
	    }

	     // Helper lambda to write numbers in big-endian format. gv::util::FloatingPointBits
	    // can be initialized with types convertible to integer or floating point types
	    // PC is in little-endian, vtk/Paraview wants big-endian
	    auto write_big_endian = [&file](auto value) {
	        gv::util::FloatingPointBits<8*sizeof(value)> converter(value);
	        auto be_value = converter.big_endian();
	        file.write(reinterpret_cast<const char*>(&be_value), sizeof(be_value));
	    };
		
		//NODE DETAILS
		int n_node_fields = 2; //elements and boundary are always tracked
		if constexpr (requires {Vertex_t::index;}) {n_node_fields++;}
		
		file << "POINT_DATA " << nNodes << "\n";
		file << "FIELD node_info " << n_node_fields << "\n";
		
		//boundary
		size_t max_boundary_faces=1;
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			max_boundary_faces = std::max(max_boundary_faces, it->boundary_faces.size());
		}
		
		file << "boundary " << max_boundary_faces << " " << nNodes << " int\n";
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			const Vertex_t &NODE = *it;
			
			size_t i;
			for (i = 0; i < NODE.boundary_faces.size(); i++) {write_big_endian(static_cast<int>(NODE.boundary_faces[i]));}
			for (; i < max_boundary_faces; i++) {write_big_endian(int(-1));}
		}
		file << "\n";
		
		//elements
		size_t max_elem=0;
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			max_elem = std::max(max_elem, it->elems.size());
		}
		
		file << "elements " << max_elem << " " << nNodes << " int\n";
		for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
			const Vertex_t &NODE = *it;
			size_t i;
			for (i = 0; i < NODE.elems.size(); i++) {write_big_endian(static_cast<int>(NODE.elems[i]));}
			for (; i < max_elem; i++) {write_big_endian(int(-1));}
		}
		file << "\n";

		if constexpr (requires {Vertex_t::index;}) {
		    file << "index 1 " << nNodes << " int\n";
		    for (auto it=mesh.nodeBegin(); it!=mesh.nodeEnd(); ++it) {
		        write_big_endian(static_cast<int>(it->index));
		    }
		    file << "\n";
		}
		


		//ELEMENT DETAILS
		int n_elem_fields = 0;
		if constexpr (requires {Element_t::color;})    {n_elem_fields++;}
		if constexpr (requires {Element_t::parent;})   {n_elem_fields++;}
		if constexpr (requires {Element_t::children;}) {n_elem_fields++;}
		if constexpr (requires {Element_t::index;})    {n_elem_fields++;}


		file << "CELL_DATA " << nElements << "\n";
		file << "FIELD elem_info " << n_elem_fields << "\n";
		
		if constexpr (requires {Element_t::index;}) {
			file << "index 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {write_big_endian(static_cast<int>(ELEM.index));}
		}
		
		if constexpr (requires {Element_t::color;}) {
			file << "color 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {
				if (ELEM.color < (size_t)-1) {write_big_endian(static_cast<int>(ELEM.color));} //valid color
				else {write_big_endian(int(-1));} //invalid color
			}
		}

		if constexpr (requires {Element_t::parent;}) {
			file << "parent 1 " << nElements << " integer\n";
			for (const Element_t &ELEM : mesh) {
				if (ELEM.parent < (size_t)-1) {write_big_endian(static_cast<int>(ELEM.parent));} //valid parent
				else {write_big_endian(int(-1));} //invalid parent
			}
		}
		
		if constexpr (requires {Element_t::children;}) {
			size_t max_children = 0;
			for (const Element_t &ELEM : mesh) {
				max_children = std::max(max_children, ELEM.children.size());
			}

			if (max_children>0) {
				file << "children " << max_children << " " << nElements << " integer\n";
				for (const Element_t &ELEM : mesh) {
					size_t i;
					for (i=0; i<ELEM.children.size(); i++) {write_big_endian(static_cast<int>(ELEM.children[i]));}
					for (; i<max_children; i++) {write_big_endian(int(-1));}
				}
			} else {
				file << "children " << 1 << " " << nElements << " integer\n";
				for (size_t i=0; i<nElements; i++) {write_big_endian(int(-1));}
			}
		}
	}
}