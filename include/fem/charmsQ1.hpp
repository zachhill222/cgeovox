#pragma once


#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "util/octree_util.hpp"

#include "fem/charmsQ1_util.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

namespace gv::fem
{
	class CharmsQ1Mesh
	{
	public:
		using Element_t     = CharmsQ1Element;
		using BasisFun_t    = CharmsQ1BasisFun;
		using BasisList_t   = CharmsQ1BasisFunOctree;
		using ElementList_t = CharmsQ1ElementOctree;
		using VertexList_t  = CharmsQ1BasisFun::VertexList_t;
		using Point_t       = gv::util::Point<3,double>;
		using Box_t         = gv::util::Box<3>;
		using Index_t       = gv::util::Point<3,size_t>;

		VertexList_t  vertices;
		ElementList_t elements;
		BasisList_t   basis;
		

		//constructor for creating a uniform mesh of a box
		CharmsQ1Mesh(const Box_t &domain, const Index_t &N) : vertices(domain), elements(domain), basis(domain)
		{
			//reserve space
			vertices.reserve((N[0]+1)*(N[1]+1)*(N[2]+1));
			elements.reserve(N[0]*N[1]*N[2]);
			basis.reserve(vertices.capacity());

			//construct elements
			Point_t H = domain.sidelength() / Point_t(N);
			for (size_t i=0; i<N[0]; i++) {
				for (size_t j=0; j<N[1]; j++) {
					for (size_t k=0; k<N[2]; k++) {
						Point_t low  = domain.low() + Point_t{i,j,k} * H;
						Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;

						Element_t elem(&vertices, &elements, &basis, Box_t{low,high}); //this adds the appropriate vertices to the vertex list
						size_t elem_idx;
						elements.push_back(elem, elem_idx); //add element to mesh
						elements[elem_idx].element_list_index = elem_idx;
					}
				}
			}

			//construct basis functions and add references to elements. in the coarsest mesh, each vertex corresponds to a basis function.
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFun_t fun(&vertices, &elements, &basis, v_idx);
				fun.set_support();
				size_t fun_idx;
				int flag = basis.push_back(fun, fun_idx); assert(flag==1);
				basis[fun_idx].basis_list_index = fun_idx;
				basis[fun_idx].update_element_basis_lists();
			}
		}

		//convenient functions
		size_t nNodes() const {return vertices.size();}
		size_t nElems() const {return elements.size();}
		size_t nBasis() const {return basis.size();}

		//file io
		void vtkprint(std::ostream &os) const; //write mesh (as unstructured non-conforming voxels) in vtk format to specified stream
		void save_as(std::string filename) const; //save mesh information to file, uses vtkprint()

		//refinement
		void h_refine(const size_t basis_idx)
		{
			assert(basis_idx<nBasis());
			while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
			while (elements.capacity() < elements.size()+64) {elements.reserve(2*elements.capacity());}
			while (basis.capacity() < basis.size()+27) {basis.reserve(2*basis.capacity());}
			//create candidate basis functions and elements
			std::cout << basis[basis_idx] << std::endl;
			basis[basis_idx].subdivide();
			std::cout << basis[basis_idx] << std::endl;

		}
	};



	//vtkprint implementation
	void CharmsQ1Mesh::vtkprint(std::ostream &os) const
	{
		//write to buffer and flush buffer to the stream
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes() << " float\n";
		for (size_t i=0; i<nNodes(); i++) { buffer << vertices[i] << "\n";}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENTS
		buffer << "CELLS " << nElems() << " " << (1+8)*nElems() << "\n";
		for (size_t i=0; i<nElems(); i++)
		{
			buffer << 8 << " "; //8 nodes per element
			for (size_t j=0; j<8; j++)
			{
				buffer << elements[i].node[j] << " ";
			}
			buffer << "\n";
		}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VTK IDs
		buffer << "CELL_TYPES " << nElems() << "\n";
		for (size_t i=0; i<nElems(); i++) {buffer << 11 << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH CELL (depth, is_active, global_index, #basis_a, #basis_s)
		buffer << "CELL_DATA " << nElems() << "\n";
		buffer << "FIELD mesh_element_info 5\n";

		//ELEMENT DEPTH
		buffer << "depth 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].depth << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT ACTIVE MARKER
		buffer << "is_active 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].is_active << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT GLOBAL INDEX
		buffer << "elem_index 2 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << i << " " << elements[i].element_list_index << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "basis_a 8 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {to_stream(buffer, elements[i].basis_a, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "basis_s 8 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {to_stream(buffer, elements[i].basis_s, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH VERTEX (#basis_total, active_basis_index, active_basis_depth, support)
		buffer << "POINT_DATA " << nNodes() << "\n";
		buffer << "FIELD mesh_vertex_info 4\n";

		//loop through vertices to collect info
		size_t basis_total_size[nNodes()] {0};
		size_t active_basis_index[nNodes()] {0};
		size_t active_basis_depth[nNodes()] {0};
		for (size_t i=0; i<nNodes(); i++)
		{
			std::vector<size_t> basis_total = basis.get_data_indices(vertices[i]);
			basis_total_size[i] = 0;
			int count = 0;
			for (auto it=basis_total.begin(); it!=basis_total.end(); ++it)
			{
				if (basis[*it].coord()==vertices[i])
				{
					basis_total_size[i]++;
					if (basis[*it].is_active)
					{
						count++;
						active_basis_index[i] = *it;
						active_basis_depth[i] = basis[*it].depth;
					}
				}
				
			}
			assert(count<=1);
			if (count==0)
			{
				active_basis_index[i] = (size_t) -1;
				active_basis_depth[i] = (size_t) -1;
			}
		}

		//VERTEX TOTAL NUMBER OF CONSTRUCTED BASIS FUNCTIONS
		buffer << "#basis_total 1 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++) {buffer << basis_total_size[i] << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS INDEX
		buffer << "active_basis_index 1 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]<nBasis()) {buffer << active_basis_index[i] << " ";}
			else {buffer << -1 << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS DEPTH
		buffer << "active_basis_depth 1 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_depth[i] != (size_t) -1) {buffer << active_basis_depth[i] << " ";}
			else {buffer << -1 << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS SUPPORT
		buffer << "active_basis_support 8 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 "; continue;}
			else {to_stream(buffer, basis[i].support, 8);}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}

	//save mesh implementation
	void CharmsQ1Mesh::save_as(std::string filename) const
	{
		//open and check file
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}

		//print mesh to file
		vtkprint(meshfile);
		meshfile.close();
	}
}