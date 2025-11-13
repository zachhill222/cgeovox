#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "util/octree_util.hpp"

#include "mesh/Q1.hpp"

#include "charms/charmsQ1_util.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <sstream>
#include <fstream>

namespace gv::charms
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
						size_t new_elem_idx = (size_t) -1;
						elements.push_back(elem, new_elem_idx); //add element to mesh
						assert(elements[new_elem_idx].list_index == new_elem_idx);
					}
				}
			}

			//construct basis functions and add references to elements. in the coarsest mesh, each vertex corresponds to a basis function.
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFun_t fun(&vertices, &elements, &basis, v_idx);
				fun.set_support(); //TODO: speed this up or remove from here.

				size_t fun_idx = (size_t) -1;
				int flag = basis.push_back(fun, fun_idx);
				assert(flag==1);
				assert(basis[fun_idx].list_index == fun_idx);

				//add basis fucntion to basis_s of all support elements
				for (int i=0; i<basis[fun_idx].cursor_support; i++)
				{
					elements[basis[fun_idx].support[i]].insert_basis_s(fun_idx);
				}
				// basis[fun_idx].update_element_basis_lists();
			}
		}

		//constructor for creating a charms mesh from a standard mesh
		CharmsQ1Mesh(const gv::mesh::VoxelMeshQ1 &mesh) : vertices(mesh.bbox()), elements(mesh.bbox()), basis(mesh.bbox())
		{
			//reserve space
			vertices.reserve(mesh.nNodes());
			elements.reserve(mesh.nElems());
			basis.reserve(mesh.nNodes());

			//copy vertices
			for (size_t v_idx=0; v_idx<mesh.nNodes(); v_idx++)
			{
				vertices.push_back(mesh.node(v_idx));
			}

			//construct elements
			for (size_t e_idx=0; e_idx<mesh.nElems(); e_idx++)
			{
				size_t elem_nodes[8];
				mesh.get_element(e_idx, elem_nodes);

				std::cout << "elem_nodes= ";
				for (int i=0; i<8; i++) {std::cout << elem_nodes[i] << " ";}
				std::cout << std::endl;

				Element_t elem(&vertices, &elements, &basis, elem_nodes);
				int flag = elements.push_back(elem);
				assert(flag==1);
			}

			//construct basis functions
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFun_t fun(&vertices, &elements, &basis, v_idx);
				fun.set_support();
				size_t fun_idx = v_idx;
				int flag = basis.push_back(fun, fun_idx);
				assert(flag==1);
				const BasisFun_t& FUN = basis[fun_idx];
				assert(FUN.list_index==fun_idx);
				assert(fun_idx==v_idx);

				//add index of this basis to its support elements

				for (int i=0; i<basis[FUN.list_index].cursor_support; i++)
				{
					elements[FUN.support[i]].insert_basis_s(FUN.list_index);
				}
			}
		}

		//convenient functions
		size_t nNodes() const {return vertices.size();}
		size_t nElems() const {return elements.size();}
		size_t nBasis() const {return basis.size();}

		//file io
		void vtkprint(std::ostream &os) const; //write mesh (as unstructured non-conforming voxels) in vtk format to specified stream
		void save_as(std::string filename) const; //save mesh information to file, uses vtkprint()

		//hierarchical refinement
		void h_refine(const size_t basis_idx); //add detail functions
		void h_unrefine(const size_t basis_idx); //remove detail functions

		//quasi-hierarchical refinement
		void q_refine(const size_t basis_idx); //de-activate the specified basis function and activate its children
		void q_unrefine(const size_t basis_idx); //activate the specified basis function and de-activate its children

		//matrix generation (Q1)
		template<int Format_t>
		void make_mass_matrix(Eigen::SparseMatrix<double,Format_t> &mat) const;

		template<int Format_t>
		void make_stiff_matrix(Eigen::SparseMatrix<double,Format_t> &mat) const;
	};


	//hierarchical refinement
	void CharmsQ1Mesh::h_refine(const size_t basis_idx)
	{
		assert(basis_idx<nBasis());
		
		//verify that this basis function can be refined
		assert(!basis[basis_idx].is_refined);

		while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
		while (elements.capacity() < elements.size()+64) {elements.reserve(2*elements.capacity());}
		while (basis.capacity() < basis.size()+27) {basis.reserve(2*basis.capacity());}
		
		//create candidate basis functions and elements
		basis[basis_idx].subdivide();
	
		for (int i=0; i<basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = basis[basis_idx].child[i];
			if (basis[c_idx].is_odd) {basis[c_idx].activate();}
		}

		//mark this basis function as refined
		basis[basis_idx].is_refined = true;
	}

	//hierarchical un-refinement
	void CharmsQ1Mesh::h_unrefine(const size_t basis_idx) //remove the detail functions
	{
		assert(basis_idx<nBasis());
		if (!basis[basis_idx].is_active) {return;}

		//verify that this basis function is allowed to be un-refined
		assert(basis[basis_idx].is_refined);

		//verify that no children functions are refined
		for (int i=0; i<basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = basis[basis_idx].child[i];
			assert(!basis[c_idx].is_refined);
			if (basis[c_idx].is_odd and basis[c_idx].is_active) {basis[c_idx].deactivate();}
		}

		//ensure all support elements are active
		for (int i=0; i<basis[basis_idx].cursor_support; i++)
		{
			size_t s_idx = basis[basis_idx].support[i];

			//check if there are any active descendent elements
			std::vector<size_t> desc_elems = elements[s_idx].descendent_elements();
			bool has_active_descendent = false;
			for (auto it=desc_elems.begin(); it!=desc_elems.end(); ++it)
			{
				has_active_descendent = elements[*it].is_active;
				if (has_active_descendent) {break;}
			}
			
			if (!has_active_descendent) {elements[s_idx].is_active = true;}
		}

		//mark this basis function as not refined
		basis[basis_idx].is_refined = false;
	}


	//quasi-hierarchical refinement
	void CharmsQ1Mesh::q_refine(const size_t basis_idx)
	{
		assert(basis_idx<nBasis());
		
		//verify that this basis function can be refined
		assert(!basis[basis_idx].is_refined);

		while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
		while (elements.capacity() < elements.size()+64) {elements.reserve(2*elements.capacity());}
		while (basis.capacity() < basis.size()+27) {basis.reserve(2*basis.capacity());}
		
		//create candidate basis functions and elements
		basis[basis_idx].subdivide();
		
		//deactivate current basis
		basis[basis_idx].deactivate();

		for (int i=0; i<basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = basis[basis_idx].child[i];
			basis[c_idx].activate();
		}

		//mark this basis function as refined
		basis[basis_idx].is_refined = true;
	}

	//quasi-hierarchical un-refinement
	void CharmsQ1Mesh::q_unrefine(const size_t basis_idx)
	{
		assert(basis_idx<nBasis());
		if (basis[basis_idx].is_active) {return;} //for quasi-hierarchical unrefinement, the "center" basis must be de-activated

		//verify that this basis function is allowed to be un-refined
		assert(basis[basis_idx].is_refined);

		//verify that no children functions are refined and de-activate all children
		for (int i=0; i<basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = basis[basis_idx].child[i];
			assert(!basis[c_idx].is_refined);
			if (basis[c_idx].is_active) {basis[c_idx].deactivate();}
		}

		//ensure all support elements are active
		for (int i=0; i<basis[basis_idx].cursor_support; i++)
		{
			size_t s_idx = basis[basis_idx].support[i];

			//check if there are any active descendent elements
			std::vector<size_t> desc_elems = elements[s_idx].descendent_elements();
			bool has_active_descendent = false;
			for (auto it=desc_elems.begin(); it!=desc_elems.end(); ++it)
			{
				has_active_descendent = elements[*it].is_active;
				if (has_active_descendent) {break;}
			}
			
			if (!has_active_descendent) {elements[s_idx].is_active = true;}
		}

		//mark this basis function as not refined and activate it
		basis[basis_idx].is_refined = false;
		basis[basis_idx].activate();
	}













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
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].vtk_id << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH CELL
		buffer << "CELL_DATA " << nElems() << "\n";
		buffer << "FIELD mesh_element_info 6\n";

		//ELEMENT INDEX
		buffer << "index 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].list_index << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


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

		//ELEMENT IS_SUBDIVIDED MARKER
		buffer << "is_subdivided 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].is_subdivided << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_S
		buffer << "basis_s 8 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {to_stream(buffer, elements[i].basis_s, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "basis_a[0:7] 8 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {to_stream(buffer, elements[i].basis_a, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		



		//MESH INFORMATION AT EACH VERTEX (active_basis_index, active_basis_depth, support)
		buffer << "POINT_DATA " << nNodes() << "\n";
		buffer << "FIELD mesh_vertex_info 6\n";

		//loop through vertices to get index of active basis function
		size_t* active_basis_index = new size_t[nNodes()];
		for (size_t i=0; i<nNodes(); i++)
		{
			active_basis_index[i] = 0;
			std::vector<size_t> basis_total = basis.get_data_indices(vertices[i]);
			int count = 0;
			for (auto it=basis_total.begin(); it!=basis_total.end(); ++it)
			{
				if (basis[*it].is_active and basis[*it].coord()==vertices[i])
				{
					count++;
					active_basis_index[i] = *it;
				}
			}
			assert(count<=1);
			if (count==0) {active_basis_index[i] = (size_t) -1;}
		}

		//VERTEX ACTIVE BASIS INDEX
		buffer << "index 1 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]<nBasis()) {buffer << active_basis_index[i] << " ";}
			else {buffer << -1 << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS DEPTH
		buffer << "depth 1 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]==(size_t)-1) {buffer << -1 << " ";}
			else
			{
				const BasisFun_t& FUN = basis[active_basis_index[i]];
				buffer << FUN.depth << " ";
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS SUPPORT
		buffer << "support 8 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]==(size_t)-1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = basis[active_basis_index[i]];
				to_stream(buffer, FUN.support, 8);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS PARENTS
		buffer << "parent 27 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]==(size_t)-1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = basis[active_basis_index[i]];
				to_stream(buffer, FUN.parent, 27);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS PARENTS
		buffer << "child 27 " << nNodes() << " integer\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			if (active_basis_index[i]==(size_t)-1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = basis[active_basis_index[i]];
				to_stream(buffer, FUN.child, 27);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VERTEX TEST DATA
		buffer << "testdata 2 " << nNodes() << " float\n";
		for (size_t i=0; i<nNodes(); i++)
		{
			buffer << vertices[i][0] << " " << 1 << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		delete[] active_basis_index;
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

	//mass matrix assembly implementation
	template <int Format_t>
	void CharmsQ1Mesh::make_mass_matrix(Eigen::SparseMatrix<double,Format_t> &mat) const
	{
		//pre-process to determine matrix structure so that the integration loop can be done in parallel
		using Triplet = Eigen::Triplet<double>;
		std::vector<Triplet> coo_structure;
		for (size_t e_idx=0; e_idx<elements.size(); e_idx++) //loop over active elements
		{
			const Element_t& ELEM = elements[e_idx];
			assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function
			for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
			{
				size_t basis_i = ELEM.basis_s[i];
				coo_structure.push_back(Triplet(basis_i, basis_i, 0)); //diagonal entry

				//check for interaction between basis functions on the same level
				for (int j=i+1; j<ELEM.cursor_basis_s; j++)
				{
					size_t basis_j = ELEM.basis_s[j];
					coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
					coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
				}

				//check for interaction with coarser basis functions
				for (int j=0; j<ELEM.cursor_basis_a; j++)
				{
					size_t basis_j = ELEM.basis_a[j];
					coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
					coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
				}
			}
		}

		//create matrix
		mat.setZero();
		mat.resize(basis.size(), basis.size());
		mat.setFromTriplets(coo_structure.begin(), coo_structure.end()); //all zeros, but structure is known

		//populate matrix
		for (size_t e_idx=0; e_idx<elements.size(); e_idx++) //loop over active elements
		{
			const Element_t& ELEM = elements[e_idx];
			assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function
			for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
			{
				size_t basis_i = ELEM.basis_s[i];
				//temporary: testing using mass lumped
				Point_t H = ELEM.H();
				double volume = H[0]*H[1]*H[2];
				mat.coeffRef(basis_i,basis_i) += volume*basis[basis_i].eval(ELEM.center())*basis[basis_i].eval(ELEM.center());

				//check for interaction between basis functions on the same level
				for (int j=i+1; j<ELEM.cursor_basis_s; j++)
				{
					size_t basis_j = ELEM.basis_s[j];
					double val = volume*basis[basis_i].eval(ELEM.center())*basis[basis_j].eval(ELEM.center());
					mat.coeffRef(basis_i,basis_j) += val;
					mat.coeffRef(basis_j,basis_i) += val;
				}

				//check for interaction with coarser basis functions
				for (int j=0; j<ELEM.cursor_basis_a; j++)
				{
					size_t basis_j = ELEM.basis_a[j];
					double val = volume*basis[basis_i].eval(ELEM.center())*basis[basis_j].eval(ELEM.center());
					mat.coeffRef(basis_i,basis_j) += val;
					mat.coeffRef(basis_j,basis_i) += val;
				}
			}
		}
	}


	//stiffness matrix assembly implementation
	template <int Format_t>
	void CharmsQ1Mesh::make_stiff_matrix(Eigen::SparseMatrix<double,Format_t> &mat) const
	{
		//pre-process to determine matrix structure so that the integration loop can be done in parallel
		using Triplet = Eigen::Triplet<double>;
		std::vector<Triplet> coo_structure;
		for (size_t e_idx=0; e_idx<elements.size(); e_idx++) //loop over active elements
		{
			const Element_t& ELEM = elements[e_idx];
			assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function
			for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
			{
				size_t basis_i = ELEM.basis_s[i];
				coo_structure.push_back(Triplet(basis_i, basis_i, 0)); //diagonal entry

				//check for interaction between basis functions on the same level
				for (int j=i+1; j<ELEM.cursor_basis_s; j++)
				{
					size_t basis_j = ELEM.basis_s[j];
					coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
					coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
				}

				//check for interaction with coarser basis functions
				for (int j=0; j<ELEM.cursor_basis_a; j++)
				{
					size_t basis_j = ELEM.basis_a[j];
					coo_structure.push_back(Triplet(basis_i, basis_j, 0)); //off-diagonal entry
					coo_structure.push_back(Triplet(basis_j, basis_i, 0)); //symmetric off-diagonal entry
				}
			}
		}

		//create matrix
		mat.setZero();
		mat.resize(basis.size(), basis.size());
		mat.setFromTriplets(coo_structure.begin(), coo_structure.end()); //all zeros, but structure is known

		//populate matrix
		for (size_t e_idx=0; e_idx<elements.size(); e_idx++) //loop over active elements
		{
			const Element_t& ELEM = elements[e_idx];
			assert(ELEM.cursor_basis_s>0); //any active element should be a support element for at least one basis function
			for (int i=0; i<ELEM.cursor_basis_s; i++) //loop over basis functions for which this element is a support element
			{
				size_t basis_i = ELEM.basis_s[i];
				//temporary: testing using mass lumped
				Point_t H = ELEM.H();
				double volume = H[0]*H[1]*H[2];
				mat.coeffRef(basis_i,basis_i) += volume*gv::util::dot(basis[basis_i].grad(ELEM.center()),basis[basis_i].grad(ELEM.center()));

				//check for interaction between basis functions on the same level
				for (int j=i+1; j<ELEM.cursor_basis_s; j++)
				{
					size_t basis_j = ELEM.basis_s[j];
					double val = volume*gv::util::dot(basis[basis_i].grad(ELEM.center()),basis[basis_j].grad(ELEM.center()));
					mat.coeffRef(basis_i,basis_j) += val;
					mat.coeffRef(basis_j,basis_i) += val;
				}

				//check for interaction with coarser basis functions
				for (int j=0; j<ELEM.cursor_basis_a; j++)
				{
					size_t basis_j = ELEM.basis_a[j];
					double val = volume*gv::util::dot(basis[basis_i].grad(ELEM.center()),basis[basis_j].grad(ELEM.center()));
					mat.coeffRef(basis_i,basis_j) += val;
					mat.coeffRef(basis_j,basis_i) += val;
				}
			}
		}
	}
}