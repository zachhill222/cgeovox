#pragma once

#include "charms/charmsQ1_util.hpp"
#include "geometry/assembly.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "util/octree_util.hpp"

#include <Eigen/SparseCore>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

namespace gv::charms
{
	template <class Assembly_t>
	class AssemblyCharmsQ1Mesh
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
		using MeshOpts      = gv::geometry::AssemblyMeshOptions;	
		using ScalarFun_t   = double (*)(Point_t); //function from Point_t to double
		using VectorFun_t   = Point_t (*)(Point_t); //function from Point_t to Point_t

		const Box_t   domain;
		VertexList_t  vertices;
		ElementList_t coarse_elements;
		ElementList_t fine_elements;
		BasisList_t   coarse_basis;
		BasisList_t   fine_basis;
		MeshOpts      opts;
		const Assembly_t&   assembly;
		std::vector<int> coarse_element_marker;

		//track indices of active basis functions and elements
		std::vector<size_t> coarse_basis_active2all;
		std::vector<size_t> coarse_basis_all2active;
		std::vector<size_t> coarse_elem_active2all;
		std::vector<size_t> coarse_elem_all2active;

		//scalar variables to maintain while refining (coefficients of active basis functions)
		std::vector<double> u; //fine basis
		std::vector<double> v; //fine basis
		std::vector<double> w; //fine basis
		std::vector<double> p; //coarse basis

		//constructor for creating a uniform mesh of a box
		AssemblyCharmsQ1Mesh(const Box_t &domain, const MeshOpts &opts, const Assembly_t &assembly) :
			domain(domain),
			vertices(domain),
			coarse_elements(domain),
			fine_elements(domain),
			coarse_basis(domain),
			fine_basis(domain),
			opts(opts),
			assembly(assembly)
		{
			//reserve space
			vertices.reserve((opts.N[0]+1)*(opts.N[1]+1)*(opts.N[2]+1));
			coarse_elements.reserve(opts.N[0]*opts.N[1]*opts.N[2]);
			coarse_basis.reserve(vertices.capacity());
			// fine_elements
			// fine_basis

			//construct coarse elements
			Point_t H = domain.sidelength() / Point_t(opts.N);
			for (size_t i=0; i<opts.N[0]; i++) {
				for (size_t j=0; j<opts.N[1]; j++) {
					for (size_t k=0; k<opts.N[2]; k++) {
						Point_t low  = domain.low() + Point_t{i,j,k} * H;
						Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;
						Box_t   bbox  {low, high};


						//check if the element should be marked as void, solid, or interface
						int marker;
						bool include_element;
						assembly.check_voxel(bbox, opts, marker, include_element);

						//add the element to the mesh
						if (include_element)
						{
							Element_t elem(&vertices, &coarse_elements, &coarse_basis, Box_t{low, high}); //this adds the appropriate vertices to the vertex list
							coarse_elements.push_back(elem); //add element to mesh
							coarse_element_marker.push_back(marker);
						}
					}
				}
			}

			//construct basis functions and add references to elements. in the coarsest mesh, each vertex corresponds to a basis function.
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFun_t fun(&vertices, &coarse_elements, &coarse_basis, v_idx);
				fun.set_support(); //TODO: speed this up or remove from here.

				size_t fun_idx = (size_t) -1;
				int flag = coarse_basis.push_back(fun, fun_idx);
				assert(flag==1);
				assert(coarse_basis[fun_idx].list_index == fun_idx);

				//add basis fucntion to basis_s of all support elements
				for (int i=0; i<coarse_basis[fun_idx].cursor_support; i++)
				{
					coarse_elements[coarse_basis[fun_idx].support[i]].insert_basis_s(fun_idx);
				}
			}
		}

		
		//convenient functions
		// size_t coarse_basis.size() const {return vertices.size();}
		// size_t coarse_elements.size() const {return coarse_elements.size();}
		// size_t coarse_basis.size() const {return coarse_basis.size();}

		//scalar field evaluations and assignments (call get_active_indices() ahead of time!)
		void _init_coarse_scalar_field(std::vector<double> &scalar, ScalarFun_t fun)
		{
			scalar.resize(coarse_basis_active2all.size());
			std::fill(scalar.begin(), scalar.end(), 0);

			//evaluate active depth 0 functions and calculate total depth
			size_t max_depth = 0;
			for (size_t i=0; i<coarse_basis_active2all.size(); i++)
			{
				const BasisFun_t& FUN = coarse_basis[coarse_basis_active2all[i]];
				assert(FUN.is_active);

				max_depth = std::max(max_depth,FUN.depth);
				if (FUN.depth==0)
				{
					scalar[i] = fun(FUN.coord());
					std::cout << FUN.coord() << " |--> " << scalar[i] << std::endl;
				}
			}

			//evaluate the remainder of the basis functions
			for (size_t d=1; d<=max_depth; d++)
			{
				for (size_t i=0; i<coarse_basis_active2all.size(); i++)
				{	
					const BasisFun_t& FUN = coarse_basis[coarse_basis_active2all[i]];
					assert(FUN.is_active);

					if (FUN.depth==d)
					{
						//TODO: replace _interpolate with a more efficient function (only need to evaluate at mesh vertices)
						assert(scalar[i]==0);
						scalar[i] = fun(FUN.coord()) - _interpolate_coarse_scalar_field(scalar, FUN.coord()); 
					}
				}
			}
		}

		double _interpolate_coarse_scalar_field(const std::vector<double> &scalar, const Point_t &point) const
		{
			//note the point is not required to be a mesh vertex

			//get coarse element(s) associated with the point and then get indices to active basis functions
			//whose support contains the point
			std::vector<size_t> coarse_elem_ind = coarse_elements.get_data_indices(point);
			std::vector<size_t> basis_ind;
			for (size_t i=0; i<coarse_elem_ind.size(); i++)
			{
				const Element_t& ELEM = coarse_elements[coarse_elem_ind[i]];
				if (ELEM.is_active and ELEM.contains(point))
				{
					//check ancestor and same basis functions
					for (int j=0; j<ELEM.cursor_basis_a; j++) {basis_ind.push_back(ELEM.basis_a[j]);}
					for (int j=0; j<ELEM.cursor_basis_s; j++) {basis_ind.push_back(ELEM.basis_s[j]);}
				}
			}

			//get unique basis indices to check
			std::sort(basis_ind.begin(), basis_ind.end());
			auto it = std::unique(basis_ind.begin(), basis_ind.end());
			basis_ind.resize(std::distance(basis_ind.begin(), it));
			assert(basis_ind.size() > 0);

			//evaluate active basis functions
			double result = 0;
			for (size_t i=0; i<basis_ind.size(); i++)
			{
				const BasisFun_t &FUN = coarse_basis[basis_ind[i]];
				if (FUN.is_active)
				{
					size_t active_idx = coarse_basis_all2active[basis_ind[i]];
					result += scalar[active_idx] * FUN.eval(point);
				}
			}

			return result;
		}

		void get_active_indices()
		{
			//coarse basis indices
			coarse_basis_active2all.clear();
			coarse_basis_active2all.reserve(coarse_basis.size());

			coarse_basis_all2active.clear();
			coarse_basis_all2active.resize(coarse_basis.size());

			for (size_t i=0; i<coarse_basis.size(); i++)
			{
				if (coarse_basis[i].is_active)
				{
					coarse_basis_all2active[i] = coarse_basis_active2all.size();
					coarse_basis_active2all.push_back(i);
				}
				else {coarse_basis_all2active[i] = (size_t) -1;}
			}

			//coarse element indices
			coarse_elem_active2all.clear();
			coarse_elem_active2all.reserve(coarse_elements.size());

			coarse_elem_all2active.clear();
			coarse_elem_all2active.resize(coarse_elements.size());

			for (size_t i=0; i<coarse_elements.size(); i++)
			{
				if (coarse_elements[i].is_active)
				{
					coarse_elem_all2active[i] = coarse_elem_active2all.size();
					coarse_elem_active2all.push_back(i);
				}
				else {coarse_elem_all2active[i] = (size_t) -1;}
			}
		}

		//file io
		void vtkprint(std::ostream &os) const; //write coarse mesh (as unstructured non-conforming voxels) in vtk format to specified stream
		void save_as(std::string filename) const; //save coarse mesh information to file, uses vtkprint()

		//hierarchical refinement
		void h_refine(const size_t basis_idx); //add detail functions
		void h_unrefine(const size_t basis_idx); //remove detail functions

		//quasi-hierarchical refinement
		int q_refine(const size_t basis_idx); //de-activate the specified basis function and activate its children
		void q_unrefine(const size_t basis_idx); //activate the specified basis function and de-activate its children
		void refine(const size_t element_idx); //refine all basis functions in basis_s of the specified element
	};


	//hierarchical refinement
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::h_refine(const size_t basis_idx)
	{
		assert(basis_idx<coarse_basis.size());
		
		//verify that this basis function can be refined
		assert(!coarse_basis[basis_idx].is_refined);

		while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
		while (coarse_elements.capacity() < coarse_elements.size()+64) {coarse_elements.reserve(2*coarse_elements.capacity());}
		while (coarse_basis.capacity() < coarse_basis.size()+27) {coarse_basis.reserve(2*coarse_basis.capacity());}
		
		//create candidate basis functions and elements
		coarse_basis[basis_idx].subdivide();
	
		for (int i=0; i<coarse_basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = coarse_basis[basis_idx].child[i];
			if (coarse_basis[c_idx].is_odd) {coarse_basis[c_idx].activate();}
		}

		//mark this basis function as refined
		coarse_basis[basis_idx].is_refined = true;
	}

	//hierarchical un-refinement
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::h_unrefine(const size_t basis_idx) //remove the detail functions
	{
		assert(basis_idx<coarse_basis.size());
		if (!coarse_basis[basis_idx].is_active) {return;}

		//verify that this basis function is allowed to be un-refined
		assert(coarse_basis[basis_idx].is_refined);

		//verify that no children functions are refined
		for (int i=0; i<coarse_basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = coarse_basis[basis_idx].child[i];
			assert(!coarse_basis[c_idx].is_refined);
			if (coarse_basis[c_idx].is_odd and coarse_basis[c_idx].is_active) {coarse_basis[c_idx].deactivate();}
		}

		//ensure all support elements are active
		for (int i=0; i<coarse_basis[basis_idx].cursor_support; i++)
		{
			size_t s_idx = coarse_basis[basis_idx].support[i];

			//check if there are any active descendent elements
			std::vector<size_t> desc_elems = coarse_elements[s_idx].descendent_elements();
			bool has_active_descendent = false;
			for (auto it=desc_elems.begin(); it!=desc_elems.end(); ++it)
			{
				has_active_descendent = coarse_elements[*it].is_active;
				if (has_active_descendent) {break;}
			}
			
			if (!has_active_descendent) {coarse_elements[s_idx].is_active = true;}
		}

		//mark this basis function as not refined
		coarse_basis[basis_idx].is_refined = false;
	}


	//quasi-hierarchical refinement (basis function)
	template <class Assembly_t>
	int AssemblyCharmsQ1Mesh<Assembly_t>::q_refine(const size_t basis_idx)
	{
		assert(basis_idx<coarse_basis.size());
		
		//verify that this basis function can be refined
		// assert(!coarse_basis[basis_idx].is_refined);
		if (coarse_basis[basis_idx].is_refined) {return 1;}

		while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
		while (coarse_elements.capacity() < coarse_elements.size()+64) {coarse_elements.reserve(2*coarse_elements.capacity());}
		while (coarse_basis.capacity() < coarse_basis.size()+27) {coarse_basis.reserve(2*coarse_basis.capacity());}
		
		//create candidate basis functions and elements
		std::vector<int> new_coarse_element_marker;
		coarse_basis[basis_idx].subdivide(assembly, opts, new_coarse_element_marker);
		
		//deactivate current basis
		coarse_basis[basis_idx].deactivate();

		for (int i=0; i<coarse_basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = coarse_basis[basis_idx].child[i];
			coarse_basis[c_idx].activate();
		}

		//mark this basis function as refined
		coarse_basis[basis_idx].is_refined = true;

		//track new element markers
		for (size_t i=0; i<new_coarse_element_marker.size(); i++) {coarse_element_marker.push_back(new_coarse_element_marker[i]);}
		assert(coarse_element_marker.size() == coarse_elements.size());
		return 0;
	}


	//quasi-hierarchical refinement (all basis functions in an element)
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::refine(const size_t element_idx)
	{
		// Element_t &ELEM = coarse_elements[element_idx];
		// std::cout << coarse_elements[element_idx].cursor_basis_s << std::endl;
		for (int j=0; j<coarse_elements[element_idx].cursor_basis_s; j++)
		{
			// std::cout << "refine basis " << coarse_elements[element_idx].basis_s[j] << " (" << j << "/" << coarse_elements[element_idx].cursor_basis_s << ")" << std::endl;
			q_refine(coarse_elements[element_idx].basis_s[j]);
		}
	}

	//quasi-hierarchical un-refinement
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::q_unrefine(const size_t basis_idx)
	{
		assert(basis_idx<coarse_basis.size());
		if (coarse_basis[basis_idx].is_active) {return;} //for quasi-hierarchical unrefinement, the "center" basis must be de-activated

		//verify that this basis function is allowed to be un-refined
		assert(coarse_basis[basis_idx].is_refined);

		//verify that no children functions are refined and de-activate all children
		for (int i=0; i<coarse_basis[basis_idx].cursor_child; i++)
		{
			size_t c_idx = coarse_basis[basis_idx].child[i];
			assert(!coarse_basis[c_idx].is_refined);
			if (coarse_basis[c_idx].is_active) {coarse_basis[c_idx].deactivate();}
		}

		//ensure all support elements are active
		for (int i=0; i<coarse_basis[basis_idx].cursor_support; i++)
		{
			size_t s_idx = coarse_basis[basis_idx].support[i];

			//check if there are any active descendent elements
			std::vector<size_t> desc_elems = coarse_elements[s_idx].descendent_elements();
			bool has_active_descendent = false;
			for (auto it=desc_elems.begin(); it!=desc_elems.end(); ++it)
			{
				has_active_descendent = coarse_elements[*it].is_active;
				if (has_active_descendent) {break;}
			}
			
			if (!has_active_descendent) {coarse_elements[s_idx].is_active = true;}
		}

		//mark this basis function as not refined and activate it
		coarse_basis[basis_idx].is_refined = false;
		coarse_basis[basis_idx].activate();
	}

	
	//vtkprint implementation
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::vtkprint(std::ostream &os) const
	{
		//ensure the active basis and element lists are up to date
		// get_active_indices();


		//write to buffer and flush buffer to the stream
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << vertices.size() << " float\n";
		for (size_t i=0; i<vertices.size(); i++) { buffer << vertices[i] << "\n";}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENTS
		buffer << "CELLS " << coarse_elements.size() << " " << (1+8)*coarse_elements.size() << "\n";
		for (size_t i=0; i<coarse_elements.size(); i++)
		{
			buffer << 8 << " "; //8 nodes per element
			for (size_t j=0; j<8; j++)
			{
				buffer << coarse_elements[i].node[j] << " ";
			}
			buffer << "\n";
		}
		buffer << "\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VTK IDs
		buffer << "CELL_TYPES " << coarse_elements.size() << "\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {buffer << coarse_elements[i].vtk_id << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH CELL
		buffer << "CELL_DATA " << coarse_elements.size() << "\n";
		buffer << "FIELD mesh_element_info 8\n";

		//ELEMENT INDEX
		buffer << "index 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++)
		{
			assert(coarse_elements[i].list_index == i);
			buffer << coarse_elements[i].list_index << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT ACTIVE INDEX
		buffer << "active_index 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++)
		{
			if (coarse_elements[i].is_active) {buffer << coarse_elem_all2active[i] << " ";}
			else {buffer << "-1 ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT TYPE MARKER
		buffer << "marker 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {buffer << coarse_element_marker[i] << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT DEPTH
		buffer << "depth 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {buffer << coarse_elements[i].depth << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT ACTIVE MARKER
		buffer << "is_active 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {buffer << coarse_elements[i].is_active << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT IS_SUBDIVIDED MARKER
		buffer << "is_subdivided 1 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {buffer << coarse_elements[i].is_subdivided << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_S
		buffer << "basis_s 8 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {to_stream(buffer, coarse_elements[i].basis_s, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "basis_a[0:7] 8 " << coarse_elements.size() << " integer\n";
		for (size_t i=0; i<coarse_elements.size(); i++) {to_stream(buffer, coarse_elements[i].basis_a, 8);}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");




		//MESH INFORMATION AT EACH VERTEX
		//map vertex index to active basis index
		std::vector<size_t> vertex2active_basis;
		vertex2active_basis.resize(vertices.size());
		for (size_t i=0; i<vertices.size(); i++)
		{
			std::vector<size_t> basis_total = coarse_basis.get_data_indices(vertices[i]);
			int count = 0;
			for (auto it=basis_total.begin(); it!=basis_total.end(); ++it)
			{
				if (coarse_basis[*it].is_active and coarse_basis[*it].coord()==vertices[i])
				{
					count++;
					vertex2active_basis[i] = coarse_basis_all2active[*it];
					assert(vertex2active_basis[i]<coarse_basis_active2all.size());
				}
			}
			assert(count<=1);
			if (count==0) {vertex2active_basis[i] = (size_t) -1;}
		}





		buffer << "POINT_DATA " << vertices.size() << "\n";
		buffer << "FIELD mesh_vertex_info 6\n";

		//VERTEX ACTIVE BASIS DEPTH
		buffer << "index 1 " << vertices.size() << " integer\n";
		for (size_t i=0; i<vertices.size(); i++)
		{	
			size_t active_basis_idx = vertex2active_basis[i];
			if (active_basis_idx==(size_t) -1) {buffer << "-1 ";}
			else {buffer << coarse_basis[coarse_basis_active2all[active_basis_idx]].depth << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS INDEX
		buffer << "active_index 1 " << vertices.size() << " integer\n";
		for (size_t i=0; i<vertices.size(); i++)
		{	
			size_t active_basis_idx = vertex2active_basis[i];
			if (active_basis_idx==(size_t) -1) {buffer << "-1 ";}
			else {buffer << active_basis_idx << " ";}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS SUPPORT
		buffer << "support 8 " << vertices.size() << " integer\n";
		for (size_t i=0; i<vertices.size(); i++)
		{	
			size_t active_basis_idx = vertex2active_basis[i];
			if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = coarse_basis[coarse_basis_active2all[active_basis_idx]];
				to_stream(buffer, FUN.support, 8);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS PARENTS
		buffer << "parent 27 " << vertices.size() << " integer\n";
		for (size_t i=0; i<vertices.size(); i++)
		{	
			size_t active_basis_idx = vertex2active_basis[i];
			if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = coarse_basis[coarse_basis_active2all[active_basis_idx]];
				to_stream(buffer, FUN.parent, 27);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//VERTEX ACTIVE BASIS CHILDREN
		buffer << "child 27 " << vertices.size() << " integer\n";
		for (size_t i=0; i<vertices.size(); i++)
		{	
			size_t active_basis_idx = vertex2active_basis[i];
			if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
			else
			{
				const BasisFun_t& FUN = coarse_basis[coarse_basis_active2all[active_basis_idx]];
				to_stream(buffer, FUN.child, 27);
			}
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VERTEX TEST DATA
		if (p.size() == coarse_basis_active2all.size())
		{
			buffer << "p 1 " << vertices.size() << " float\n";
			for (size_t i=0; i<vertices.size(); i++)
			{
				buffer << _interpolate_coarse_scalar_field(p, vertices[i]) << " ";
			}
		}
		else
		{
			buffer << "test_data 1 " << vertices.size() << " float\n";
			for (size_t i=0; i<vertices.size(); i++)
			{
				buffer << 0 << " ";
			}
		}
		
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");
	}

	//save mesh implementation
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::save_as(std::string filename) const
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