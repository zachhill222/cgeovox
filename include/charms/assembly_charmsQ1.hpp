#pragma once

#include "charms/charmsQ1_util.hpp"
#include "geometry/assembly.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/point_octree.hpp"
#include "util/octree_util.hpp"

#include <Eigen/SparseCore>

#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>

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
		using ScalarFun_t   = std::function<double(Point_t)>;  //function from Point_t to double
		using VectorFun_t   = std::function<Point_t(Point_t)>; //function from Point_t to Point_t

		const Box_t   domain;
		VertexList_t  vertices;
		ElementList_t elements;
		BasisList_t   basis;
		MeshOpts      opts;
		const Assembly_t& assembly;
		std::vector<int> element_marker;
		std::vector<double> max_dist; //track the maximum distance from a point in the 

		//track indices of active basis functions and elements
		std::vector<size_t> basis_active2all;
		std::vector<size_t> basis_all2active;
		std::vector<size_t> elem_active2all;
		std::vector<size_t> elem_all2active;
		std::vector<double> coarse_heaviside_vector;
		bool active_indices_up_to_date = false;

		//scalar variables to maintain while refining (coefficients of active basis functions)
		//the length of these vectors should be the same as the total basis size (active+inactive)
		// std::vector<double> p; //coarse basis

		//constructor for creating a uniform mesh of a box
		AssemblyCharmsQ1Mesh(const Box_t &domain, const MeshOpts &opts, const Assembly_t &assembly) :
			domain(domain),
			vertices(domain),
			elements(domain),
			basis(domain),
			opts(opts),
			assembly(assembly)
		{
			//reserve space
			vertices.reserve((opts.N[0]+1)*(opts.N[1]+1)*(opts.N[2]+1));
			elements.reserve(opts.N[0]*opts.N[1]*opts.N[2]);
			basis.reserve(vertices.capacity());

			//construct elements
			Point_t H = domain.sidelength() / Point_t(opts.N);
			for (size_t i=0; i<opts.N[0]; i++) {
				for (size_t j=0; j<opts.N[1]; j++) {
					for (size_t k=0; k<opts.N[2]; k++) {
						Point_t low  = domain.low() + Point_t{i,j,k} * H;
						Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;
						Box_t   bbox  {low, high};


						//check if the element should be marked as void, solid, or interface
						int marker=opts.interface_marker;
						bool include_element=true;
						assembly.check_voxel(bbox, opts, marker, include_element);

						//add the element to the mesh
						if (include_element)
						{
							Element_t elem(&vertices, &elements, &basis, Box_t{low, high}); //this adds the appropriate vertices to the vertex list
							elements.push_back(elem); //add element to mesh
							element_marker.push_back(marker);
						}
					}
				}
			}

			//construct basis functions and add references to elements. in the coarsest mesh, each vertex corresponds to a basis function.
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFun_t fun(&vertices, &elements, &basis, v_idx);
				fun.set_support(); //TODO: speed this up or remove from here.

				size_t fun_idx = (size_t) -1;
				[[maybe_unused]] int flag = basis.push_back(fun, fun_idx);
				assert(flag==1);
				assert(basis[fun_idx].list_index == fun_idx);

				//add basis fucntion to basis_s of all support elements
				for (int i=0; i<basis[fun_idx].cursor_support; i++)
				{
					elements[basis[fun_idx].support[i]].insert_basis_s(fun_idx);
				}

				//all basis functions should be active in the coarsest mesh
				basis[fun_idx].activate(assembly, opts);
			}
		}

		double _coarse_heaviside(const Point_t& point) const
		{
			std::vector<size_t> e_idx = elements.get_data_indices(point);
			assert(e_idx.size()>0);

			for (size_t i=0; i<e_idx.size(); i++)
			{
				const Element_t& ELEM = elements[e_idx[i]];
				if (element_marker[ELEM.list_index] == opts.solid_marker) {return 0;}
				else if (element_marker[ELEM.list_index] == opts.void_marker) {return 1;}
			}
			return assembly.heaviside(point, elements[e_idx[0]].bbox().sidelength()[0]);
		}

		void _init_coarse_heaviside()
		{
			coarse_heaviside_vector.resize(basis.size());
			std::fill(coarse_heaviside_vector.begin(), coarse_heaviside_vector.end(), 0);
			ScalarFun_t heaviside = [this](const Point_t& point) {return this->_coarse_heaviside(point);};
			_init_scalar_field(coarse_heaviside_vector, heaviside);
		}

		double coarse_heaviside(const Point_t& point) const
		{
			return _interpolate_scalar_field(coarse_heaviside_vector, point);
		}

		//scalar field evaluations and assignments (call get_active_indices() ahead of time!)
		void _init_scalar_field(std::vector<double> &scalar, ScalarFun_t fun)
		{
			assert(active_indices_up_to_date);
			assert(scalar.size() == basis.size());

			//evaluate active depth 0 functions and calculate total depth
			size_t max_depth = 0;
			for (size_t i=0; i<basis_active2all.size(); i++)
			{
				const BasisFun_t& FUN = basis[basis_active2all[i]];
				assert(FUN.is_active);

				max_depth = std::max(max_depth,FUN.depth);
				if (FUN.depth==0)
				{
					scalar[basis_active2all[i]] = fun(FUN.coord());
					// std::cout << FUN.coord() << " |--> " << scalar[i] << std::endl;
				}
			}

			//evaluate the remainder of the basis functions
			for (size_t d=1; d<=max_depth; d++)
			{
				for (size_t i=0; i<basis_active2all.size(); i++)
				{	
					const BasisFun_t& FUN = basis[basis_active2all[i]];
					assert(FUN.is_active);

					if (FUN.depth==d)
					{
						//TODO: replace _interpolate with a more efficient function (only need to evaluate at mesh vertices)
						assert(scalar[FUN.list_index]==0);
						scalar[FUN.list_index] = fun(FUN.coord()) - _interpolate_scalar_field(scalar, FUN.coord()); 
					}
				}
			}
		}

		template<typename Vector_t>
		double _interpolate_scalar_field(const Vector_t &scalar, const Point_t &point) const
		{
			//note the point is not required to be a mesh vertex

			//get element(s) associated with the point and then get indices to active basis functions
			//whose support contains the point
			std::vector<size_t> coarse_elem_ind = elements.get_data_indices(point);
			std::vector<size_t> basis_ind;
			for (size_t i=0; i<coarse_elem_ind.size(); i++)
			{
				const Element_t& ELEM = elements[coarse_elem_ind[i]];
				if (ELEM.is_active and ELEM.contains(point))
				{
					//check ancestor and same basis functions
					for (int j=0; j<ELEM.cursor_basis_a; j++)
					{
						if (basis[ELEM.basis_a[j]].is_active)
						{
							basis_ind.push_back(ELEM.basis_a[j]);
						}
					}

					for (int j=0; j<ELEM.cursor_basis_s; j++)
					{
						if (basis[ELEM.basis_s[j]].is_active)
						{
							basis_ind.push_back(ELEM.basis_s[j]);
						}
					}
				}
			}

			//get unique basis indices to check
			std::sort(basis_ind.begin(), basis_ind.end());
			auto it = std::unique(basis_ind.begin(), basis_ind.end());
			basis_ind.resize(std::distance(basis_ind.begin(), it));
			// assert(basis_ind.size() > 0);

			//evaluate active basis functions
			double result = 0;
			for (size_t i=0; i<basis_ind.size(); i++)
			{
				const BasisFun_t &FUN = basis[basis_ind[i]];
				if (FUN.is_active)
				{
					// size_t active_idx = basis_all2active[basis_ind[i]];
					result += scalar[basis_ind[i]] * FUN.eval(point);
				}
			}

			return result;
		}

		void get_active_indices()
		{
			if (active_indices_up_to_date) {return;}
			active_indices_up_to_date = true;

			//basis indices
			basis_active2all.clear();
			basis_active2all.reserve(basis.size());

			basis_all2active.clear();
			basis_all2active.resize(basis.size());

			for (size_t i=0; i<basis.size(); i++)
			{
				if (basis[i].is_active)
				{
					basis_all2active[i] = basis_active2all.size();
					basis_active2all.push_back(i);
				}
				else {basis_all2active[i] = (size_t) -1;}
			}

			//element indices
			elem_active2all.clear();
			elem_active2all.reserve(elements.size());

			elem_all2active.clear();
			elem_all2active.resize(elements.size());

			for (size_t i=0; i<elements.size(); i++)
			{
				if (elements[i].is_active)
				{
					elem_all2active[i] = elem_active2all.size();
					elem_active2all.push_back(i);
				}
				else {elem_all2active[i] = (size_t) -1;}
			}
		}

		//get interior boundary (total basis index, not active basis index)
		std::vector<size_t> active_basis_interior_boundary() const
		{
			assert(active_indices_up_to_date);

			std::vector<size_t> result;

			//loop over active interface elements
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				const Element_t& ELEM = elements[elem_active2all[e_idx]];
				if (element_marker[ELEM.list_index] != opts.interface_marker) {continue;}

				//check basis_s
				for (int i=0; i<ELEM.cursor_basis_s; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_s[i]];
					if (!FUN.is_active) {continue;}

					int n_active = 0;
					for (int j=0; j<FUN.cursor_support; j++)
					{
						if (elements[FUN.support[j]].is_active) {n_active++;}
					}
					if (n_active < 8) {result.push_back(FUN.list_index);}
				}

				//check basis_a
				for (int i=0; i<ELEM.cursor_basis_a; i++)
				{
					const BasisFun_t& FUN = basis[ELEM.basis_a[i]];
					if (!FUN.is_active) {continue;}
					int n_active = 0;
					for (int j=0; j<FUN.cursor_support; j++)
					{
						if (elements[FUN.support[j]].is_active) {n_active++;}
					}
					if (n_active < 8) {result.push_back(FUN.list_index);}
				}
			}

			return result;
		}

		//get volume of active elements
		double volume() const
		{
			assert(active_indices_up_to_date);

			double result = 0;
			#pragma omp parallel for reduction(+:result)
			for (size_t e_idx=0; e_idx<elem_active2all.size(); e_idx++)
			{
				const Element_t& ELEM = elements[elem_active2all[e_idx]];
				assert(ELEM.is_active);
				const Point_t H = ELEM.H();
				result += H[0]*H[1]*H[2];
			}

			return result;
		}

		//file io
		void vtkprint(std::ostream &os, const std::vector<double>& scalar_field = std::vector<double>(), double eps=-1 ) const; //write mesh (as unstructured non-conforming voxels) in vtk format to specified stream
		void save_as(std::string filename, const std::vector<double>& scalar_field = std::vector<double>(), double eps=-1) const; //save mesh information to file, uses vtkprint()

		//quasi-hierarchical refinement
		int q_refine(const size_t basis_idx, std::vector<double>& scalar_field); //de-activate the specified basis function and activate its children
		void refine(const size_t element_idx, std::vector<double>& scalar_field); //refine all basis functions in basis_s of the specified element
	};


	//quasi-hierarchical refinement (basis function)
	template <class Assembly_t>
	int AssemblyCharmsQ1Mesh<Assembly_t>::q_refine(const size_t basis_idx, std::vector<double>& scalar_field)
	{
		assert(scalar_field.size() == basis.size());
		assert(basis_idx<basis.size());
		
		active_indices_up_to_date = false;

		//verify that this basis function can be refined
		// assert(!basis[basis_idx].is_refined);
		if (basis[basis_idx].is_refined) {return 1;}

		while (vertices.capacity() < vertices.size()+125) {vertices.reserve(2*vertices.capacity());}
		while (elements.capacity() < elements.size()+64) {elements.reserve(2*elements.capacity());}
		while (basis.capacity() < basis.size()+27) {basis.reserve(2*basis.capacity());}
		
		//convenient references
		BasisFun_t& FUN = basis[basis_idx];

		//create candidate basis functions and elements
		std::vector<int> new_element_marker;
		FUN.subdivide(assembly, opts, new_element_marker);
		
		//deactivate current basis function
		FUN.deactivate();

		scalar_field.resize(basis.size());
		for (int i=0; i<FUN.cursor_child; i++)
		{
			BasisFun_t& CHILD = basis[FUN.child[i]];
			CHILD.activate(assembly, opts); //activate all valid support elements and this function (if it has at least one valid support element)
			if (CHILD.is_active) {scalar_field[CHILD.list_index] += scalar_field[FUN.list_index]*FUN.eval(CHILD.coord());}
		}

		//mark this basis function as refined
		FUN.is_refined = true;

		//track new element markers
		for (size_t i=0; i<new_element_marker.size(); i++) {element_marker.push_back(new_element_marker[i]);}
		assert(element_marker.size() == elements.size());

		return 0;
	}


	//quasi-hierarchical refinement (all basis functions in an element)
	//extend the coefficients in scalar_field so that it can be interpolated in the refined basis
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::refine(const size_t element_idx, std::vector<double>& scalar_field)
	{
		active_indices_up_to_date = false;

		// Element_t &ELEM = elements[element_idx];
		// std::cout << elements[element_idx].cursor_basis_s << std::endl;
		for (int j=0; j<elements[element_idx].cursor_basis_s; j++)
		{
			// std::cout << "refine basis " << elements[element_idx].basis_s[j] << " (" << j << "/" << elements[element_idx].cursor_basis_s << ")" << std::endl;
			q_refine(elements[element_idx].basis_s[j], scalar_field);
		}
	}

	
	//vtkprint implementation
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::vtkprint(std::ostream &os, const std::vector<double>& scalar_field, double eps) const
	{
		//ensure the active basis and element lists are up to date
		assert(active_indices_up_to_date);


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
		buffer << "CELLS " << elements.size() << " " << (1+8)*elements.size() << "\n";
		for (size_t i=0; i<elements.size(); i++)
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
		buffer << "CELL_TYPES " << elements.size() << "\n";
		for (size_t i=0; i<elements.size(); i++) {buffer << elements[i].vtk_id << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH CELL
		buffer << "CELL_DATA " << elements.size() << "\n";
		buffer << "FIELD mesh_element_info 2\n";

		//ELEMENT INDEX
		// buffer << "index 1 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++)
		// {
		// 	assert(elements[i].list_index == i);
		// 	buffer << elements[i].list_index << " ";
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT ACTIVE INDEX
		// buffer << "active_index 1 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++)
		// {
		// 	if (elements[i].is_active) {buffer << elem_all2active[i] << " ";}
		// 	else {buffer << "-1 ";}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT IS_ACTIVE
		buffer << "is_active 1 " << elements.size() << " integer\n";
		for (size_t i=0; i<elements.size(); i++)
		{
			buffer << elements[i].is_active << " ";
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT TYPE MARKER
		buffer << "marker 1 " << elements.size() << " integer\n";
		for (size_t i=0; i<elements.size(); i++) {buffer << element_marker[i] << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT DEPTH
		// buffer << "depth 1 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++) {buffer << elements[i].depth << " ";}
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT PARENT
		// buffer << "parent 1 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++)
		// {
		// 	if (elements[i].parent < elements.size())
		// 	{
		// 		buffer << elements[i].parent << " ";
		// 	}
		// 	else {buffer << "-1 ";}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT CHILDREN
		// buffer << "children 8 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++)
		// {
		// 	for (int j=0; j<elements[i].cursor_child; j++)
		// 	{
		// 		buffer << elements[i].child[j] << " ";
		// 	}

		// 	for (int j=elements[i].cursor_child; j<8; j++)
		// 	{
		// 		buffer << "-1 ";
		// 	}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT IS_SUBDIVIDED MARKER
		// buffer << "is_subdivided 1 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++) {buffer << elements[i].is_subdivided << " ";}
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT BASIS_S
		// buffer << "basis_s 8 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++) {to_stream(buffer, elements[i].basis_s, 8);}
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//ELEMENT BASIS_A
		// buffer << "basis_a[0:7] 8 " << elements.size() << " integer\n";
		// for (size_t i=0; i<elements.size(); i++) {to_stream(buffer, elements[i].basis_a, 8);}
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");




		//MESH INFORMATION AT EACH VERTEX
		//map vertex index to active basis index
		// std::vector<size_t> vertex2active_basis;
		// vertex2active_basis.resize(vertices.size());
		// for (size_t i=0; i<vertices.size(); i++)
		// {
		// 	std::vector<size_t> basis_total = basis.get_data_indices(vertices[i]);
		// 	int count = 0;
		// 	for (auto it=basis_total.begin(); it!=basis_total.end(); ++it)
		// 	{
		// 		if (basis[*it].is_active and basis[*it].coord()==vertices[i])
		// 		{
		// 			count++;
		// 			vertex2active_basis[i] = basis_all2active[*it];
		// 			assert(vertex2active_basis[i]<basis_active2all.size());
		// 		}
		// 	}
		// 	assert(count<=1);
		// 	if (count==0) {vertex2active_basis[i] = (size_t) -1;}
		// }





		buffer << "POINT_DATA " << vertices.size() << "\n";
		buffer << "FIELD mesh_vertex_info 3\n";

		//VERTEX ACTIVE BASIS DEPTH
		// buffer << "depth 1 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 ";}
		// 	else {buffer << basis[basis_active2all[active_basis_idx]].depth << " ";}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX BASIS INDEX
		// buffer << "index 1 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 ";}
		// 	else {buffer << basis_active2all[active_basis_idx] << " ";}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX ACTIVE BASIS INDEX
		// buffer << "active_index 1 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 ";}
		// 	else {buffer << active_basis_idx << " ";}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX ACTIVE BASIS SUPPORT
		// buffer << "support 8 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 ";}
		// 	else
		// 	{
		// 		const BasisFun_t& FUN = basis[basis_active2all[active_basis_idx]];
		// 		to_stream(buffer, FUN.support, 8);
		// 	}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX ACTIVE BASIS PARENTS
		// buffer << "parent 27 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
		// 	else
		// 	{
		// 		const BasisFun_t& FUN = basis[basis_active2all[active_basis_idx]];
		// 		to_stream(buffer, FUN.parent, 27);
		// 	}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX ACTIVE BASIS CHILDREN
		// buffer << "child 27 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	size_t active_basis_idx = vertex2active_basis[i];
		// 	if (active_basis_idx==(size_t) -1) {buffer << "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ";}
		// 	else
		// 	{
		// 		const BasisFun_t& FUN = basis[basis_active2all[active_basis_idx]];
		// 		to_stream(buffer, FUN.child, 27);
		// 	}
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		//VERTEX SIGNED DISTANCE TO ASSEMBLY
		// double distance[vertices.size()] {0};

		// for (size_t i=0; i<elem_active2all.size(); i++)
		// {
		// 	const Element_t& ELEM = elements[elem_active2all[i]];
		// 	if (element_marker[i]==opts.solid_marker)
		// 	{
		// 		for (int j=0; j<ELEM.cursor_node; j++) {distance[ELEM.node[j]] = -1;}
		// 	}
		// 	else if (element_marker[i]==opts.void_marker)
		// 	{
		// 		for (int j=0; j<ELEM.cursor_node; j++) {distance[ELEM.node[j]] = 1;}
		// 	}
		// }

		// for (size_t i=0; i<elem_active2all.size(); i++)
		// {
		// 	const Element_t& ELEM = elements[elem_active2all[i]];
		// 	if (element_marker[i]==opts.interface_marker)
		// 	{
		// 		for (int j=0; j<ELEM.cursor_node; j++) {
		// 			size_t v_idx = ELEM.node[j];
		// 			distance[v_idx] = assembly.signed_distance(vertices[v_idx]);
		// 			// std::cout << "v_idx= " << v_idx << ": " << vertices[v_idx] << "\t " << distance[v_idx] << std::endl;
		// 		}
		// 	}
		// }


		float heaviside[vertices.size()];
		float dirac[vertices.size()];

		if (eps<0) {eps = 0.03125 * gv::util::norm2(domain.sidelength());}

		#pragma omp parallel for
		for (size_t i=0; i<vertices.size(); i++)
		{
			heaviside[i] = assembly.heaviside(vertices[i], eps);
			dirac[i] = assembly.dirac_delta(vertices[i], eps);
		}

		buffer << "heaviside 1 " << vertices.size() << " float\n";
		for (size_t i=0; i<vertices.size(); i++)
		{
			if (std::isnan(heaviside[i])) {buffer << -1 << " "; }
			else {buffer << heaviside[i] << " "; }
			
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		// buffer << "coarse_heaviside 1 " << vertices.size() << " float\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {
		// 	buffer << coarse_heaviside(vertices[i]) << " ";
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");

		buffer << "dirac_delta 1 " << vertices.size() << " float\n";
		for (size_t i=0; i<vertices.size(); i++)
		{
			if (std::isnan(dirac[i])) {buffer << -1 << " "; }
			else {buffer << dirac[i] << " "; }
		}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");


		//VERTEX TEST DATA
		if (scalar_field.size() == basis.size())
		{
			buffer << "u 1 " << vertices.size() << " float\n";
			for (size_t i=0; i<vertices.size(); i++)
			{
				buffer << (float) _interpolate_scalar_field(scalar_field, vertices[i]) << " ";
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
		
		//VERTEX BOUNDARY INDICATOR
		// std::vector<size_t> boundary_basis = active_basis_interior_boundary();
		// int boundary_vertex[vertices.size()] {0};
		// for (size_t i=0; i<boundary_basis.size(); i++)
		// {
		// 	const BasisFun_t& FUN = basis[boundary_basis[i]];
		// 	boundary_vertex[FUN.node_index] = 1;
		// }



		// buffer << "boundary 1 " << vertices.size() << " integer\n";
		// for (size_t i=0; i<vertices.size(); i++)
		// {	
		// 	buffer << boundary_vertex[i] << " ";
		// }
		// buffer << "\n\n";
		// os << buffer.rdbuf();
		// buffer.str("");
	}

	//save mesh implementation
	template <class Assembly_t>
	void AssemblyCharmsQ1Mesh<Assembly_t>::save_as(std::string filename, const std::vector<double>& scalar_field, double eps) const
	{
		//open and check file
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}

		//print mesh to file
		vtkprint(meshfile, scalar_field, eps);
		meshfile.close();
	}


	//print mesh info
	template <class Assembly_t>
	std::ostream& operator<<(std::ostream& os, const AssemblyCharmsQ1Mesh<Assembly_t>& mesh)
	{
		assert(mesh.active_indices_up_to_date);
		os << "n_vertices= " << mesh.vertices.size() << std::endl;
		os << "n_basis_functions= " << mesh.basis.size() << std::endl;
		os << "n_active_basis_functions= " << mesh.basis_active2all.size() << std::endl;
		os << "n_elements= " << mesh.elements.size() << std::endl;
		os << "n_active_elements= " << mesh.elem_active2all.size() << std::endl;

		return os;
	}
}