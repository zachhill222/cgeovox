//These classes are helper classes to implement the CHARMS
//(Conforming, Hierarchical, Adaptive Refinement Methods)
//method for finite elements.
//Paper: "CHARMS: A Simple Framework for Adaptive Simulation" (2002)
//Authors: Eitan Grinspun, Petr Krysl, Peter Schroder



#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"
#include "util/point_octree.hpp"

#include <vector>
#include <algorithm>
#include <cassert>

#include <sstream>
#include <iostream>
#include <fstream>

namespace gv::fem
{
	//classes defined in this file
	class ElementQ1; //logic for tracking voxel elements
	class ElementQ1Octree; //container for voxel elements (built from gv::util::BasicOctree)
	class BasisFunctionQ1; //logic for tracking basis functions
	class BasisFunctionQ1Octree; //container for basis functions (build from gv::util::BasicOctree)
	class CharmsQ1Mesh; //main class used for interacting with the CHARMS mesh (Q1 voxel elements in 3D)


	//////////////// ELEMENT IMPLEMENTATION ////////////////
	class ElementQ1
	{
	public:
		//common typedefs
		using VertexList_t = gv::util::PointOctree<3>;
		using Index_t      = gv::util::Point<3,size_t>;
		using Point_t      = gv::util::Point<3,double>;
		using Box_t        = gv::util::Box<3>;

		//constructor for coarsest mesh
		ElementQ1(const Box_t& root_bbox, VertexList_t* vertices, bool add_vertices=true) : depth(0), index{0,0,0}, root_bbox(root_bbox), vertices(vertices)
		{
			//add vertices if necessary and find their indices. add these indices nodes[]
			for (int k=0; k<8; k++)
			{
				Point_t vertex = root_bbox.voxelvertex(k);
				size_t idx;
				if (add_vertices) //allow this construction to add vertices if needed
				{
					int flag = vertices->push_back(vertex, idx);
					assert(flag!=-1); //ensure node was added succesfully
				}
				else //use add_vertices=false if the element should have already been constructed
				{
					idx = vertices->find(vertex);
					assert(idx!=(size_t) -1); //ensure vertex was found
				}
				
				nodes[k] = idx;
			}
		}

		//constructor for refinements, refering only to the coarsest mesh
		ElementQ1(const int depth, const Index_t& index, const Box_t& root_bbox, VertexList_t* vertices, bool add_vertices=true) : depth(depth), index(index), root_bbox(root_bbox), vertices(vertices)
		{
			//get bounding box for this element
			Point_t H = std::pow(0.5,depth) * root_bbox.sidelength();
			Point_t low = root_bbox.low() + H * Point_t(index);
			Box_t bbox(low, low+H);
			
			//convert index triplet to vertices and find their indices. add to vertices if necessary.
			for (int k=0; k<8; k++)
			{
				Point_t vertex = bbox.voxelvertex(k);
				size_t idx;
				if (add_vertices) //allow this construction to add vertices if needed
				{
					int flag = vertices->push_back(vertex, idx);
					assert(flag!=-1); //ensure node was added succesfully
				}
				else //use add_vertices=false if the element should have already been constructed
				{
					idx = vertices->find(vertex);
					assert(idx!=(size_t) -1); //ensure vertex was found
				}

				nodes[k] = idx;
			}
		}

		const int depth; //number of divisions of the root element required to reach this element
		const Index_t index; //index to easily refer to elements which may or may not have been instantiated yet
		const Box_t root_bbox; //element in the coarsest mesh that this element is a subdivision of
		VertexList_t* vertices; //pointer to list of vertices

		size_t nodes[8]; //mesh information. indices of nodes that make this element (vtk voxel ordering)
		static const int vtk_id = 11; //all elements are vtk voxels
		bool is_active = false; //rather than deleting elements for un-refinement, set is_active to false
		bool is_refined = false; //set to true when this element has been subdivided
		std::vector<size_t> basis_a; //indices of active ancestor basis functions whose support overlaps with this element. TODO: change to set?
		std::vector<size_t> basis_s; //indices of active same-level basis functions whose support overlaps with this element. TODO: change to set?
		std::vector<size_t> ancestors; //elements that this element is a subdivision of (all ancestors should not be active)

		bool contains(const Point_t& coord) const {return (*vertices)[nodes[0]]<=coord and coord<=(*vertices)[nodes[7]];} //check if the element contains the specified coordinate
		Point_t H() const {return (*vertices)[nodes[7]]-(*vertices)[nodes[0]];}
		Point_t centroid() const {return 0.5*( (*vertices)[nodes[0]] + (*vertices)[nodes[7]]);} //return centroid of this element, used for octree storage
		bool operator==(const ElementQ1 &other) const //compare two elements by their nodes. used for octree storage
		{
			if (vertices!=other.vertices) {return false;} //elements must use the same vertex list
			for (int k=0; k<8; k++) {if (nodes[k]!=other.nodes[k]) {return false;} } //elements must have the same node indices
			return true;
		}
	};

	//OCTREE BASED STORAGE CONTAINER FOR A LIST OF ELEMENTS
	class ElementQ1Octree : public gv::util::BasicOctree<ElementQ1, 3, true, 16>
	{
	public:
		ElementQ1Octree(const gv::util::Box<3> &domain) : gv::util::BasicOctree<ElementQ1, 3, true, 16>(domain) {}
		std::vector<size_t> get_elements_containing_coordinate(const gv::util::Point<3,double> &coord) const
		{
			std::vector<size_t> result;
			std::vector<const Node*> octree_nodes = getnodes(coord); //get nodes of octree that must contain any elements containing the specified coordinate
			
			for (size_t idx=0; idx<octree_nodes.size(); idx++)
			{
				for (size_t i=0; i<octree_nodes[idx]->cursor; i++) //check which elements contain the specified coordinate
				{
					size_t elem_idx = octree_nodes[idx]->data_idx[i];
					// std::cout << "\telement " << elem_idx << " contains " << coord << " : " << _data[elem_idx].contains(coord) << std::endl;
					if (_data[elem_idx].contains(coord)) {result.push_back(elem_idx);}
				}
			}
			// assert(result.size()>0); //if the coordinate is in the domain, then it is contained in at least one element
			return result;
		}
	private:
		bool is_data_valid(const gv::util::Box<3> &box, const ElementQ1 &data) const override {return box.contains(data.centroid());}
	};



	//////////////// BASIS FUNCTION IMPLEMENTATION ////////////////
	class BasisFunctionQ1
	{
	public:
		//common typedefs
		using VertexList_t = gv::util::PointOctree<3>;
		using Point_t      = gv::util::Point<3,double>;

		//construct basis function by reference to the vertex index. support managemement must be done outside of this class.
		BasisFunctionQ1(VertexList_t* vertices, const size_t node_idx, const int depth) :
			vertices(vertices), node_idx(node_idx), depth(depth) {assert(node_idx<(*vertices).size());}

		VertexList_t* vertices; //pointer to list of vertices

		const size_t node_idx; //this basis function "lives" at this vertex
		const int depth; //number of divisions of the root element required to reach this basis function. must be equall to the depth of all elements in its support.
		std::vector<size_t> support; //natural support of this basis function, the elements must be active and on the same depth as this basis function
		bool is_active = false; //rather than deleting basis functions on un-refinement, set is_active to false
		bool is_refined = false; //track if this basis function has been refined
		Point_t coord() const {return (*vertices)[node_idx];} //get coordinate where this basis function "lives". used for octree storage
		bool operator==(const BasisFunctionQ1& other) const //determine if two basis functions are the same. used in constructing octrees
		{
			if (vertices!=other.vertices) {return false;} //ensure both basis functions refer to the same list of vertices
			if (depth!=other.depth) {return false;}
			if (node_idx!=other.node_idx) {return false;}
			return true; //same vertex location and depth
		}
	};

	//OCTREE BASED STORAGE CONTAINER FOR A LIST OF BASIS FUNCTIONS
	class BasisFunctionQ1Octree : public gv::util::BasicOctree<BasisFunctionQ1, 3, false, 16>
	{
	public:
		BasisFunctionQ1Octree(const gv::util::Box<3> &domain) : gv::util::BasicOctree<BasisFunctionQ1, 3, false, 16>(domain) {}
		std::vector<size_t> get_basis_functions_with_coordinate(const gv::util::Point<3,double> &coord) const
		{
			std::vector<size_t> result;
			std::vector<const Node*> octree_nodes = getnodes(coord); //get nodes of octree that must contain any elements containing the specified coordinate
			
			for (size_t idx=0; idx<octree_nodes.size(); idx++)
			{
				for (size_t i=0; i<octree_nodes[idx]->cursor; i++) //check which elements contain the specified coordinate
				{
					size_t basis_idx = octree_nodes[idx]->data_idx[i];
					// std::cout << "\telement " << basis_idx << " contains " << coord << " : " << _data[basis_idx].contains(coord) << std::endl;
					if (_data[basis_idx].coord()==coord) {result.push_back(basis_idx);}
				}
			}
			return result;
		}
	private:
		bool is_data_valid(const gv::util::Box<3> &box, const BasisFunctionQ1 &data) const override {return box.contains(data.coord());}
	};



	/////////////// CHARMS Q1 MESH IMPLEMENTATION /////////////////
	class CharmsQ1Mesh
	{
	public:
		//basic typedefs
		using Index_t         = gv::util::Point<3,size_t>;
		using Point_t         = gv::util::Point<3,double>;
		using Box_t           = gv::util::Box<3>;
		using Element_t       = ElementQ1;
		using BasisFunction_t = BasisFunctionQ1;

		//constructor for meshing a box domain with equal-sized elements as the coarsest mesh
		CharmsQ1Mesh(const Box_t &domain, const Index_t &N) : vertices(domain), elements(domain), basis(domain)
		{
			//reserve space
			vertices.reserve((N[0]+1)*(N[1]+1)*(N[2]+1));
			elements.reserve(N[0]*N[1]*N[2]);
			basis.reserve(vertices.capacity());

			//construct elements
			Point_t H = domain.sidelength() / Point_t(N);
			for (size_t i=0; i<N[0]; i++)
			{
				for (size_t j=0; j<N[1]; j++)
				{
					for (size_t k=0; k<N[2]; k++)
					{
						Point_t low  = domain.low() + Point_t{i,j,k} * H;
						Point_t high = domain.low() + Point_t{i+1,j+1,k+1} * H;
						Element_t elem(Box_t(low,high), &vertices); //this adds the appropriate vertices to the vertex list
						elem.is_active = true; //coarsest level elements are all active when initialized
						elements.push_back(elem); //add element to mesh
					}
				}
			}

			//construct basis functions and add references to elements. in the coarsest mesh, each vertex corresponds to a basis function.
			for (size_t v_idx=0; v_idx<vertices.size(); v_idx++)
			{
				BasisFunction_t fun(&vertices, v_idx, 0);
				fun.is_active = true; //all basis functions on the coarsest level are initially active
				fun.support = elements.get_elements_containing_coordinate(fun.coord()); //there is only one level in the current mesh
				int flag = basis.push_back(fun); assert(flag==1);
				for (size_t idx=0; idx<fun.support.size(); idx++) {elements[fun.support[idx]].basis_s.push_back(v_idx);} //the vertex index and basis function index are the same in the current mesh
			}
		}

		//storage
		gv::util::PointOctree<3> vertices; //list of vertices used in the mesh
		ElementQ1Octree elements; //list of elements (both active and inactive)
		BasisFunctionQ1Octree basis; //list of basis functions (both active and inactive)

		//refinement operators
		void refine_element(const size_t elem_idx); //split an element. this marks the specified element as inactive and its children as active. should not be called directly.
		void refine_basis(const size_t basis_idx); //primary method to refine the mesh. elements are subdivided if necessary.

		//convenient references
		size_t nNodes() const {return vertices.size();} //number of nodes in the mesh
		size_t nElems() const {return elements.size();} //number of elements (active or otherwise) in the mesh
		size_t nBasis() const {return basis.size();} //number of basis functions (active or otherwise) in the mesh

		//file io
		void vtkprint(std::ostream &os) const; //write mesh (as unstructured non-conforming voxels) in vtk format to specified stream
		void save_as(std::string filename) const; //save mesh information to file, uses vtkprint()
	};


	//refine_element implementation. does not create new basis functions
	void CharmsQ1Mesh::refine_element(const size_t elem_idx)
	{

		std::cout << "\n\n=== refine element: " << elem_idx << " ===" << std::endl;

		assert(elem_idx<nElems()); //the element index must have a valid index
		assert(!elements[elem_idx].is_refined); //the specified element must not have been previously refined

		//re-sizing mid refinement is bad (pointers must be updated)
		bool is_resized = false;
		if (elements.capacity() < elements.size()+8) {elements.reserve(2*elements.size()); is_resized=true;}
		if (vertices.capacity() < vertices.size()+7) {vertices.reserve(2*vertices.size()); is_resized=true;}
		if (basis.capacity() < basis.size()+7) {basis.reserve(2*basis.size()); is_resized=true;}
		if (is_resized) //must update references after vectors are resized
		{
			for (size_t idx=0; idx<nElems(); idx++) {elements[idx].vertices = &vertices;} //update vertex list reference in elements
			for (size_t idx=0; idx<nBasis(); idx++) {basis[idx].vertices = &vertices;} //update vertex list reference in basis functions
		}
		
		//update the element to be subdivided
		elements[elem_idx].is_active = false; //de-activate the element being refined
		elements[elem_idx].is_refined = true; //mark the element as having been refined

		//create children elements and add to element list
		for (int i=0; i<2; i++)
		{
			for (int j=0; j<2; j++)
			{
				for (int k=0; k<2; k++)
				{
					Element_t elem(elements[elem_idx].depth+1, 2*elements[elem_idx].index+Index_t{i,j,k}, elements[elem_idx].root_bbox, &vertices); //this adds the appropriate new vertex
					elem.is_active = true; //new elements are active
					elem.ancestors = elements[elem_idx].ancestors; elem.ancestors.push_back(elem_idx); //make list of ancestor elements
					for (size_t basis_idx=0; basis_idx<elements[elem_idx].basis_a.size(); basis_idx++) {elem.basis_a.push_back(elements[elem_idx].basis_a[basis_idx]);} //add ancestor basis functions
					for (size_t basis_idx=0; basis_idx<elements[elem_idx].basis_s.size(); basis_idx++) {elem.basis_a.push_back(elements[elem_idx].basis_s[basis_idx]);} //add ancestor basis functions

					elements.push_back(elem); //add elements to mesh
				}
			}
		}
	}

	//hierarchical basis refinement.
	void CharmsQ1Mesh::refine_basis(const size_t basis_idx)
	{
		std::cout << "\n\n=== refine basis: " << basis_idx << " ===" << std::endl;
		std::cout << "\tsupport= [ ";
		for (size_t idx=0; idx<basis[basis_idx].support.size(); idx++) {std::cout << basis[basis_idx].support[idx] << " ";}
		std::cout << "]" << std::endl;

		assert(basis[basis_idx].is_active); //only refine active basis functions?
		assert(!basis[basis_idx].is_refined);

		//mark basis as refined
		basis[basis_idx].is_active = true; //mark the basis function as active?
		basis[basis_idx].is_refined = true;
		
		//subdivide support elements if necessary
		for (size_t idx=0; idx<basis[basis_idx].support.size(); idx++) //refine support elements if needed
		{
			size_t elem_idx = basis[basis_idx].support[idx];
			if (!elements[elem_idx].is_refined) {refine_element(elem_idx);}
		}

		//loop through the detail basis functions ("odd" vertices of the subdivided support elements)
		//there are 26 such vertices if the specified basis function has all 8 support elements (i.e., it corresponds to an interior vertex)
		size_t support_elem = basis[basis_idx].support[0];
		Point_t H = 0.5*elements[support_elem].H(); //stepsize, use the first support element for convenience
		for (int i=-1; i<2; i++)
		{
			for (int j=-1; j<2; j++)
			{
				for (int k=-1; k<2; k++)
				{
					if (i==0 and j==0 and k==0) {continue;} //this is the location of the node that is being hierarchically refined. nothing to do.

					Point_t vertex = basis[basis_idx].coord() + H * Point_t{i,j,k};
					size_t v_idx = vertices.find(vertex);
					if (v_idx<vertices.size()) //this is a valid "odd" vertex in the domain
					{
						//initialize a new basis function
						BasisFunction_t fun(&vertices, v_idx, basis[basis_idx].depth+1); //depth is not strictly needed. used for comparing basis functions.
						fun.is_active = true;

						//check if basis function already exists
						size_t existing_basis_idx = basis.find(fun);
						if (existing_basis_idx < nBasis())
						{
							// TODO: make activation routine
							std::cout << "\nbasis " << existing_basis_idx << " already exists" << std::endl;
							assert(basis[existing_basis_idx].is_active); //keep until activation routine is made
							continue;
						}


						std::cout << "\nnew basis index= " << nBasis() << std::endl;
						std::cout << "basis coord= " << fun.coord() << std::endl;
						std::cout << "basis depth= " << fun.depth << std::endl;

						//set the support for the new basis function
						std::vector<size_t> elem_list = elements.get_elements_containing_coordinate(fun.coord());
						for (size_t idx=0; idx<elem_list.size(); idx++)
						{
							size_t elem_idx = elem_list[idx];
							std::cout << "\t\telem_idx= " << elem_idx << " depth= " << elements[elem_idx].depth << std::endl;
							if (elements[elem_idx].depth==fun.depth) {fun.support.push_back(elem_idx);}
						}

						std::cout << "elem_list= [ ";
						for (size_t idx=0; idx<elem_list.size(); idx++) {std::cout << elem_list[idx] << " ";}
						std::cout << "]" << std::endl;

						std::cout << "support= [ ";
						for (size_t idx=0; idx<fun.support.size(); idx++) {std::cout << fun.support[idx] << " ";}
						std::cout << "]" << std::endl;
						
						// if (fun.support.size()==0) //the support should never be 0 here. do linear search. TODO: remove this.
						// {
						// 	for (size_t idx=0; idx<nElems(); idx++)
						// 	{
						// 		if (elements[idx].depth == fun.depth and elements[idx].contains(fun.coord())) {fun.support.push_back(idx);}
						// 	}
						// 	std::cout << "re-check support (linear search)\n";
						// 	std::cout << "support= [ ";
						// 	for (size_t idx=0; idx<fun.support.size(); idx++) {std::cout << fun.support[idx] << " ";}
						// 	std::cout << "]" << std::endl;
						// }

						assert(fun.support.size()>1); //can only be 1 if fun.coord() is the corner of the coarsest mesh. this cannot happen in a hierarchical refinement.

						//add basis function to list
						size_t new_basis_idx;
						int flag = basis.push_back(fun, new_basis_idx);
						assert(flag!=-1); //ensure that this basis function is in the list (it is possible that a single basis function was previously created)

						//add new basis to constructed elements
						for (size_t idx=0; idx<elem_list.size(); idx++)
						{
							size_t elem_idx = elem_list[idx];
							if (elements[elem_idx].depth==fun.depth) //should be in basis_s
							{
								if (std::find(elements[elem_idx].basis_s.begin(), elements[elem_idx].basis_s.end(), elem_idx) == elements[elem_idx].basis_s.end()) //basis_s does not contain elem_idx
								{
									elements[elem_idx].basis_s.push_back(elem_idx);
								}
							}
							else if (elements[elem_idx].depth<fun.depth) //should be in basis_a
							{
								if (std::find(elements[elem_idx].basis_a.begin(), elements[elem_idx].basis_a.end(), elem_idx) == elements[elem_idx].basis_a.end()) //basis_a does not contain elem_idx
								{
									elements[elem_idx].basis_a.push_back(elem_idx);
								}
							}
						}
					}
				}
			}
		}
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
				buffer << elements[i].nodes[j] << " ";
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



		//MESH INFORMATION AT EACH CELL (depth, index, is_active, global_index, #basis_a, #basis_s)
		buffer << "CELL_DATA " << nElems() << "\n";
		buffer << "FIELD mesh_element_info 6\n";

		//ELEMENT DEPTH
		buffer << "depth 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].depth << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT INDEX
		buffer << "index 3 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].index << " ";}
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
		buffer << "elem_index 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << i << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "#basis_a 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].basis_a.size() << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");

		//ELEMENT BASIS_A
		buffer << "#basis_s 1 " << nElems() << " integer\n";
		for (size_t i=0; i<nElems(); i++) {buffer << elements[i].basis_s.size() << " ";}
		buffer << "\n\n";
		os << buffer.rdbuf();
		buffer.str("");



		//MESH INFORMATION AT EACH VERTEX (#basis_total, active_basis_index, active_basis_depth)
		buffer << "POINT_DATA " << nNodes() << "\n";
		buffer << "FIELD mesh_vertex_info 3\n";

		//loop through vertices to collect info
		size_t basis_total_size[nNodes()] {0};
		size_t active_basis_index[nNodes()] {0};
		size_t active_basis_depth[nNodes()] {0};
		for (size_t i=0; i<nNodes(); i++)
		{
			std::vector<size_t> basis_total = basis.get_basis_functions_with_coordinate(vertices[i]);
			basis_total_size[i] = basis_total.size();
			int count = 0;
			for (size_t idx=0; idx<basis_total.size(); idx++)
			{
				if (basis[basis_total[idx]].is_active)
				{
					count++;
					active_basis_index[i] = basis_total[idx];
					active_basis_depth[i] = basis[basis_total[idx]].depth;
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
			if (active_basis_depth[i]<nBasis()) {buffer << active_basis_depth[i] << " ";}
			else {buffer << -1 << " ";}
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