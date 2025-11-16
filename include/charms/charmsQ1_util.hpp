#pragma once


#include "charms/charms_util.hpp"
#include "mesh/vtk_voxel.hpp"

#include "geometry/assembly.hpp"

#include "util/point_octree.hpp"
#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>

#define CHARMS_Q1_BASIS_SUPPORT_SIZE 8
#define CHARMS_Q1_BASIS_CHILD_SIZE   125
#define CHARMS_Q1_BASIS_PARENT_SIZE  27

#define CHARMS_Q1_ELEMENT_NODE_SIZE    8
#define CHARMS_Q1_ELEMENT_CHILD_SIZE   8
#define CHARMS_Q1_ELEMENT_BASIS_S_SIZE 8

#define CHARMS_Q1_BASIS_OCTREE_DATA_PER_LEAF 64
#define CHARMS_Q1_ELEMENT_OCTREE_DATA_PER_LEAF 64

namespace gv::charms
{
	//constructors for basis functions and elements may need to refer to eachother
	class CharmsQ1BasisFun;
	class CharmsQ1Element;
	class CharmsQ1BasisFunOctree;
	class CharmsQ1ElementOctree;
	
	using CharmsQ1BasisFun_BASE = CharmsBasisFun<CHARMS_Q1_BASIS_SUPPORT_SIZE,CHARMS_Q1_BASIS_CHILD_SIZE,CHARMS_Q1_BASIS_PARENT_SIZE>;
	using CharmsQ1Element_BASE  = CharmsElement<CHARMS_Q1_ELEMENT_NODE_SIZE,CHARMS_Q1_ELEMENT_CHILD_SIZE,CHARMS_Q1_ELEMENT_BASIS_S_SIZE>;
	

	class CharmsQ1BasisFun : public CharmsQ1BasisFun_BASE
	{
	public:
		using VertexList_t = gv::util::PointOctree<3>;
		using Point_t      = gv::util::Point<3,double>;
		using Box_t        = gv::util::Box<3>;

		//pointers to lists of vertices and elements.
		VertexList_t*           vertices;
		CharmsQ1ElementOctree*  elements;
		CharmsQ1BasisFunOctree* basis;
		
		size_t list_index = CharmsQ1BasisFun_BASE::NO_DATA; //track where this basis function occurs in the basis function list. set by octree container.

		//default constructor
		constexpr CharmsQ1BasisFun() : CharmsQ1BasisFun_BASE(), vertices(nullptr), elements(nullptr), basis(nullptr) {}

		//constructor for creating a new basis function in the coarsest mesh. elements[] must be up to date.
		//sets is_active and is_odd to true
		CharmsQ1BasisFun(VertexList_t* vertices, CharmsQ1ElementOctree* elements, CharmsQ1BasisFunOctree* basis, const size_t node_index) :
			CharmsQ1BasisFun_BASE(node_index, 0, true, true, false, false),
			vertices(vertices),
			elements(elements),
			basis(basis) {}

		//constructor for creating a new basis function using information from a parent basis function
		//parent[], child[], and support[] are initialized but only filled with NO_DATA (i.e. (size_t) -1)
		CharmsQ1BasisFun(const CharmsQ1BasisFun& some_parent, const size_t node_index, const bool is_odd):
			CharmsQ1BasisFun_BASE(node_index, some_parent.depth+1, is_odd, false, false, false),
			vertices(some_parent.vertices),
			elements(some_parent.elements),
			basis(some_parent.basis) {}

		~CharmsQ1BasisFun()
		{
			vertices = nullptr;
			elements = nullptr;
			basis    = nullptr;
		}

		//move constructor
		CharmsQ1BasisFun(CharmsQ1BasisFun&& other) noexcept :
			CharmsQ1BasisFun_BASE(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			list_index(other.list_index)
		{
			other.vertices = nullptr;
			other.elements = nullptr;
			other.basis    = nullptr;
		}

		//copy constructor
		CharmsQ1BasisFun(const CharmsQ1BasisFun& other) :
			CharmsQ1BasisFun_BASE(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			list_index(other.list_index){}

		//update move operator
		CharmsQ1BasisFun& operator=(CharmsQ1BasisFun&& other) noexcept
		{
			if (this!=&other)
			{
				CharmsBasisFun::operator=(other);
				vertices = other.vertices; other.vertices = nullptr;
				elements = other.elements; other.elements = nullptr;
				basis    = other.basis;    other.basis    = nullptr;
				list_index = other.list_index;
			}
			return *this;	
		}

		//update copy assignment
		CharmsQ1BasisFun& operator=(const CharmsQ1BasisFun& other)
		{
			CharmsBasisFun::operator=(other);
			vertices = other.vertices;
			elements = other.elements; 
			basis    = other.basis;    
			list_index = other.list_index;
			
			return *this;
		}

		//populate the support of this basis function. the elements[] list must be updated first.
		int set_support();
		bool in_support(const Point_t& coord) const; //check if a point is in the support of this basis function

		//add this function to the appropriate basis_s and basis_a of the elements.
		void update_element_basis_lists();

		//add detail functions of this basis function. divides elements as needed. adds new basis functions to the basis list.
		//does not change is_active and sets is_active=false for the new basis functions.
		void subdivide();
		template <class Particle_t, size_t n_data>
		void subdivide(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts, std::vector<int> &new_markers);

		//activate a basis function and any necessary elements
		void activate();

		template <class Particle_t, size_t n_data>
		void activate(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts);

		//deactivate a basis function and any unused elements
		void deactivate();

		inline Point_t coord() const {return (*vertices)[this->node_index];}
		Box_t bbox() const; //get bounding box for the support of this element

		double  eval(const Point_t &point) const; //evaluate this basis function
		Point_t grad(const Point_t &point) const; //evaluate the gradient of this function
	};


	
	class CharmsQ1Element : public CharmsQ1Element_BASE
	{
	public:
		using VertexList_t = CharmsQ1BasisFun::VertexList_t;
		using Point_t      = gv::util::Point<3,double>;
		using Box_t        = gv::util::Box<3>;

		//pointers to lists of vertices and elements.
		VertexList_t*           vertices;
		CharmsQ1ElementOctree*  elements;
		CharmsQ1BasisFunOctree* basis;
		
		size_t list_index = CharmsQ1Element_BASE::NO_DATA; //track where this element occurs in the element list. set by octree container.
		static constexpr int vtk_id = 11;

		//default constructor
		constexpr CharmsQ1Element() : CharmsQ1Element_BASE(), vertices(nullptr), elements(nullptr), basis(nullptr) {}

		//constructor for creating the coarsest mesh. adds vertices to the list, but does not add the element or create any basis functions
		//sets is_active=true
		CharmsQ1Element(VertexList_t* vertices, CharmsQ1ElementOctree* elements, CharmsQ1BasisFunOctree* basis, const Box_t &bbox) :
			CharmsQ1Element_BASE(0, (size_t)-1, CHARMS_Q1_ELEMENT_BASIS_S_SIZE, true, false),
			vertices(vertices),
			elements(elements),
			basis(basis)
		{
			size_t node_idx = (size_t) -1;
			for (int i=0; i<8; i++)
			{
				vertices->push_back(bbox.voxelvertex(i), node_idx);
				this->insert_node(node_idx);
			}
		}

		//constructor for creating the coarsest mesh from an existing list of vertices. does not add element, vertices, or basis functions to the lists.
		//sets is_active=true
		CharmsQ1Element(VertexList_t* vertices, CharmsQ1ElementOctree* elements, CharmsQ1BasisFunOctree* basis, const size_t (&elem_nodes)[8]) :
			CharmsQ1Element_BASE(0, (size_t)-1, CHARMS_Q1_ELEMENT_BASIS_S_SIZE, true, false),
			vertices(vertices),
			elements(elements),
			basis(basis)
		{
			for (int i=0; i<8; i++)
			{
				this->insert_node(elem_nodes[i]);
			}
		}



		//constructor for creating a new element using information from its parent. adds vertices to the list if necessary.
		//sets is_active=false
		CharmsQ1Element(const CharmsQ1Element& parent, int sibling_number) :
			CharmsQ1Element_BASE(parent.depth+1, parent.list_index, parent.capacity_basis_a, false, false),
			vertices(parent.vertices),
			elements(parent.elements),
			basis(parent.basis)
		{
			for (int i=0; i<parent.cursor_basis_s; i++) {this->insert_basis_a(parent.basis_s[i]);}
			for (int i=0; i<parent.cursor_basis_a; i++) {this->insert_basis_a(parent.basis_a[i]);}

			Box_t bbox((*vertices)[parent.node[sibling_number]], parent.center());
			size_t node_idx = (size_t) -1;
			for (int i=0; i<8; i++)
			{
				[[maybe_unused]] int flag = vertices->push_back(bbox.voxelvertex(i), node_idx);
				assert(flag!=-1);
				assert(node_idx != (size_t) -1);
				this->insert_node(node_idx);
			}
		}

		~CharmsQ1Element()
		{
			vertices = nullptr;
			elements = nullptr;
			basis    = nullptr;
		}

		//move constructor
		CharmsQ1Element(CharmsQ1Element&& other) noexcept :
			CharmsQ1Element_BASE(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			list_index(other.list_index)
		{
			other.vertices = nullptr;
			other.elements = nullptr;
			other.basis    = nullptr;
		}

		//copy constructor
		CharmsQ1Element(const CharmsQ1Element& other) :
			CharmsQ1Element_BASE(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			list_index(other.list_index){}

		//update move operator
		CharmsQ1Element& operator=(CharmsQ1Element&& other) noexcept
		{
			if (this!=&other)
			{
				CharmsElement::operator=(other);
				vertices = other.vertices; other.vertices = nullptr;
				elements = other.elements; other.elements = nullptr;
				basis    = other.basis;    other.basis    = nullptr;
				list_index = other.list_index;
			}
			return *this;	
		}

		//update copy assignment
		CharmsQ1Element& operator=(const CharmsQ1Element& other)
		{
			CharmsElement::operator=(other);
			vertices = other.vertices; 
			elements = other.elements; 
			basis    = other.basis;    
			list_index = other.list_index;
			
			return *this;
		}


		//split an element. adds vertices and elements to their lists, but does not create any basis functions.
		//does not change the is_active marker of any element or basis function
		void subdivide();
		template <class Particle_t, size_t n_data>
		void subdivide(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts, std::vector<int> &new_markers); //

		//get all ancestor basis functions (i.e. union of all basis_s and basis_a for all ancestor elements)
		std::vector<size_t> ancestor_basis_fun() const;

		inline Box_t bbox() const {return Box_t{(*vertices)[node[0]], (*vertices)[node[7]]};}
		inline Point_t center() const {return 0.5*((*vertices)[node[0]] + (*vertices)[node[7]]);}
		inline Point_t H() const {return (*vertices)[node[7]] - (*vertices)[node[0]];}
		inline bool contains(const Point_t &coord) const {return (*vertices)[node[0]] <= coord and coord <= (*vertices)[node[7]];}

		std::vector<size_t> ancestor_elements() const
		{
			std::vector<size_t> result;
			result.reserve(this->depth);
			_ancestor_elements(result);
			return result;
		}

		std::vector<size_t> descendent_elements() const
		{
			std::vector<size_t> result;
			_descendent_elements(result);
			return result;
		}

	private:
		void _ancestor_elements(std::vector<size_t> &result) const;
		void _descendent_elements(std::vector<size_t> &result) const;
	};


	//Octree definitions
	class CharmsQ1BasisFunOctree : public gv::util::BasicOctree_Point<CharmsQ1BasisFun,3,CHARMS_Q1_BASIS_OCTREE_DATA_PER_LEAF>
	{
	public:
		CharmsQ1BasisFunOctree(const gv::util::Box<3> &domain) : 
			gv::util::BasicOctree_Point<CharmsQ1BasisFun,3,CHARMS_Q1_BASIS_OCTREE_DATA_PER_LEAF>(domain,64) {}

		int push_back(CharmsQ1BasisFun &val)
		{
			size_t idx = -1;
			int result = gv::util::BasicOctree_Point<CharmsQ1BasisFun,3,CHARMS_Q1_BASIS_OCTREE_DATA_PER_LEAF>::push_back(std::move(val), idx);
			(*this)[idx].list_index = idx;
			return result;
		}

		int push_back(CharmsQ1BasisFun &val, size_t &idx)
		{
			int result = gv::util::BasicOctree_Point<CharmsQ1BasisFun,3,CHARMS_Q1_BASIS_OCTREE_DATA_PER_LEAF>::push_back(std::move(val), idx);
			(*this)[idx].list_index = idx;
			return result;
		}


	private:
		bool is_data_valid(const gv::util::Box<3> &box, const CharmsQ1BasisFun &data) const override {return box.contains(data.coord());}
	};


	class CharmsQ1ElementOctree : public gv::util::BasicOctree_Vol<CharmsQ1Element,3,CHARMS_Q1_ELEMENT_OCTREE_DATA_PER_LEAF>
	{
	public:
		CharmsQ1ElementOctree(const gv::util::Box<3> &domain) : 
			gv::util::BasicOctree_Vol<CharmsQ1Element,3,CHARMS_Q1_ELEMENT_OCTREE_DATA_PER_LEAF>(domain,64) {}

		int push_back(CharmsQ1Element &val)
		{
			size_t idx = -1;
			int result = gv::util::BasicOctree_Vol<CharmsQ1Element,3,CHARMS_Q1_ELEMENT_OCTREE_DATA_PER_LEAF>::push_back(std::move(val), idx);
			(*this)[idx].list_index = idx;
			return result;
		}

		int push_back(CharmsQ1Element &val, size_t &idx)
		{
			int result = gv::util::BasicOctree_Vol<CharmsQ1Element,3,CHARMS_Q1_ELEMENT_OCTREE_DATA_PER_LEAF>::push_back(std::move(val), idx);
			(*this)[idx].list_index = idx;
			return result;
		}

	private:
		bool is_data_valid(const gv::util::Box<3> &box, const CharmsQ1Element &data) const override {return box.intersects(data.bbox());}
	};


	//implement methods for CharmsQ1BasisFun
	int CharmsQ1BasisFun::set_support()
	{
		std::vector<size_t> check_elements_idx = elements->get_data_indices(this->coord());
		for (auto it=check_elements_idx.begin(); it!=check_elements_idx.end(); ++it)
		{
			CharmsQ1Element& ELEM = (*elements)[*it];
			if (ELEM.depth==this->depth and ELEM.contains(this->coord())) {this->insert_support(*it);}
		}
		// assert(this->cursor_support>0);
		return this->cursor_support;
	}

	bool CharmsQ1BasisFun::in_support(const Point_t& coord) const
	{
		for (int i=0; i<cursor_support; i++)
		{
			if ((*elements)[support[i]].contains(coord)) {return true;}
		}
		return false;
	}

	void CharmsQ1BasisFun::update_element_basis_lists()
	{
		assert(list_index!=this->NO_DATA);
		Box_t support_bbox = bbox();

		//get a list of all elements that intersect the support
		std::vector<size_t> check_elements_idx = elements->get_data_indices(support_bbox);
		for (auto it=check_elements_idx.begin(); it!=check_elements_idx.end(); ++it)
		{
			CharmsQ1Element& ELEM = (*elements)[*it];
			if (ELEM.bbox().contains(this->coord()))
			{
				if (this->depth==ELEM.depth) {ELEM.insert_basis_s(this->list_index);}
				else if (this->depth<ELEM.depth) {ELEM.insert_basis_a(this->list_index);}
			}
		}
	}

	void CharmsQ1BasisFun::subdivide()
	{
		if (this->is_subdivided) {return;}
		this->is_subdivided = true;

		//subdivide each support element if necessary
		for (int i=0; i<this->cursor_support; i++)
		{
			CharmsQ1Element& ELEM = (*elements)[this->support[i]];
			ELEM.subdivide();
		}

		//create detail basis functions (up to 27)
		Point_t H = 0.5*(*elements)[this->support[0]].H();
		for (int i=-1; i<2; i++){
			for (int j=-1; j<2; j++){
				for (int k=-1; k<2; k++){

					//get coordinate and index for the location of the new basis function
					Point_t new_coord = this->coord() + H * Point_t{i,j,k};
					size_t  new_coord_idx = vertices->find(new_coord);
					if(new_coord_idx >= vertices->size()) //happens near domain boundary
					{
						// Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
						// if (domain.contains(new_coord)) {std::cout << "WARNING: could not find vertex at " << new_coord << std::endl;}
						continue;
					}

					//initialize the new basis function
					bool new_fun_is_odd = true;
					if (i==0 and j==0 and k==0) {new_fun_is_odd=false;}
					CharmsQ1BasisFun fun(*this, new_coord_idx, new_fun_is_odd);
					for (int depth0_idx=0; depth0_idx<this->cursor_support; depth0_idx++)
					{
						const CharmsQ1Element& ELEM0 = (*elements)[this->support[depth0_idx]];
						for (int depth1_idx=0; depth1_idx<ELEM0.cursor_child; depth1_idx++)
						{
							const CharmsQ1Element& ELEM1 = (*elements)[ELEM0.child[depth1_idx]];
							if (ELEM1.contains(fun.coord()))
							{
								fun.insert_support(ELEM1.list_index);
							}
						}
					}
					// assert(fun.cursor_support>0);
					if (fun.cursor_support==0) {std::cout<< fun << std::endl;}

					//add the new basis function to the list
					size_t new_list_index = (size_t) -1;
					int flag = basis->push_back(fun, new_list_index);
					if (flag==0) {continue;} //it is possible that the basis function already exists
					assert(flag==1); //we should only alter newly created functions here

					CharmsQ1BasisFun& CHILD = (*basis)[new_list_index];
					assert(CHILD.list_index==basis->size()-1);
					assert(CHILD.list_index==new_list_index);
					
					//add the new basis to basis_s list of all support elements
					for (int l=0; l<CHILD.cursor_support; l++)
					{
						size_t s_idx = CHILD.support[l];
						(*elements)[s_idx].insert_basis_s(new_list_index);
					}


					//update parents and children
					//support[i].basis_s[j] is a sibling (or this element) and could be a parent of this function's children
					for (int i=0; i<this->cursor_support; i++)
					{
						const CharmsQ1Element& ELEM = (*elements)[this->support[i]];
						for (int j=0; j<ELEM.cursor_basis_s; j++)
						{
							CharmsQ1BasisFun& SIBLING = (*basis)[ELEM.basis_s[j]];
							if (SIBLING.in_support(CHILD.coord()))
							{
								CHILD.insert_parent(SIBLING.list_index);
								SIBLING.insert_child(CHILD.list_index);
							}
						}
					}
				}
			}
		}
	}

	template <class Particle_t, size_t n_data>
	void CharmsQ1BasisFun::subdivide(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts, std::vector<int> &new_markers)
	{
		if (this->is_subdivided) {return;}
		this->is_subdivided = true;

		//subdivide each support element if necessary
		for (int i=0; i<this->cursor_support; i++)
		{
			CharmsQ1Element& ELEM = (*elements)[this->support[i]];
			ELEM.subdivide(assembly, opts, new_markers);
			// std::cout << "subdivide support element " << i << "/" << this->cursor_support << ": " << new_markers.size() << " new elements" << std::endl;
		}

		//create detail basis functions (up to 27)
		Point_t H = 0.5*(*elements)[this->support[0]].H();
		for (int i=-1; i<2; i++){
			for (int j=-1; j<2; j++){
				for (int k=-1; k<2; k++){

					//get coordinate and index for the location of the new basis function
					Point_t new_coord = this->coord() + H * Point_t{i,j,k};
					size_t  new_coord_idx = vertices->find(new_coord);
					if(new_coord_idx >= vertices->size()) //happens near domain boundary
					{
						// Box_t domain(Point_t{0,0,0}, Point_t{1,1,1});
						// if (domain.contains(new_coord)) {std::cout << "WARNING: could not find vertex at " << new_coord << std::endl;}
						continue;
					}

					//initialize the new basis function
					bool new_fun_is_odd = true;
					if (i==0 and j==0 and k==0) {new_fun_is_odd=false;}
					CharmsQ1BasisFun fun(*this, new_coord_idx, new_fun_is_odd);
					for (int depth0_idx=0; depth0_idx<this->cursor_support; depth0_idx++)
					{
						const CharmsQ1Element& ELEM0 = (*elements)[this->support[depth0_idx]];
						for (int depth1_idx=0; depth1_idx<ELEM0.cursor_child; depth1_idx++)
						{
							const CharmsQ1Element& ELEM1 = (*elements)[ELEM0.child[depth1_idx]];
							if (ELEM1.contains(fun.coord()))
							{
								fun.insert_support(ELEM1.list_index);
							}
						}
					}

					//possibly a basis function belongs in a coarse mesh, but the basis function at the same coordinate does not belong to the refinement.
					//this can happen when refining near a curved boundary
					if (fun.cursor_support==0) {continue;}

					//add the new basis function to the list
					size_t new_list_index = (size_t) -1;
					int flag = basis->push_back(fun, new_list_index);
					if (flag==0) {continue;} //it is possible that the basis function already exists
					assert(flag==1); //we should only alter newly created functions here

					CharmsQ1BasisFun& CHILD = (*basis)[new_list_index];
					assert(CHILD.list_index==basis->size()-1);
					assert(CHILD.list_index==new_list_index);
					
					//add the new basis to basis_s list of all support elements
					for (int l=0; l<CHILD.cursor_support; l++)
					{
						size_t s_idx = CHILD.support[l];
						(*elements)[s_idx].insert_basis_s(new_list_index);
					}


					//update parents and children
					//support[i].basis_s[j] is a sibling (or this element) and could be a parent of this function's children
					for (int i=0; i<this->cursor_support; i++)
					{
						const CharmsQ1Element& ELEM = (*elements)[this->support[i]];
						for (int j=0; j<ELEM.cursor_basis_s; j++)
						{
							CharmsQ1BasisFun& SIBLING = (*basis)[ELEM.basis_s[j]];
							if (SIBLING.in_support(CHILD.coord()))
							{
								CHILD.insert_parent(SIBLING.list_index);
								SIBLING.insert_child(CHILD.list_index);
							}
						}
					}
				}
			}
		}
	}

	CharmsQ1BasisFun::Box_t CharmsQ1BasisFun::bbox() const //get bounding box for the support of this function
	{
		//create a bounding box for the support of this function
		Box_t result = (*elements)[this->support[0]].bbox();
		for (int i=1; i<this->cursor_support; i++) {result.combine((*elements)[this->support[i]].bbox());}
		return result;
	}

	void CharmsQ1BasisFun::activate()
	{
		if (this->is_active) {return;}
		this->is_active = true;

		//deactivate all ancestor elements
		std::vector<size_t> ancestor_elems = (*elements)[this->support[0]].ancestor_elements();
		for (auto it=ancestor_elems.begin(); it!=ancestor_elems.end(); ++it)
		{
			(*elements)[*it].is_active = false;
		}

		for (int i=0; i<this->cursor_support; i++)
		{
			size_t s_idx = this->support[i];
			if (!(*elements)[s_idx].is_subdivided) //activate support elements if they are not already active
			{
				//this assumes that the domain is covered by active elements at all times with no overlap
				(*elements)[s_idx].is_active  = true;

				//add ancestor basis functions. TODO: is this necessary?
				// std::vector<size_t> ancestor_funs = (*elements)[s_idx].ancestor_basis_fun();
				// for (auto it=ancestor_funs.begin(); it!=ancestor_funs.end(); ++it)
				// {
				// 	(*elements)[s_idx].insert_basis_a(*it);
				// }

				// //add this basis function to all descendent elements. TODO: is this necessary?
				// std::vector<size_t> desc_elems = (*elements)[s_idx].descendent_elements();
				// for (auto it=desc_elems.begin(); it!=desc_elems.end(); ++it)
				// {
				// 	(*elements)[*it].insert_basis_a(this->list_index);
				// }
			}
		}
	}

	template <class Particle_t, size_t n_data>
	void CharmsQ1BasisFun::activate(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts)
	{
		if (this->is_active) {return;}
		
		//deactivate all ancestor elements
		std::vector<size_t> ancestor_elems = (*elements)[this->support[0]].ancestor_elements();
		for (auto it=ancestor_elems.begin(); it!=ancestor_elems.end(); ++it)
		{
			(*elements)[*it].is_active = false;
		}

		for (int i=0; i<this->cursor_support; i++)
		{
			size_t s_idx = this->support[i];
			if (!(*elements)[s_idx].is_subdivided) //activate support elements if they are not already active
			{
				//this assumes that the domain is covered by active elements at all times with no overlap
				bool include_element = true;
				int marker = opts.interface_marker;
				assembly.check_voxel((*elements)[s_idx].bbox(), opts, marker, include_element);
				if (include_element)
				{
					(*elements)[s_idx].is_active = true;
					this->is_active = true; //this basis function can only be active if it has at least one active support element
				}
			}
		}
	}

	void CharmsQ1BasisFun::deactivate()
	{
		//skip if this basis function is already deactivated
		if (!this->is_active) {return;}

		//ensure that any/all children basis functions are not active
		// bool any_child_active = false;
		// for (int i=0; i<this->cursor_child; i++)
		// {
		// 	size_t c_idx = this->child[i];
		// 	any_child_active = any_child_active or (*basis)[c_idx].is_active;
		// }
		// assert(!any_child_active);
		// if (any_child_active) {return;}


		//we are allowed to de-activate this basis function
		this->is_active = false;

		//check if support elements have any active basis functions is basis_s. if not, deactivate those elements.
		for (int i=0; i<this->cursor_support; i++)
		{
			size_t s_idx = this->support[i];
			bool any_active_fun = false;
			for (int j=0; j<(*elements)[s_idx].cursor_basis_s; j++)
			{
				size_t b_idx = (*elements)[s_idx].basis_s[j];
				any_active_fun = any_active_fun or (*basis)[b_idx].is_active;
			}

			if (!any_active_fun)
			{
				(*elements)[s_idx].is_active = false;
			}
		}

	}

	//evaluate this basis function
	double  CharmsQ1BasisFun::eval(const Point_t &point) const
	{
		//loop through support elements
		for (int i=0; i<cursor_support; i++)
		{
			const CharmsQ1Element& ELEM = (*elements)[support[i]];
			if (ELEM.contains(point))
			{
				//create voxel and determine which index this basis function corresponds to
				int idx=0;
				for (int j=0; j<8; j++)
				{
					if (ELEM.node[j] == node_index) {idx=j;}
				}

				//map point back to reference element
				Point_t ref_point = 2.0 * (point - ELEM.center()) / ELEM.H();

				gv::mesh::Voxel voxel;
				return voxel.eval_basis(idx, ref_point);
			}
		}

		//no support element contains the point
		return 0;
	}

	//evaluate the gradient of this function (should only be called on interior points)
	CharmsQ1BasisFun::Point_t CharmsQ1BasisFun::grad(const Point_t &point) const
	{
		//loop through support elements
		for (int i=0; i<cursor_support; i++)
		{
			const CharmsQ1Element& ELEM = (*elements)[support[i]];
			if (ELEM.contains(point))
			{
				//create voxel and determine which index this basis function corresponds to
				int idx=0;
				for (int j=0; j<8; j++)
				{
					if (ELEM.node[j] == node_index) {idx=j;}
				}

				//map point back to reference element
				Point_t ref_point = 2.0 * (point - ELEM.center()) / ELEM.H();

				gv::mesh::Voxel voxel;
				return voxel.eval_grad_basis(idx, ref_point) / (0.5*ELEM.H());
			}
		}

		//no support element contains the point
		return Point_t{0,0,0};
	} 


	//implemtation of CharmsQ1Element methods
	void CharmsQ1Element::subdivide()
	{
		if(this->is_subdivided) {return;}
		this->is_subdivided = true;

		for (int i=0; i<8; i++)
		{
			CharmsQ1Element elem(*this, i); //updates vertex list, sets basis_a[], basis_s[], and node[]

			size_t new_list_index = (size_t) -1;
			[[maybe_unused]] int flag = elements->push_back(elem,new_list_index);
			assert(flag==1); //elements should always be newly created when this routine is called
			const CharmsQ1Element& ELEM = (*elements)[new_list_index];
			assert(ELEM.list_index==new_list_index);
			this->insert_child(ELEM.list_index);
		}
	}

	template <class Particle_t, size_t n_data>
	void CharmsQ1Element::subdivide(const gv::geometry::Assembly<Particle_t,n_data> &assembly, const gv::geometry::AssemblyMeshOptions &opts, std::vector<int> &new_markers)
	{
		if(this->is_subdivided) {return;}
		this->is_subdivided = true;

		for (int i=0; i<8; i++)
		{
			Box_t bbox((*vertices)[node[i]], center());
			int marker = opts.interface_marker;
			bool include_element = true;
			assembly.check_voxel(bbox, opts, marker, include_element);

			// if (include_element)
			if (true)
			{
				CharmsQ1Element elem(*this, i); //updates vertex list, sets basis_a[], basis_s[], and node[]
				size_t new_list_index = (size_t) -1;
				[[maybe_unused]] int flag = elements->push_back(elem,new_list_index);
				assert(flag==1); //elements should always be newly created when this routine is called
				const CharmsQ1Element& ELEM = (*elements)[new_list_index];
				assert(ELEM.list_index==new_list_index);
				this->insert_child(ELEM.list_index);

				new_markers.push_back(marker);
			}
		}
	}

	void CharmsQ1Element::_ancestor_elements(std::vector<size_t> &result) const
	{
		if (this->parent == this->NO_DATA) {return;}
		result.push_back(this->parent);
		(*elements)[this->parent]._ancestor_elements(result);
	}

	void CharmsQ1Element::_descendent_elements(std::vector<size_t> &result) const
	{
		if (!this->is_subdivided) {return;}
		for (int i=0; i<this->cursor_child; i++)
		{
			result.push_back(this->child[i]);
			(*elements)[this->child[i]]._descendent_elements(result);
		}
	}

	std::vector<size_t> CharmsQ1Element::ancestor_basis_fun() const //TODO: make this recursive using this->parent?
	{
		std::vector<size_t> result;
		
		//loop through all ancestor elements
		std::vector<size_t> ancestors = this->ancestor_elements();

		for (auto it=ancestors.begin(); it!=ancestors.end(); ++it)
		{
			const CharmsQ1Element& ELEM = (*elements)[*it];
			for (int i=0; i<ELEM.cursor_basis_s; i++) {result.push_back(ELEM.basis_s[i]);}
			for (int i=0; i<ELEM.cursor_basis_a; i++) {result.push_back(ELEM.basis_a[i]);}
		}

		//make result contain only unique values
		std::sort(result.begin(), result.end());
		auto delete_past = std::unique(result.begin(), result.end());
		result.erase(delete_past, result.end());

		return result;
	}


	//UPDATE PRINTING
	std::ostream& operator<<(std::ostream& os, const CharmsQ1BasisFun& fun)
	{
		os << static_cast<const CharmsQ1BasisFun_BASE&>(fun);
		os << "coordinate\t: " << fun.coord()    << "\n";
		os << "list_index\t: " << fun.list_index << "\n";
		os << "&vertices \t: " << fun.vertices   << "\n";
		os << "&elements \t: " << fun.elements   << "\n";
		os << "&basis    \t: " << fun.basis      << "\n";
		return os;
	}


	std::ostream& operator<<(std::ostream& os, const CharmsQ1Element& elem)
	{
		os << static_cast<const CharmsQ1Element_BASE&>(elem);
		os << "aabb      \t: " << elem.bbox()     << "\n";
		os << "vtk_id    \t: " << elem.vtk_id     << "\n";
		os << "list_index\t: " << elem.list_index << "\n";
		os << "&vertices \t: " << elem.vertices   << "\n";
		os << "&elements \t: " << elem.elements   << "\n";
		os << "&basis    \t: " << elem.basis      << "\n";
		return os;
	}


}