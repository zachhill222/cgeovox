#pragma once


#include "fem/charms_util.hpp"
#include "util/point_octree.hpp"
#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include <vector>
#include <cassert>
#include <iostream>

namespace gv::fem
{
	//constructors for basis functions and elements may need to refer to eachother
	class CharmsQ1BasisFun;
	class CharmsQ1Element;
	class CharmsQ1BasisFunOctree;
	class CharmsQ1ElementOctree;
	
	class CharmsQ1BasisFun : public CharmsBasisFun<8>
	{
	public:
		using VertexList_t = gv::util::PointOctree<3>;
		using Point_t      = gv::util::Point<3,double>;
		using Box_t        = gv::util::Box<3>;

		//pointers to lists of vertices and elements.
		VertexList_t*           vertices;
		CharmsQ1ElementOctree*  elements;
		CharmsQ1BasisFunOctree* basis;
		
		size_t basis_list_index = CharmsBasisFun<8>::NO_DATA; //track where this basis function occurs in the basis function list. set by octree container.
		
		//default constructor
		constexpr CharmsQ1BasisFun() : CharmsBasisFun<8>(), vertices(nullptr), elements(nullptr), basis(nullptr) {}

		//constructor for creating a new basis function in the coarsest mesh. elements[] must be up to date.
		//sets is_active and is_odd to true
		CharmsQ1BasisFun(VertexList_t* vertices, CharmsQ1ElementOctree* elements, CharmsQ1BasisFunOctree* basis, const size_t node_index) :
			CharmsBasisFun<8>(node_index, 0, true, true, false),
			vertices(vertices),
			elements(elements),
			basis(basis) {}

		//constructor for creating a new basis function using information from a parent basis function
		//parent[], child[], and support[] are initialized but only filled with NO_DATA (i.e. (size_t) -1)
		CharmsQ1BasisFun(const CharmsQ1BasisFun& some_parent, const size_t node_index, const bool is_odd) :
			CharmsBasisFun<8>(node_index, some_parent.depth+1, is_odd, false, false),
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
		CharmsQ1BasisFun(CharmsQ1BasisFun&& other) :
			CharmsBasisFun<8>(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			basis_list_index(other.basis_list_index)
		{
			other.vertices = nullptr;
			other.elements = nullptr;
			other.basis    = nullptr;
		}

		//copy constructor
		CharmsQ1BasisFun(const CharmsQ1BasisFun& other) :
			CharmsBasisFun<8>(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			basis_list_index(other.basis_list_index){}

		//update move operator
		CharmsQ1BasisFun& operator=(CharmsQ1BasisFun&& other)
		{
			if (this!=&other)
			{
				CharmsBasisFun::operator=(other);
				vertices = other.vertices; other.vertices = nullptr;
				elements = other.elements; other.elements = nullptr;
				basis    = other.basis;    other.basis    = nullptr;
				basis_list_index = other.basis_list_index;
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
			basis_list_index = other.basis_list_index;
			
			return *this;
		}

		//populate the support of this basis function. the elements[] list must be updated first.
		void set_support();

		//add this function to the appropriate basis_s and basis_a of the elements.
		void update_element_basis_lists();

		//add detail functions of this basis function. divides elements as needed. adds new basis functions to the basis list.
		//does not change is_active and sets is_active=false for the new basis functions.
		void subdivide();

		inline Point_t coord() const {return (*vertices)[this->node_index];}
		Box_t bbox() const; //get bounding box for the support of this element
	};


	
	class CharmsQ1Element : public CharmsElement<8>
	{
	public:
		using VertexList_t = CharmsQ1BasisFun::VertexList_t;
		using Point_t      = gv::util::Point<3,double>;
		using Box_t        = gv::util::Box<3>;

		//pointers to lists of vertices and elements.
		VertexList_t*           vertices;
		CharmsQ1ElementOctree*  elements;
		CharmsQ1BasisFunOctree* basis;
		
		size_t element_list_index = CharmsElement<8>::NO_DATA; //track where this element occurs in the element list. set by octree container.

		//default constructor
		constexpr CharmsQ1Element() : CharmsElement<8>(), vertices(nullptr), elements(nullptr), basis(nullptr) {}

		//constructor for creating the coarsest mesh. adds vertices to the list, but does not add the element or create any basis functions
		//sets is_active=true
		CharmsQ1Element(VertexList_t* vertices, CharmsQ1ElementOctree* elements, CharmsQ1BasisFunOctree* basis, const Box_t &bbox) :
			CharmsElement<8>(0, (size_t)-1, true, false),
			vertices(vertices),
			elements(elements),
			basis(basis)
		{
			size_t node_idx;
			for (int i=0; i<8; i++)
			{
				vertices->push_back(bbox.voxelvertex(i), node_idx);
				this->insert_node(node_idx);
			}
		}

		//constructor for creating a new element using information from its parent. adds vertices to the list if necessary.
		//sets is_active=false
		CharmsQ1Element(const CharmsQ1Element& parent, int sibling_number) :
			CharmsElement<8>(parent.depth+1, parent.element_list_index, false, false),
			vertices(parent.vertices),
			elements(parent.elements),
			basis(parent.basis)
		{
			for (int i=0; i<parent.cursor_basis_s; i++) {this->insert_basis_a(parent.basis_s[i]);}
			for (int i=0; i<parent.cursor_basis_a; i++) {this->insert_basis_a(parent.basis_a[i]);}

			Box_t bbox((*vertices)[parent.node[sibling_number]], parent.center());
			size_t node_idx;
			for (int i=0; i<8; i++)
			{
				int flag = vertices->push_back(bbox.voxelvertex(i), node_idx);
				if (flag==1) {std::cout << "vertex: " << node_idx << "\t" << bbox.voxelvertex(i) << std::endl;}
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
		CharmsQ1Element(CharmsQ1Element&& other) :
			CharmsElement<8>(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			element_list_index(other.element_list_index)
		{
			other.vertices = nullptr;
			other.elements = nullptr;
			other.basis    = nullptr;
		}

		//copy constructor
		CharmsQ1Element(const CharmsQ1Element& other) :
			CharmsElement<8>(other),
			vertices(other.vertices),
			elements(other.elements),
			basis(other.basis),
			element_list_index(other.element_list_index){}

		//update move operator
		CharmsQ1Element& operator=(CharmsQ1Element&& other)
		{
			if (this!=&other)
			{
				CharmsElement::operator=(other);
				vertices = other.vertices; other.vertices = nullptr;
				elements = other.elements; other.elements = nullptr;
				basis    = other.basis;    other.basis    = nullptr;
				element_list_index = other.element_list_index;
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
			element_list_index = other.element_list_index;
			
			return *this;
		}


		//split an element. adds vertices and elements to their lists, but does not create any basis functions.
		//does not change the is_active marker of any element or basis function
		void subdivide();

		inline Box_t bbox() const {return Box_t{(*vertices)[node[0]], (*vertices)[node[7]]};}
		inline Point_t center() const {return 0.5*((*vertices)[node[0]] + (*vertices)[node[7]]);}
		inline Point_t H() const {return (*vertices)[node[7]] - (*vertices)[node[0]];}
		inline bool contains(const Point_t &coord) const {return (*vertices)[node[0]] <= coord and coord <= (*vertices)[node[7]];}

	};


	//Octee definitions
	class CharmsQ1BasisFunOctree : public gv::util::BasicOctree<CharmsQ1BasisFun,3,32>
	{
	public:
		CharmsQ1BasisFunOctree(const gv::util::Box<3> &domain) : gv::util::BasicOctree<CharmsQ1BasisFun,3,32>(domain,64) {}
	private:
		bool is_data_valid(const gv::util::Box<3> &box, const CharmsQ1BasisFun &data) const override {return box.contains(data.coord());}
	};


	class CharmsQ1ElementOctree : public gv::util::BasicOctree<CharmsQ1Element,3,32>
	{
	public:
		CharmsQ1ElementOctree(const gv::util::Box<3> &domain) : gv::util::BasicOctree<CharmsQ1Element,3,32>(domain,64) {}
	private:
		bool is_data_valid(const gv::util::Box<3> &box, const CharmsQ1Element &data) const override {return box.intersects(data.bbox());}
	};


	//implement methods for CharmsQ1BasisFun
	void CharmsQ1BasisFun::set_support()
	{
		std::vector<size_t> check_elements_idx = elements->get_data_indices(this->coord());
		for (auto it=check_elements_idx.begin(); it!=check_elements_idx.end(); ++it)
		{
			const CharmsQ1Element& ELEM = (*elements)[*it];
			if (ELEM.contains(this->coord()) and ELEM.depth==this->depth) {this->insert_support(*it);}
		}
		assert(this->cursor_support>0);
	}

	void CharmsQ1BasisFun::update_element_basis_lists()
	{
		assert(basis_list_index!=this->NO_DATA);
		Box_t support_bbox = bbox();

		//get a list of all elements that intersect the support
		std::vector<size_t> check_elements_idx = elements->get_data_indices(support_bbox);
		for (auto it=check_elements_idx.begin(); it!=check_elements_idx.end(); ++it)
		{
			CharmsQ1Element& ELEM = (*elements)[*it];
			if (ELEM.bbox().contains(this->coord()))
			{
				if (this->depth==ELEM.depth) {ELEM.insert_basis_s(this->basis_list_index);}
				else if (this->depth<ELEM.depth) {ELEM.insert_basis_a(this->basis_list_index);}
			}
		}
	}

	void CharmsQ1BasisFun::subdivide()
	{
		std::cout << "subdivide basis function " << basis_list_index << std::endl;
		if (this->is_refined) {return;}
		this->is_refined = true;

		//subdivide each support element if necessary
		// while (elements->capacity() < elements->size()+8) {elements->reserve(2*elements->capacity());}
		std::cout << elements->size() << "/" << elements->capacity() << std::endl;
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
					std::cout << "basis: " << basis->size() << "/" << basis->capacity() << std::endl;
					Point_t new_coord = this->coord() + H * Point_t{i,j,k};
					size_t  new_coord_idx = vertices->find(new_coord);
					if(new_coord_idx >= vertices->size()) {continue;} //happens near domain boundary

					//initialize the new basis function
					bool new_fun_is_odd = true;
					if (i==0 and j==0 and k==0) {new_fun_is_odd=false;}
					CharmsQ1BasisFun fun(*this, new_coord_idx, new_fun_is_odd);
					fun.set_support();

					//add the new basis function to the list
					size_t new_basis_list_index;
					int flag = basis->push_back(fun, new_basis_list_index);
					assert(flag!=-1);
					(*basis)[new_basis_list_index].basis_list_index = new_basis_list_index;

					//add the new basis function as a child of this function
					// this->insert_child(new_basis_list_index);

					//add this function as a parent of the new basis function
					(*basis)[new_basis_list_index].insert_parent(this->basis_list_index);
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

	//implemtation of CharmsQ1Element methods
	void CharmsQ1Element::subdivide()
	{
		std::cout << "subdivide element " << element_list_index << std::endl;
		std::cout << *this << std::endl;

		if(this->is_refined) {return;}
		this->is_refined = true;

		for (int i=0; i<8; i++)
		{
			CharmsQ1Element elem(*this, i); //updates vertex list, sets basis_a[], basis_s[], and node[]
			std::cout << elem << std::endl;
			std::cout << elements << "\t" << elem.elements << std::endl;

			size_t new_element_list_index;
			int flag = elements->push_back(elem,new_element_list_index);
			assert(flag!=-1);
			std::cout << flag << std::endl;
			std::cout << elements << std::endl;
			(*elements)[new_element_list_index].element_list_index = new_element_list_index;
			this->insert_child(elem.element_list_index);
		}
	}


}