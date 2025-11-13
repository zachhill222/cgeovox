#pragma once

#include<cassert>
#include<iostream>

//this file provedes the most basic utility to use the CHARMS method. element types are not assumed to be known.


namespace gv::charms
{
	//bare essentials for the charms basis functions
	template<int SUPPORT_SIZE, int CHILD_SIZE, int PARENT_SIZE>
	class CharmsBasisFun
	{
	public:
		static constexpr size_t NO_DATA = (size_t) -1; //not every basis function will have the maximum number of parents/children.
		size_t  node_index;     //index to coordinate in space
		size_t* support;        //indices of support elements with the same depth as this basis function
		int     cursor_support; //index of next support data to be inserted
		size_t  depth;          //number of mesh refinements from the coarsest mesh
		bool    is_odd;         //a basis function is odd if it has the minimum possible depth for its spatial coordinate
		size_t* child;          //indices of basis functions with depth one greater and whose support intersects the support of this basis function (with positive measure)
		int     cursor_child;   //index of next child data to be inserted
		size_t* parent;         //adjoint of child
		int     cursor_parent;  //index of next parent data to be inserted
		bool    is_active;      //mark if this basis function is active or not
		bool    is_refined;     //mark if this basis function has been refined (e.g., h_refine has been called on it. used to track if it is a valid un-refine target)
		bool    is_subdivided;  //mark if this basis function has beed refined/split

		constexpr CharmsBasisFun() : 
			node_index(NO_DATA), 
			support(nullptr),
			cursor_support(0), 
			depth(0), 
			is_odd(false), 
			child(nullptr),
			cursor_child(0), 
			parent(nullptr),
			cursor_parent(0), 
			is_active(false), 
			is_refined(false),
			is_subdivided(false) {};

		CharmsBasisFun(const size_t node_index, const size_t depth, const bool is_odd, const bool is_active, const bool is_refined, const bool is_subdivided) :
			node_index(node_index), 
			support(new size_t[SUPPORT_SIZE]),
			cursor_support(0),
			depth(depth),
			is_odd(is_odd), 
			child(new size_t[CHILD_SIZE]),
			cursor_child(0),
			parent(new size_t[PARENT_SIZE]),
			cursor_parent(0),
			is_active(is_active),
			is_refined(is_refined),
			is_subdivided(is_subdivided)
			{
				fill_array(NO_DATA, support, SUPPORT_SIZE);
				fill_array(NO_DATA, child,   CHILD_SIZE);
				fill_array(NO_DATA, parent,  PARENT_SIZE);
			}

		//move constructor
		CharmsBasisFun(CharmsBasisFun&& other) noexcept :
			node_index(other.node_index), 
			support(other.support),
			cursor_support(other.cursor_support),
			depth(other.depth),
			is_odd(other.is_odd), 
			child(other.child),
			cursor_child(other.cursor_child),
			parent(other.parent),
			cursor_parent(other.cursor_parent),
			is_active(other.is_active),
			is_refined(other.is_refined),
			is_subdivided(other.is_subdivided)
		{
			if (other.support!=nullptr) {delete[] other.support; other.support=nullptr;}
			if (other.child!=nullptr)   {delete[] other.child; other.child=nullptr;}
			if (other.parent!=nullptr)  {delete[] other.parent; other.parent=nullptr;}
		}

		//copy constructor
		CharmsBasisFun(const CharmsBasisFun& other) :
			node_index(other.node_index), 
			support(new size_t[SUPPORT_SIZE]),
			cursor_support(other.cursor_support),
			depth(other.depth),
			is_odd(other.is_odd), 
			child(new size_t[CHILD_SIZE]),
			cursor_child(other.cursor_child),
			parent(new size_t[PARENT_SIZE]),
			cursor_parent(other.cursor_parent),
			is_active(other.is_active),
			is_refined(other.is_refined),
			is_subdivided(other.is_subdivided)
		{
			//copy data
			for (int i=0; i<SUPPORT_SIZE; i++) {support[i]=other.support[i];}
			for (int i=0; i<CHILD_SIZE; i++)   {child[i]=other.child[i];}
			for (int i=0; i<PARENT_SIZE; i++)  {parent[i]=other.parent[i];}
		}

		~CharmsBasisFun()
		{
			if (support!=nullptr) {delete[] support;}
			if (child!=nullptr)   {delete[] child;}
			if (parent!=nullptr)  {delete[] parent;}
		}

		//move assignment
		CharmsBasisFun& operator=(CharmsBasisFun&& other) noexcept
		{
			if (this!=&other)
			{
				//copy non-pointers
				node_index     = other.node_index;
				depth          = other.depth;
				is_odd         = other.is_odd;
				is_active      = other.is_active;
				is_refined     = other.is_refined;
				cursor_support = other.cursor_support;
				cursor_child   = other.cursor_child;
				cursor_parent  = other.cursor_parent;
				is_subdivided  = other.is_subdivided;

				//move support
				if (support!=nullptr) {delete[] support;}
				support = other.support;
				other.support=nullptr;

				//move child
				if (child!=nullptr) {delete[] child;}
				child = other.child;
				other.child=nullptr;

				//move parent
				if (parent!=nullptr) {delete[] parent;}
				parent = other.parent;
				other.parent=nullptr;
			}
			return *this;
		}

		//copy assignment
		CharmsBasisFun& operator=(const CharmsBasisFun& other)
		{
			//copy non-pointers
			node_index     = other.node_index;
			depth          = other.depth;
			is_odd         = other.is_odd;
			is_active      = other.is_active;
			is_refined     = other.is_refined;
			cursor_support = other.cursor_support;
			cursor_child   = other.cursor_child;
			cursor_parent  = other.cursor_parent;
			is_subdivided  = other.is_subdivided;

			//copy support
			if (other.support==nullptr)
			{
				if (support!=nullptr) {delete[] support; support=nullptr;}
			}
			else
			{
				if (support==nullptr) {support = new size_t[SUPPORT_SIZE];}
				for (int i=0; i<SUPPORT_SIZE; i++) {support[i]=other.support[i];}
			}

			//copy child
			if (other.child==nullptr)
			{
				if (child!=nullptr) {delete[] child; child=nullptr;}
			}
			else
			{
				if (child==nullptr) {child = new size_t[CHILD_SIZE];}
				for (int i=0; i<CHILD_SIZE; i++) {child[i]=other.child[i];}
			}

			//copy parent
			if (other.parent==nullptr)
			{
				if (parent!=nullptr) {delete[] parent; parent=nullptr;}
			}
			else
			{
				if (parent==nullptr) {parent = new size_t[PARENT_SIZE];}
				for (int i=0; i<PARENT_SIZE; i++) {parent[i]=other.parent[i];}
			}
			
			return *this;
		}


		void insert_support(const size_t value) {insert(value, support, cursor_support, SUPPORT_SIZE);}
		void insert_child(const size_t value)   {insert(value, child, cursor_child, CHILD_SIZE);}
		void insert_parent(const size_t value)  {insert(value, parent, cursor_parent, PARENT_SIZE);}

		bool operator==(const CharmsBasisFun& other) const
		{
			if (this->depth!=other.depth) {return false;}
			if (this->node_index!=other.node_index) {return false;}
			return true;
		}

	private:
		//data insertion.
		void fill_array(const size_t val, size_t* arr, const int ARR_SIZE)
		{
			assert(arr!=nullptr);
			for (int i=0; i<ARR_SIZE; i++) {arr[i]=val;}
		}

		void insert(const size_t value, size_t* array, int &cursor, const int ARR_SIZE)
		{
			assert(array!=nullptr);
			if (cursor>=ARR_SIZE) //if the array is full, verify that the value is already contained. TODO: make a debug flag to remove this?
			{
				for (int i=0; i<ARR_SIZE; i++) {if (value==array[i]) {return;}}
				assert("array is full\n" and false);
			}

			assert(cursor<ARR_SIZE);
			assert(array[cursor]==NO_DATA);

			//verify that the value is unique
			for (int i=0; i<cursor; i++) {if (array[i]==value) {return;}} //data was already contained, no changes made.

			array[cursor] = value;
			cursor++;
		}
	};

	//bare essentials for a charms element
	template<int NODE_SIZE, int CHILD_SIZE, int BASIS_S_SIZE>
	class CharmsElement
	{
	public:
		static constexpr size_t NO_DATA = (size_t) -1; //not every element will have the maximum number of parent/child.
		size_t* node;             //index to coordinate in space for each vertex (i.e. elem2node mesh data)
		int     cursor_node;      //location of next node data to add
		size_t  depth;            //number of mesh refinements from the coarsest mesh
		size_t* child;            //indices of elements with depth one greater and are contained in this element
		int     cursor_child;     //location of next child to add
		size_t  parent;           //adjoint of child, elements can only have a single parent
		size_t* basis_s;          //indices of basis functions with the same depth
		int     cursor_basis_s;   //location of next basis_s to add
		int     capacity_basis_a; //current capacity of basis_a to allow re-sizing
		size_t* basis_a;          //indices of basis functions with a lower depth and whose support overlap this element. this must be dynamically resized.
		int     cursor_basis_a;   //location of next basis_a to add
		bool    is_active;        //mark if this element is active or not
		bool    is_subdivided;    //mark if this element has been refined/subdiveded

		constexpr CharmsElement() : node(nullptr), depth(0), child(nullptr), parent(NO_DATA), basis_s(nullptr), basis_a(nullptr), is_active(false), is_subdivided(false) {}

		CharmsElement(const int depth, const size_t parent_index, const int parent_capacity_basis_a, const bool is_active, bool is_subdivided) :
			node(new size_t[NODE_SIZE]),
			cursor_node(0),
			depth(depth),
			child(new size_t[CHILD_SIZE]),
			cursor_child(0),
			parent(parent_index),
			basis_s(new size_t[BASIS_S_SIZE]),
			cursor_basis_s(0),
			capacity_basis_a(parent_capacity_basis_a),
			basis_a(new size_t[capacity_basis_a]),
			cursor_basis_a(0),
			is_active(is_active),
			is_subdivided(is_subdivided)
		{
			fill_array(NO_DATA, node,    NODE_SIZE);
			fill_array(NO_DATA, child,   CHILD_SIZE);
			fill_array(NO_DATA, basis_s, BASIS_S_SIZE);
			fill_array(NO_DATA, basis_a, capacity_basis_a);
		}

		~CharmsElement()
		{
			if (node!=nullptr)    {delete[] node;}
			if (child!=nullptr)   {delete[] child;}
			if (basis_s!=nullptr) {delete[] basis_s;}
			if (basis_a!=nullptr) {delete[] basis_a;}
		}

		//move constructor
		CharmsElement(CharmsElement&& other) noexcept :
			node(other.node),
			cursor_node(other.cursor_node),
			depth(other.depth),
			child(other.child),
			cursor_child(other.cursor_child),
			parent(other.parent),
			basis_s(other.basis_s),
			cursor_basis_s(other.cursor_basis_s),
			capacity_basis_a(other.capacity_basis_a),
			basis_a(other.basis_a),
			cursor_basis_a(other.cursor_basis_a),
			is_active(other.is_active),
			is_subdivided(other.is_subdivided)
		{
			other.node    = nullptr;
			other.child   = nullptr;
			other.basis_s = nullptr;
			other.basis_a = nullptr;
		}

		//copy constructor
		CharmsElement(const CharmsElement& other) :
			node(new size_t[NODE_SIZE]),
			cursor_node(other.cursor_node),
			depth(other.depth),
			child(new size_t[CHILD_SIZE]),
			cursor_child(other.cursor_child),
			parent(other.parent),
			basis_s(new size_t[BASIS_S_SIZE]),
			cursor_basis_s(other.cursor_basis_s),
			capacity_basis_a(other.capacity_basis_a),
			basis_a(new size_t[other.capacity_basis_a]),
			cursor_basis_a(other.cursor_basis_a),
			is_active(other.is_active),
			is_subdivided(other.is_subdivided)
		{
			//copy data
			for (int i=0; i<NODE_SIZE; i++) {node[i]=other.node[i];}
			for (int i=0; i<CHILD_SIZE; i++) {child[i]=other.child[i];}
			for (int i=0; i<BASIS_S_SIZE; i++) {basis_s[i]=other.basis_s[i];}
			for (int i=0; i<capacity_basis_a; i++) {basis_a[i]=other.basis_a[i];}
		}

		//move assignment
		CharmsElement& operator=(CharmsElement&& other) noexcept
		{
			if (this!=&other)
			{
				//copy non-pointers
				cursor_node      = other.cursor_node;
				depth            = other.depth;
				cursor_child     = other.cursor_child;
				parent           = other.parent;
				cursor_basis_s   = other.cursor_basis_s;
				cursor_basis_a   = other.cursor_basis_a;
				capacity_basis_a = other.capacity_basis_a;
				is_active        = other.is_active;
				is_subdivided    = other.is_subdivided;

				//move node
				if (node!=nullptr) {delete[] node;}
				node = other.node;
				other.node=nullptr;

				//move child
				if (child!=nullptr) {delete[] child;}
				child = other.child;
				other.child=nullptr;

				//move basis_s
				if (basis_s!=nullptr) {delete[] basis_s;}
				basis_s = other.basis_s;
				other.basis_s=nullptr;

				//move basis_a
				if (basis_a!=nullptr) {delete[] basis_a;}
				basis_a = other.basis_a;
				other.basis_a=nullptr;
			}
			return *this;
		}

		//copy assignment
		CharmsElement& operator=(const CharmsElement& other)
		{
			//copy non-pointers
			cursor_node      = other.cursor_node;
			depth            = other.depth;
			cursor_child     = other.cursor_child;
			parent           = other.parent;
			cursor_basis_s   = other.cursor_basis_s;
			cursor_basis_a   = other.cursor_basis_a;
			capacity_basis_a = other.capacity_basis_a;
			is_active        = other.is_active;
			is_subdivided    = other.is_subdivided;

			//copy node
			if (other.node==nullptr)
			{
				if (node!=nullptr) {delete[] node; node=nullptr;}
			}
			else
			{
				if (node==nullptr) {node = new size_t[NODE_SIZE];}
				for (int i=0; i<NODE_SIZE; i++) {node[i]=other.node[i];}
			}

			//copy child
			if (other.child==nullptr)
			{
				if (child!=nullptr) {delete[] child; child=nullptr;}
			}
			else
			{
				if (child==nullptr) {child = new size_t[CHILD_SIZE];}
				for (int i=0; i<CHILD_SIZE; i++) {child[i]=other.child[i];}
			}

			//copy basis_s
			if (other.basis_s==nullptr)
			{
				if (basis_s!=nullptr) {delete[] basis_s; basis_s=nullptr;}
			}
			else
			{
				if (basis_s==nullptr) {basis_s = new size_t[BASIS_S_SIZE];}
				for (int i=0; i<BASIS_S_SIZE; i++) {basis_s[i]=other.basis_s[i];}
			}

			//copy basis_a
			if (other.basis_a==nullptr)
			{
				if (basis_a!=nullptr) {delete[] basis_a; basis_a=nullptr;}
			}
			else
			{
				if (basis_a==nullptr) {basis_a = new size_t[capacity_basis_a];}
				for (int i=0; i<capacity_basis_a; i++) {basis_a[i]=other.basis_a[i];}
			}
		
			return *this;
		}
		
		inline void insert_node(const size_t value)    {insert(value, node, cursor_node, NODE_SIZE);}
		inline void insert_child(const size_t value)   {insert(value, child, cursor_child, CHILD_SIZE);}
		inline void insert_basis_s(const size_t value) {insert(value, basis_s, cursor_basis_s, BASIS_S_SIZE);}
		inline void insert_basis_a(const size_t value)
		{
			if (cursor_basis_a>=capacity_basis_a) //expand capacity if needed
			{
				size_t* old_basis_a          = basis_a;
				int     old_capacity_basis_a = capacity_basis_a;

				capacity_basis_a = 2*capacity_basis_a;
				basis_a          = new size_t[capacity_basis_a];

				for (int i=0; i<old_capacity_basis_a; i++) {basis_a[i]=old_basis_a[i];}
				for (int i=cursor_basis_a; i<capacity_basis_a; i++) {basis_a[i]=NO_DATA;}

				delete[] old_basis_a;
			}
			
			insert(value, basis_a, cursor_basis_a, capacity_basis_a);
		}

		bool operator==(const CharmsElement& other) const //useful for storing in an octree
		{
			if (this->depth!=other.depth) {return false;}
			if (this->node!=nullptr and other.node==nullptr) {return false;}
			if (this->node==nullptr and other.node!=nullptr) {return false;}
			if (this->node!=nullptr and other.node!=nullptr)
			{
				for (int i=0; i<NODE_SIZE; i++)
				{
					if (this->node[i]!=other.node[i]) {return false;}
				}
			}

			return true;
		}

	private:
		//data insertion.
		void fill_array(const size_t val, size_t* arr, const int ARR_SIZE)
		{
			assert(arr!=nullptr);
			for (int i=0; i<ARR_SIZE; i++) {arr[i]=val;}
		}

		void insert(const size_t value, size_t* array, int &cursor, const int ARR_SIZE)
		{
			assert(array!=nullptr);
			if (cursor>=ARR_SIZE) //if the array is full, verify that the value is already contained
			{
				for (int i=0; i<ARR_SIZE; i++) {if (value==array[i]) {return;}}
				assert("array is full" and false);
			}

			assert(cursor<ARR_SIZE);
			assert(array[cursor]==NO_DATA);

			//verify that the value is unique
			for (int i=0; i<cursor; i++) {if (array[i]==value) {return;}} //data was already contained, no changes made.

			array[cursor] = value;
			cursor++;
		}
	};

	//DEBUG AND PRINTING HELP
	void to_stream(std::ostream& os, const size_t* arr, const int len)
	{
		for (int i=0; i<len; i++)
		{
			if (arr[i]!=(size_t)-1) {os << arr[i] << " ";}
			else {os << "-1 ";}
		}
	}


	template<int SUPPORT_SIZE, int CHILD_SIZE, int PARENT_SIZE>
	std::ostream& operator<<(std::ostream& os, const CharmsBasisFun<SUPPORT_SIZE,CHILD_SIZE,PARENT_SIZE>& fun)
	{
		os << "node_index\t: " << fun.node_index << "\n";
		os << "support (" << fun.cursor_support <<")\t: "; to_stream(os, fun.support, fun.cursor_support); os << "\n";
		os << "depth     \t: " << fun.depth  << "\n";
		os << "is_odd    \t: " << fun.is_odd << "\n";
		os << "parent (" << fun.cursor_parent << ")\t: "; to_stream(os, fun.parent, fun.cursor_parent); os << "\n";
		os << "child (" << fun.cursor_child << ")\t: "; to_stream(os, fun.child, fun.cursor_child);   os << "\n";
		os << "is_active \t: " << fun.is_active  << "\n";
		os << "is_refined\t: " << fun.is_refined << "\n";
		os << "is_subdivided\t: " << fun.is_subdivided << "\n";
		return os;
	}

	template<int NODE_SIZE, int CHILD_SIZE, int BASIS_S_SIZE>
	std::ostream& operator<<(std::ostream& os, const CharmsElement<NODE_SIZE,CHILD_SIZE,BASIS_S_SIZE>& elem)
	{
		os << "node (" << elem.cursor_node << ")\t: "; to_stream(os, elem.node, elem.cursor_node); os << "\n";
		os << "depth      \t: " << elem.depth  << "\n";
		os << "parent     \t: " << elem.parent << "\n";
		os << "child (" << elem.cursor_child << ")\t: "; to_stream(os, elem.child, elem.cursor_child);     os << "\n";
		os << "basis_s (" << elem.cursor_basis_s << ")\t: "; to_stream(os, elem.basis_s, elem.cursor_basis_s); os << "\n";
		os << "basis_a (" << elem.cursor_basis_a << ")\t: "; to_stream(os, elem.basis_a, elem.cursor_basis_a); os << "\n";
		os << "is_active  \t: " << elem.is_active  << "\n";
		os << "is_subdivided \t: " << elem.is_subdivided << "\n";
		return os;
	}
}