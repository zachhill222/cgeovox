#pragma once

#include "mesh/mesh_util.hpp"

#include <vector>
#include <iterator>
#include <type_traits>

namespace gv::mesh {
	//forward declare the basic mesh class
	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	class BasicMesh;


	/////////////////////////////////////////////////
	/// Enum to pass the option for an iterator to point to _elements or _boundary in the mesh
	/////////////////////////////////////////////////
	enum class ContainerType {ELEMENTS, BOUNDARY};


	/////////////////////////////////////////////////
	/// Iterator class for std::vector<Element_t> and std::vector<Face_t>.
	/// (For now) these are not thread safe.
	/// Other classes should inherit these fine, but may need to override isElementValid().
	/// These iterators are primarily so that we can loop through the elements of the mesh using the same API regardless of the specific mesh type.
	/////////////////////////////////////////////////
	template<typename Mesh_t, ContainerType CONTAINER>
	class ElementIterator {
	private:
		///Get the appropriate element type from the mesh
		using Element_t = std::conditional_t<CONTAINER == ContainerType::ELEMENTS, typename Mesh_t::element_type, typename Mesh_t::face_type>;

		Mesh_t* mesh;
		std::vector<Element_t>* _elements; //pointer to either _elements or _boundary in the main mesh
		size_t curr_idx;

		bool isValid(const size_t idx) {
			// if constexpr (CONTAINER == ContainerType::ELEMENTS) {return mesh->isElementValid((*_elements)[idx]);}
			// if constexpr (CONTAINER == ContainerType::BOUNDARY) {return mesh->isFaceValid((*_elements)[idx]);}
			if constexpr (HierarchicalMeshElement<Element_t>) {return (*_elements)[idx].is_active;}
			else {return true;}
		}

		void advance() {
			if (curr_idx>=_elements->size()) {return;}
			++curr_idx;
			while (curr_idx < _elements->size() and !isValid(curr_idx)) {++curr_idx;}
		}

		void retreat() {
			if (curr_idx==0) {return;} //no valid element to decrease to

			size_t tmp = curr_idx - 1;
		    while (!isValid(tmp)) {
		        if (tmp == 0) return;  // Can't retreat further
		        --tmp;
		    }
		    curr_idx = tmp;  // Found a valid element
		}

	public:
		using iterator_catagory = std::bidirectional_iterator_tag;
		using pointer           = Element_t*;
		using reference         = Element_t&;

		//Default constructor
		ElementIterator() : mesh(nullptr), _elements(nullptr), curr_idx(0) {}

		//Constructor to link to a mesh
		ElementIterator(Mesh_t* mesh, size_t idx) : mesh(mesh), curr_idx(idx) {
			if constexpr      (CONTAINER == ContainerType::ELEMENTS) {_elements = &(mesh->_elements);}
			else if constexpr (CONTAINER == ContainerType::BOUNDARY) {_elements = &(mesh->_boundary);}
			else {throw std::runtime_error("Unknown ContainerType");}

			if (idx==0) {moveToBegin();}
			else if (idx>=_elements->size()) {moveToEnd();}
		}

		//Constructor to link to a mesh and specific vector
		ElementIterator(Mesh_t* mesh, std::vector<Element_t>* elems, size_t idx) : mesh(mesh), _elements(elems), curr_idx(idx) {
			if (idx==0) {moveToBegin();}
			else if (idx==(size_t) -1) {moveToEnd();}
		}

		//Copy constructor
		ElementIterator(const ElementIterator &other) = default;

		//Assignment operator
		ElementIterator& operator=(const ElementIterator &other) = default;

		//Pre-increment
		ElementIterator& operator++() {
			advance();
			return *this;
		}

		//Post-increment
		ElementIterator operator++(int) {
			ElementIterator tmp = *this;
			++(*this);
			return tmp;
		}

		// Pre-decrement  
		ElementIterator& operator--() {
		    retreat();
		    return *this;
		}

		//Post-decrement
		ElementIterator operator--(int) {
			ElementIterator tmp = *this;
			--(*this);
			return tmp;
		}


		// Comparison operators
	    bool operator==(const ElementIterator& other) const {
	        return curr_idx == other.curr_idx && _elements == other._elements;
	    }
	    
	    bool operator!=(const ElementIterator& other) const {
	        return !(*this == other);
	    }
	    
	    // Get current index
	    size_t index() const { return curr_idx; }

	    // Reference
		Element_t& operator*() { 
		    return (*_elements)[curr_idx]; 
		}

		const Element_t& operator*() const { 
		    return (*_elements)[curr_idx]; 
		}

		Element_t* operator->() { 
		    return &((*_elements)[curr_idx]); 
		}

		const Element_t* operator->() const { 
		    return &((*_elements)[curr_idx]); 
		}

		// Helpers for begin() and end()
		ElementIterator moveToEnd() {curr_idx = _elements->size(); return *this;}
		ElementIterator moveToBegin() {
			curr_idx=0;
			if (!isValid(curr_idx)) {advance();}
			return *this;
		}
	};
}