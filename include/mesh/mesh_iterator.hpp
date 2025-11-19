#pragma once

#include "mesh/mesh_util.hpp"

#include <vector>
#include <iterator>
#include <type_traits>

namespace gv::mesh {
	
	// Forward declaration
	template<BasicMeshNode Node_t, BasicMeshElement Element_t, BasicMeshElement Face_t>
	class BasicMesh;

	/////////////////////////////////////////////////
	/// Container type selector
	/////////////////////////////////////////////////
	enum class ContainerType {
		ELEMENTS,
		BOUNDARY
	};

	/////////////////////////////////////////////////
	/// Bidirectional iterator for mesh elements and boundary faces
	/// 
	/// This iterator automatically skips invalid elements (e.g., inactive elements
	/// in hierarchical meshes). The validity check is done via compile-time
	/// detection of the HierarchicalMeshElement concept.
	///
	/// @tparam Mesh_t The mesh type to iterate over
	/// @tparam CONTAINER Which container to iterate (ELEMENTS or BOUNDARY)
	/////////////////////////////////////////////////
	template<typename Mesh_t, ContainerType CONTAINER>
	class ElementIterator {
	public:
		// Iterator traits
		using iterator_category = std::bidirectional_iterator_tag;
		using difference_type   = std::ptrdiff_t;
		using value_type        = std::conditional_t<
			CONTAINER == ContainerType::ELEMENTS,
			typename Mesh_t::element_type,
			typename Mesh_t::face_type
		>;
		using pointer           = value_type*;
		using reference         = value_type&;

	private:
		using Element_t = value_type;

		Mesh_t* _mesh;
		std::vector<Element_t>* _container;
		size_t _curr_idx;

		/////////////////////////////////////////////////
		/// Check if element at index is valid
		/////////////////////////////////////////////////
		bool isValid(size_t idx) const {
			if (idx >= _container->size()) {
				return false;
			}

			// Check if element type has is_active field (hierarchical mesh)
			if constexpr (HierarchicalMeshElement<Element_t>) {
				return (*_container)[idx].is_active;
			} else {
				return true;
			}
		}

		/////////////////////////////////////////////////
		/// Move forward to next valid element
		/////////////////////////////////////////////////
		void advance() {
			if (_curr_idx >= _container->size()) {
				return;
			}

			++_curr_idx;
			while (_curr_idx < _container->size() && !isValid(_curr_idx)) {
				++_curr_idx;
			}
		}

		/////////////////////////////////////////////////
		/// Move backward to previous valid element
		/////////////////////////////////////////////////
		void retreat() {
			if (_curr_idx == 0) {
				return;
			}

			size_t tmp = _curr_idx - 1;
			while (!isValid(tmp)) {
				if (tmp == 0) {
					return;  // Can't retreat further
				}
				--tmp;
			}
			_curr_idx = tmp;
		}

	public:
		/////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////
		
		// Default constructor
		ElementIterator()
			: _mesh(nullptr)
			, _container(nullptr)
			, _curr_idx(0)
		{}

		// Standard constructor
		ElementIterator(Mesh_t* mesh, size_t idx)
			: _mesh(mesh)
			, _curr_idx(idx)
		{
			// Select container based on template parameter
			if constexpr (CONTAINER == ContainerType::ELEMENTS) {
				_container = &(mesh->_elements);
			} else if constexpr (CONTAINER == ContainerType::BOUNDARY) {
				_container = &(mesh->_boundary);
			} else {
				static_assert(CONTAINER == ContainerType::ELEMENTS || 
				             CONTAINER == ContainerType::BOUNDARY,
				             "Unknown ContainerType");
			}

			// Position iterator correctly
			if (idx == 0) {
				moveToBegin();
			} else if (idx >= _container->size()) {
				moveToEnd();
			}
		}

		// Copy constructor
		ElementIterator(const ElementIterator&) = default;

		// Copy assignment
		ElementIterator& operator=(const ElementIterator&) = default;

		/////////////////////////////////////////////////
		/// Iterator operations
		/////////////////////////////////////////////////
		
		// Pre-increment
		ElementIterator& operator++() {
			advance();
			return *this;
		}

		// Post-increment
		ElementIterator operator++(int) {
			ElementIterator tmp = *this;
			advance();
			return tmp;
		}

		// Pre-decrement
		ElementIterator& operator--() {
			retreat();
			return *this;
		}

		// Post-decrement
		ElementIterator operator--(int) {
			ElementIterator tmp = *this;
			retreat();
			return tmp;
		}

		/////////////////////////////////////////////////
		/// Comparison operators
		/////////////////////////////////////////////////
		
		bool operator==(const ElementIterator& other) const {
			return _curr_idx == other._curr_idx && _container == other._container;
		}

		bool operator!=(const ElementIterator& other) const {
			return !(*this == other);
		}

		/////////////////////////////////////////////////
		/// Dereference operators
		/////////////////////////////////////////////////
		
		reference operator*() {
			return (*_container)[_curr_idx];
		}

		const reference operator*() const {
			return (*_container)[_curr_idx];
		}

		pointer operator->() {
			return &((*_container)[_curr_idx]);
		}

		const pointer operator->() const {
			return &((*_container)[_curr_idx]);
		}

		/////////////////////////////////////////////////
		/// Accessors
		/////////////////////////////////////////////////
		
		/// Get current index in container
		size_t index() const {
			return _curr_idx;
		}

		/// Move to beginning (first valid element)
		ElementIterator& moveToBegin() {
			_curr_idx = 0;
			if (_curr_idx < _container->size() && !isValid(_curr_idx)) {
				advance();
			}
			return *this;
		}

		/// Move to end (past last element)
		ElementIterator& moveToEnd() {
			_curr_idx = _container->size();
			return *this;
		}
	};

} // namespace gv::mesh