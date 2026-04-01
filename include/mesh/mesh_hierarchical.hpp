#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include <vector>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>
#include <thread>
namespace gv::mesh {
	/////////////////////////////////////////////////
	/// This class extends the ColoredMesh class allow elements to be split. In general, the resulting mesh is non-conforming.
	/// In order to track the boundary, the boundary must be initialized from a conforming state. When an element is split,
	/// some number of children elements are created. The children elements have disjoint interiors and their union is the original element.
	/// The original element is the parent of the children elements and this relationship is stored in each element. The number of child
	/// elements depends on the element type. For example, voxels and hexahedra have eight children while pixels and quads have four.
	/// 
	/// @tparam space_dim        The dimension of the space that the mesh is embedded in. Usually 3. (Untested for 2).
	/// @tparam ref_dim          The dimension of the space that the reference elements are embedded in. Usually 3. (2 for surface meshes).
	/// @tparam Scalar_t         The scalar type to emulate the real line. This should be robust under comparisions and arithmetic ordering. (e.g., FixedPrecision instead of float)
	/// @tparam ElementStruct_t  The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam COLOR_METHOD     The method used to color the elements. Either greedy (ColorMethod::GREEDY) or balanced (ColorMethod::BALANCED).
	/// @tparam MAX_COLORS       The maximum number of colors that the mesh can have. Colors are stored in an std::array<std::atomic<size_t>> structure that is not resized.
	/////////////////////////////////////////////////
	template<
			HierarchicalColorableMeshElement Element_type = HierarchicalColoredElement<VOXEL_VTK_ID>,
			Scalar                           Scalar_type  = gutil::FixedPoint64<>,
			ColorMethod                      COLOR_METHOD = ColorMethod::BALANCED,
			size_t                           MAX_COLORS   = 64
			>
	class HierarchicalMesh : public ColoredMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS> {
	private:
		using BASE = ColoredMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>;
		
		mutable std::shared_mutex   _el_split_rw_mtx;   //mutex to lock _elements_to_split

		//indices of elements that are to be refined. Allow classes that have a const reference (e.g. dofhandler) to the mesh
		//to mark elements for refinement, but postpone the actual refinement to a more controlled time.
		//TODO: find a way to remove the mutable tag
		mutable std::unordered_set<size_t>	_elements_to_split;

	public:
		//import aliases
		using typename BASE::Element_t;
		using typename BASE::HalfEdge_t;
		using typename BASE::Vertex_t;
		
		using typename BASE::Scalar_t;
		using typename BASE::GeoPoint_t; //data type for computing spatial coordinates
		using typename BASE::RefPoint_t; //data type for evaluating basis functions, computing jacobians, etc.
		
		using Mesh_t = HierarchicalMesh<Element_type,Scalar_type,COLOR_METHOD,MAX_COLORS>; //type of this mesh
		using typename BASE::Index_t; //index for creating structured mesh in the constructor
		using typename BASE::GeoBox_t; //boxes in the domain space
		using typename BASE::RefBox_t; //boxes in the reference space

		using typename BASE::VertexList_t;
		using ElementLogic_t = VtkElementType_t<Mesh_t, Element_type::VTK_ID>; //type to handle logic of creating children, getting faces, etc.


		/////////////////////////////////////////////////
		/// Pass constructors to BASE
		/////////////////////////////////////////////////
		using BASE::BASE;


		/////////////////////////////////////////////////
		/// Get the total number of active elements in the mesh.
		/////////////////////////////////////////////////
		size_t nElements_active() const override
		{
			size_t nElems = 0;
			for (const Element_t &ELEM : this->_elements) {
				if (ELEM.active) {nElems++;}
			}
			return nElems;
		}


		/////////////////////////////////////////////////
		/// Get a list of all elements that contain a vertex.
		/// This will return indices of elements that contain the 
		/// vertex as a hanging node.
		/////////////////////////////////////////////////
		std::vector<size_t> vertexInElements(const size_t vertex_index) const;

		/////////////////////////////////////////////////
		/// Process the elements in the refinement list
		/////////////////////////////////////////////////
		void processSplit();

		/////////////////////////////////////////////////
		/// Refine all elements that have a vertex in the specified box
		/////////////////////////////////////////////////
		void refineRegion(const GeoBox_t& box);

		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified element.
		///
		/// @param elem_idx The element to get the descendents of.
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get all descendents (rather then just the leaf descendents).
		/////////////////////////////////////////////////
		void getElementDescendents_Unlocked(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly=true) const;
		void getElementAncestors_Unlocked(const size_t elem_idx, std::vector<size_t> &ancestors, const bool activeOnly=false) const;

		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified boundary face.
		///
		/// @param elem_idx The face to get the descendents of.
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get all descendents (rather then just the leaf descendents).
		/////////////////////////////////////////////////
		void getBoundaryFaceDescendents_Unlocked(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly=true) const;


		/////////////////////////////////////////////////
		/// A method to mark an element to be split/refined. The element that is split will have the new elements added as children,
		/// and the new elements that are created will have the element that was split as a parent. The new elements are of the same type as the original.
		/// New vertices will most likely be created and old vertices updated during this process.
		///
		/// This simply checks if an element can be split and marks it as such by adding its index to _elements_to_split.
		/// Note that _element_to_split is mutable, so this method can be called as const.
		///
		/// If elem_idx>=_elements.size(), then _elements_to_split is processed (if it is not empty) and then the elem_idx element is re-examined.
		/// If it is still out of bounds, an exception is thrown.
		///
		/// @param elem_idx The element to be split.
		/////////////////////////////////////////////////
		void splitElement(const size_t elem_idx) const;

		
		/////////////////////////////////////////////////
		/// A method to join/unrefine previously split elements. If _elements[elem_idx] exists, then all of the descendents of that element are de-activated and the element is activated.
		/// The de-activated elements are not deleted. The element is re-colored.
		///
		/// @param elem_idx The element whose descendents are to be joined.
		/////////////////////////////////////////////////
		void joinDescendents(const size_t elem_idx);
		

		/// Friend function to print the mesh information
		// template<HierarchicalColorableMeshType Mesh_type>
		// friend std::ostream& operator<<(std::ostream& os, const Mesh_type& mesh);

	private:
		/////////////////////////////////////////////////
		/// A method to mark an element to be split/refined. The element that is split will have the new elements added as children,
		/// and the new elements that are created will have the element that was split as a parent. The new elements are of the same type as the original.
		/// New vertices will most likely be created and old vertices updated during this process.
		///
		/// If the specified element has already been split and re-joined (i.e., the children exist), then the children are simply activated and
		/// no new elements are created in memory. If this _elements[elem_idx].active is false, then the method returns without making any changes.
		///
		/// An element is a valid refinement target if it is currently active.
		/// If _elements[elem_idx].active is false, then the method immediately returns.
		/// If _elements[elem_idx].children is populated, then the children are activated and re-colored (while the specified element is deactivated).
		/// If _elements[elem_idx].children is empty, then the creation of the children elements is delayed until they are needed and then run in parallel.
		/// These are tracked in _elements_to_split.
		///
		/// If elem_idx>=_elements.size(), then _elements_to_split is processed (if it is not empty) and then the elem_idx element is re-examined.
		/// If it is still out of bounds, an exception is thrown.
		///
		/// @param elem_idx The element to be split.
		///
		/// @todo Add support for more element types.
		/////////////////////////////////////////////////
		void reSplitElement(const size_t elem_idx);

		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start);

		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitBoundaryFace_Unlocked(const size_t face_idx, const size_t child_idx_start);

		//make sure coloring is only done by active elements
		virtual void color_element_async(const size_t e_idx) override
		{
			std::vector<size_t> neighbors = this->get_element_neighbors(e_idx, [](const Element_t& e){return e.active;});
			this->_color_manager.set_color_unlocked(e_idx, neighbors);
		}

		virtual void color_element_sync(const size_t e_idx) override
		{
			std::vector<size_t> neighbors = this->get_element_neighbors(e_idx, [](const Element_t& e){return e.active;});
			this->_color_manager.set_color_locked(e_idx, neighbors);
		}


		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		std::array<size_t, vtk_n_vertices_when_split(BASE::ELEM_VTK_ID)> generateNodesForSplit(const size_t e_idx)
		{
			const Element_t& ELEM = this->_elements[e_idx];
			ElementLogic_t vtk_elem;

			//initialize storage for the values that will be needed to create the children elements
			vtk_elem.set_element(*this, e_idx);
			vtk_elem.set_child_vertices();
			std::array<size_t, vtk_n_vertices_when_split(BASE::ELEM_VTK_ID)> split_node_numbers;

			//handle parent vertices/vertices that will be re-used
			int j;
			for (j=0; j<BASE::N_VERT_PER_ELEMENT; j++) {
				split_node_numbers[j] = ELEM.vertices[j];
			}

			//get the coordinates of the remaining vertices that the children will need
			for (;j<vtk_elem.n_vert_on_split(); j++) {
				//create new vertex and attempt to add it to the list (or get the index of the existing vertex)
				Vertex_t new_vertex(vtk_elem.child_vertex_coords[j]);
				size_t n_idx = this->_vertices.push_back_async(std::move(new_vertex));
				
				split_node_numbers[j] = n_idx;
			}

			return split_node_numbers;
		}
	};

	static_assert(HierarchicalMeshType<HierarchicalMesh<>>);
	

	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::getElementDescendents_Unlocked(
		const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const 
	{	
		const Element_t &ELEM = this->_elements[elem_idx];
		//loop through the children
		for (size_t c_idx : ELEM.children) {
			if (c_idx == (size_t) -1) {break;}
			const Element_t &CHILD = this->_elements[c_idx];

			//add the relevent children
			if (!activeOnly or CHILD.active) {descendents.push_back(c_idx);}

			//recurse if needed
			getElementDescendents_Unlocked(c_idx, descendents, activeOnly);
		}
	}

	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::getElementAncestors_Unlocked(
		const size_t elem_idx, std::vector<size_t> &ancestors, const bool activeOnly) const 
	{	
		const Element_t &ELEM = this->_elements[elem_idx];
		//recurse through the parent
		if (ELEM.parent != (size_t) -1) {
			if (!activeOnly or this->_elements[ELEM.parent].active) {
				ancestors.push_back(ELEM.parent);
				getElementAncestors_Unlocked(ELEM.parent, ancestors, activeOnly);
			}
		}
	}



	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::joinDescendents(const size_t elem_idx) {
		//get the active descendents of the element
		std::vector<size_t> descendents;
		getElementDescendents_Unlocked(elem_idx, descendents, false);

		Element_t &ELEM = this->_elements[elem_idx];
		// std::cout << "joinDescendents " << ELEM << std::endl;

		//de-activate the descendents and any boundary faces
		for (size_t d_idx : descendents) {
			Element_t &DESCENDENT = this->_elements[d_idx];
			DESCENDENT.active = false;
			this->_color_manager.decrementCount(DESCENDENT.color);
			DESCENDENT.color = (size_t) -1;
		}

		//activate and color the element
		ELEM.active = true;
		color_element_async(elem_idx);
		// std::vector<size_t> neighbors;
		// this->get_element_neighbors(elem_idx, neighbors, [](const Element_t& ELEM){return ELEM.active;});
		// this->_color_manager.set_color_unlocked(elem_idx, neighbors);
	}



	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::splitElement(const size_t elem_idx) const {
		//any changes to _elements[elem_idx] are not protected by a unique mutex lock
		//calling splitElement(k) for the same value of k in different threads will lead to undefined behavior
		//calling splitElement(k) for different values of k in defferent threads is safe
		const size_t nElements = this->_elements.size();

		//ensure that the element exists
		if (elem_idx >= nElements) {
			throw std::runtime_error("Index " + std::to_string(elem_idx) + " is out of bounds (_elements.size()= " + std::to_string(nElements) + ")");
			return;
		}

		//ensure that the element is active
		const Element_t &ELEM = this->_elements[elem_idx];
		if (!ELEM.active) {return;}

		_elements_to_split.insert(elem_idx); //TODO: remove mutable from _elements_to_split
	}


	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::reSplitElement(const size_t elem_idx) {
		assert(elem_idx < this->_elements.size());
		Element_t& ELEM = this->_elements[elem_idx];

		assert(ELEM.active);
		assert(ELEM.children[0]!=(size_t) -1);

		ELEM.active = false;
		this->_color_manager.decrementCount(ELEM.color); //when coloring using BALANCED, the parent element should no longer count.

		//activate and color the children
		std::vector<size_t> neighbors;
		for (size_t c_idx: ELEM.children) {
			Element_t &CHILD = this->_elements[c_idx];
			CHILD.active  = true;

			color_element_async(c_idx);
			// neighbors.clear();
			// this->get_element_neighbors(c_idx, neighbors, [](const Element_t& ELEM){return ELEM.active;});
			// this->_color_manager.set_color_unlocked(c_idx, neighbors);
		}
	}

	
	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start)
	{
		Element_t &ELEM = this->_elements[elem_idx];
		assert(ELEM.active);

		//check if this element has been split previously
		if (ELEM.children[0] != (size_t) -1) {
			reSplitElement(elem_idx);
			return;
		}


		//new vertices and elements must be generated
		auto split_node_numbers = generateNodesForSplit(elem_idx);
		
		ElementLogic_t vtk_elem;
		vtk_elem.set_element(*this, elem_idx);

		//now all vertices have been created to create the children
		ELEM.active = false;

		for (int k=0; k<vtk_elem.n_children(); k++) {
			//get the indices of the vertices that define child k in the correct order
			auto local_child_nodes = vtk_elem.get_child_local_vertices(k);

			//create the child
			const size_t global_child_index = child_idx_start + k;
			assert(global_child_index<this->_elements.size());
			assert(!this->_elements[global_child_index].active);

			Element_t newElem;
			for (int i=0; i<vtk_elem.n_vertices(); ++i) {
				newElem.vertices[i] = split_node_numbers[local_child_nodes[i]];
			}

			this->insert_element_async(std::move(newElem), global_child_index);
			Element_t &CHILD = this->_elements[global_child_index];
			CHILD.parent     = elem_idx;
			CHILD.depth      = ELEM.depth+1;
			CHILD.active     = true;
			ELEM.children[k] = global_child_index;
		}
	}



	

	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::processSplit() {
		//This method refines all elements in _elements_to_split by color batches
		//Each batch consists of all elements with a specific color
		//No two elements in the same color batch share a node
		//Each color batch is processed in parallel
		//There are some race conditions when using balanced coloring, so the optimal color may not be chosen for the children elements
		//The chosen color for the children elements is guarenteed to be valid and only new colors are generated when strictly necessary

		if (_elements_to_split.empty()) {return;}
		assert(this->are_colors_valid());

		//partition the specified that are active by their color
		//if any element has been split already and the children are already stored in the mesh, activate them
		std::array<std::vector<size_t>, MAX_COLORS> colored_elements_to_split;
		{
			for (size_t e_idx: _elements_to_split) {
				assert(e_idx<this->_elements.size());
				Element_t &ELEM = this->_elements[e_idx];
				
				//elements should heve been filtered before being put into _elements_to_split
				assert(ELEM.active);
				assert(ELEM.children[0]==(size_t) -1);

				//element is to be split and the children must be computed
				assert(ELEM.color!=(size_t) -1);
				colored_elements_to_split[ELEM.color].push_back(e_idx);
			}
		}

		
		//split all the elements by color (parallel in each color)
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			

			const std::vector<size_t>& this_color_elems = colored_elements_to_split[color];
			if (this_color_elems.empty()) {continue;} //no elements of this color to split
			
			std::vector<size_t> child_element_index_start;
			//the indices of where to store each child element must be known ahead of time
			//the local child j of element k will be stored at _elements[child_element_index_start[k] + j]
			const size_t nStartingElements = this->_elements.size();
			
			child_element_index_start.push_back(nStartingElements);
			
			size_t nNewElements = 0;
			size_t maxNewNodes  = 0;

			for (size_t e_idx : this_color_elems) {
				assert(this->_elements[e_idx].color == color);
				assert(this->_elements[e_idx].active);

				const Element_t &ELEM = this->_elements[e_idx];
				nNewElements += vtk_n_children(ELEM.VTK_ID);
				maxNewNodes  += vtk_n_vertices_when_split(ELEM.VTK_ID) - vtk_n_vertices(ELEM.VTK_ID);

				child_element_index_start.push_back(nStartingElements + nNewElements);
			}


			//reserve space and default-initialize the new elements
			this->elements_resize(nStartingElements+nNewElements);
			this->vertices_resize(this->_vertices.size()+maxNewNodes);

			//decrement the _colorCount by the number of elements that will be deactivated
			this->_color_manager.decrementCount(color,this_color_elems.size());

			//split the elements of this color
			#pragma omp parallel for
			for (size_t i=0; i<this_color_elems.size(); i++) {
				assert(this->_elements[this_color_elems[i]].color == color);
				splitElement_Unlocked(this_color_elems[i], child_element_index_start[i]);
			}
			this->_vertices.flush(); //ensure vertices are up-to-date before starting the next color
		}
		
		//all elements are split, go through all the new children and update halfedges
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			const std::vector<size_t>& this_color_elems = colored_elements_to_split[color];
			
			#pragma omp parallel for
			for (size_t e_idx : this_color_elems) {
				const Element_t& ELEM = this->_elements[e_idx];
				this->pair_halfedges_for_elements(ELEM.children);
			}
		}

		// this->pair_halfedges();
		//clean up data structures
		_elements_to_split.clear();
		this->_vertices.shrink_to_fit();
	}




	template<
			HierarchicalColorableMeshElement Element_type,
			Scalar                           Scalar_type,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<Element_type, Scalar_type, COLOR_METHOD, MAX_COLORS>::refineRegion(const GeoBox_t& box) {
		#pragma omp parallel for
		for (size_t e_idx=0; e_idx<this->_elements.size(); ++e_idx) {
			const Element_t& ELEM = this->_elements[e_idx];
			if (!ELEM.active) {continue;}

			for (size_t v_idx : ELEM.vertices) {
				if (box.contains(this->_vertices[v_idx].coord)) {
					#pragma omp critical
					{
						splitElement(e_idx);
					}
					break;
				}
			}
		}

		processSplit();
	}


	// template<HierarchicalColorableMeshType Mesh_type>
	// std::ostream& operator<<(std::ostream& os, const Mesh_type& mesh) {
	// 	const typename Mesh_type::BASE& base_mesh = mesh;
	// 	os << base_mesh;
	// 	return os;
	// }
}

