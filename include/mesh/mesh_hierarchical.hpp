#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_colored.hpp"

#include "mesh/vtk_elements.hpp"
#include "mesh/vtk_defs.hpp"

#include "util/point.hpp"
#include "util/octree.hpp"
#include "util/box.hpp"

#include <vector>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>

namespace gv::mesh {
	/////////////////////////////////////////////////
	/// This class extends the ColoredMesh class allow elements to be split. In general, the resulting mesh is non-conforming.
	/// In order to track the boundary, the boundary must be initialized from a conforming state. When an element is split,
	/// some number of children elements are created. The children elements have disjoint interiors and their union is the original element.
	/// The original element is the parent of the children elements and this relationship is stored in each element. The number of child
	/// elements depends on the element type. For example, voxels and hexahedra have eight children while pixels and quads have four.
	/// 
	/// @tparam Node_t       The type of node to use. Usually BasicNode<gv::util::Point<3,double>>.
	/// @tparam Element_t    The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam Face_t       The type of boundary element to use. This is usually set by the class that inherits from this class.
	/// @tparam COLOR_METHOD The method used to color the elements. Either greedy (ColorMethod::GREEDY) or balanced (ColorMethod::BALANCED).
	/// @tparam MAX_COLORS   The maximum number of colors that the mesh can have. Colors are stored in an std::array<std::atomic<size_t>> structure that is not resized.
	/////////////////////////////////////////////////
	template<BasicMeshNode                    Node_t       = BasicNode<gv::util::Point<3,double>>,
			 HierarchicalColorableMeshElement Element_t    = HierarchicalColoredElement,
			 HierarchicalMeshElement          Face_t       = HierarchicalElement,
			 ColorMethod                      COLOR_METHOD = ColorMethod::GREEDY,
			 size_t                           MAX_COLORS   = 32>
	class HierarchicalMesh : public ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS> {
	private:
		using BaseClass = ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>;
		mutable std::shared_mutex   _el_split_rw_mtx;   //mutex to lock _elements_to_split
		std::unordered_set<size_t>	_elements_to_split; //indices of elements that are to be refined
	public:
		//aliases
		template<int n=3>
		using Index_t            = gv::util::Point<n,size_t>;
		template<int n=3>
		using Box_t              = gv::util::Box<n, typename Node_t::Scalar_t>;
		using Vertex_t           = Node_t::Vertex_t;

		/////////////////////////////////////////////////
		/// Pass constructors to BaseClass
		/////////////////////////////////////////////////
		HierarchicalMesh() : BaseClass() {}
		HierarchicalMesh(const Box_t<3> &domain, const Index_t<3> &N, const bool useIsopar=false) : BaseClass(domain, N, useIsopar) {}
		HierarchicalMesh(const Box_t<2> &domain, const Index_t<2> &N, const bool useIsopar=false) : BaseClass(domain, N, useIsopar) {}


		/////////////////////////////////////////////////
		/// Get the total number of active elements in the mesh.
		/////////////////////////////////////////////////
		size_t nElems() const override {
			size_t nElems = 0;
			for (const Element_t &ELEM : this->_elements) {
				if (ELEM.is_active) {nElems++;}
			}
			return nElems;
		}


		/////////////////////////////////////////////////
		/// Get the total number of boundary faces in the mesh.
		/////////////////////////////////////////////////
		size_t nBoundaryFaces() const override {
			size_t nFaces = 0;
			for (const Face_t &FACE : this->_boundary) {
				if (FACE.is_active) {nFaces++;}
			}
			return nFaces;
		}





		/////////////////////////////////////////////////
		/// Process the elements in the refinement list
		/////////////////////////////////////////////////
		void processSplit();


		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified element.
		///
		/// @param elem_idx The element to get the descendents of.
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get all descendents (rather then just the leaf descendents).
		/////////////////////////////////////////////////
		void getElementDescendents_Unlocked(const size_t elem_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;
		

		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified boundary face.
		///
		/// @param elem_idx The face to get the descendents of.
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get all descendents (rather then just the leaf descendents).
		/////////////////////////////////////////////////
		void getBoundaryFaceDescendents_Unlocked(const size_t elem_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;


		/////////////////////////////////////////////////
		/// A method to mark an element to be split/refined. The element that is split will have the new elements added as children,
		/// and the new elements that are created will have the element that was split as a parent. The new elements are of the same type as the original.
		/// New nodes will most likely be created and old nodes updated during this process.
		///
		/// If the specified element has already been split and re-joined (i.e., the children exist), then the children are simply activated and
		/// no new elements are created in memory. If this _elements[elem_idx].is_active is false, then the method returns without making any changes.
		///
		/// An element is a valid refinement target if it is currently active.
		/// If _elements[elem_idx].is_active is false, then the method immediately returns.
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
		void splitElement(const size_t elem_idx);
		

		/////////////////////////////////////////////////
		/// A method to join/unrefine previously split elements. If _elements[elem_idx] exists, then all of the descendents of that element are de-activated and the element is activated.
		/// The de-activated elements are not deleted. The element is re-colored.
		///
		/// @param elem_idx The element whose descendents are to be joined.
		/////////////////////////////////////////////////
		void joinDescendents(const size_t elem_idx);
		

		/////////////////////////////////////////////////
		/// Check if an element is valid/active.
		///
		/// @param ELEM The element to check.
		/////////////////////////////////////////////////
		virtual bool isElementValid(const Element_t &ELEM) const {return ELEM.is_active;}
		

		/////////////////////////////////////////////////
		/// Check if a boundary face is valid/active.
		///
		/// @param FACE The face to check.
		/////////////////////////////////////////////////
		virtual bool isFaceValid(const Face_t &FACE) const {return FACE.is_active;}


		/// Friend function to print the mesh information
		template<BasicMeshNode                Node_u,
			 HierarchicalColorableMeshElement Element_u,
			 HierarchicalMeshElement          Face_u,
			 ColorMethod                      color_method,
			 size_t                           max_colors>
		friend std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<Node_u,Element_u,Face_u,color_method,max_colors> &mesh);

	private:
		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start);

		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitBoundaryFace_Unlocked(const size_t face_idx);


		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		template<BasicMeshElement Element_u>
		void generateNodesForSplit(const Element_u &ELEM) {
			VTK_ELEMENT<Vertex_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Vertex_t>(ELEM);

			//initialize storage for the values that will be needed to create the children elements
			std::vector<Vertex_t> child_vertex_coords;
			std::vector<size_t>   child_node_idx(vtk_n_nodes_when_split(ELEM.vtkID));
			
			//handle parent nodes/vertices that will be re-used
			size_t j;
			for (j=0; j<ELEM.nodes.size(); j++) {
				child_vertex_coords.push_back(this->_nodes[ELEM.nodes[j]].vertex);
				child_node_idx[j] = ELEM.nodes[j];
			}

			//get the verticices of the remaining nodes that the children will need
			vtk_elem->split(child_vertex_coords);

			//add any new nodes
			std::vector<size_t> local_node;
			for (;j<child_vertex_coords.size(); j++) {
				Node_t NODE(child_vertex_coords[j]);
				size_t n_idx = this->_nodes.push_back_async(std::move(NODE));
				this->_nodes[n_idx].index = n_idx;
			}

			delete vtk_elem;
		}
	};
	

	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::getElementDescendents_Unlocked(
		const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const 
	{	
		const Element_t &ELEM = this->_elements[elem_idx];
		//loop through the children
		for (size_t c_idx : ELEM.children) {
			const Element_t &CHILD = this->_elements[c_idx];

			//add the relevent children
			if (!activeOnly or CHILD.is_active) {descendents.push_back(c_idx);}

			//recurse if needed
			if (!CHILD.children.empty()) {getElementDescendents_Unlocked(c_idx, descendents, activeOnly);}
		}
	}


	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::getBoundaryFaceDescendents_Unlocked(
		const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const 
	{	
		const Face_t &ELEM = this->_boundary[elem_idx];
		//loop through the children
		for (size_t c_idx : ELEM.children) {
			const Face_t &CHILD = this->_boundary[c_idx];

			//add the relevent children
			if (!activeOnly or CHILD.is_active) {descendents.push_back(c_idx);}

			//recurse if needed
			if (CHILD.children.size()>0) {getBoundaryFaceDescendents_Unlocked(c_idx, descendents, activeOnly);}
		}
	}

	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::joinDescendents(const size_t elem_idx) {
		//get the active descendents of the element
		std::vector<size_t> descendents;
		getElementDescendents_Unlocked(elem_idx, descendents, false);

		Element_t &ELEM = this->_elements[elem_idx];
		// std::cout << "joinDescendents " << ELEM << std::endl;

		//de-activate the descendents and any boundary faces
		for (size_t d_idx : descendents) {
			Element_t &DESCENDENT = this->_elements[d_idx];
			DESCENDENT.is_active = false;
			this->_color_manager.decrementCount(DESCENDENT.color);
		}

		//activate and color the element
		ELEM.is_active = true;

		std::vector<size_t> neighbors;
		this->getElementNeighbors_Unlocked(elem_idx, neighbors);
		this->_color_manager.setColor_Unlocked(elem_idx, neighbors);

		//get the descendents of any boundary faces
		std::vector<size_t> boundary_faces;
		this->getBoundaryFaces_Unlocked(elem_idx, boundary_faces);
		for (size_t f_idx : boundary_faces) {
			Face_t &FACE = this->_boundary[f_idx];

			std::vector<size_t> descendents;
			getBoundaryFaceDescendents_Unlocked(f_idx, descendents, true);

			//deactivate the descendent boundary elements
			for (size_t d_idx : descendents) {
				this->_boundary[d_idx].is_active = false;
				//TODO: check if we need to change color counts
			}

			//activate the boundary faces of the coarse element
			if (FACE.depth == ELEM.depth) {
				FACE.is_active = true;	
			}

			//update the color of the boundary element
			// this->_boundary[boundary_faces[f_idx]].color = this->_elements[elem_idx].color;
		}
	}


	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::splitElement(const size_t elem_idx) {
		//any changes to _elements[elem_idx] are not protected by a unique mutex lock
		//calling splitElement(k) for the same value of k in different threads will lead to undefined behavior
		//calling splitElement(k) for different values of k in defferent threads is safe
		const size_t nElements = nElems();

		//if the specified does not exist, process any elements that need to be refined
		if (elem_idx >= nElements) {
			bool can_process_elements = !_elements_to_split.empty();
			if (can_process_elements) {processSplit();}
		}
		

		//ensure that the element exists
		if (elem_idx >= nElements) {
			throw std::runtime_error("Index " + std::to_string(elem_idx) + " is out of bounds (_elements.size()= " + std::to_string(nElements) + ")");
			return;
		}


		//ensure that the element is active
		Element_t &ELEM = this->_elements[elem_idx];
		if (!ELEM.is_active) {return;}


		//if the element has been previously refined, activate and re-color its children
		if (!ELEM.children.empty()) {
			ELEM.is_active = false;
			this->_color_manager.decrementCount(ELEM.color); //when coloring using BALANCED, the parent element should no longer count.

			//activate and color the children
			for (size_t e_idx: ELEM.children) {
				Element_t &CHILD = this->_elements[e_idx];
				CHILD.is_active  = true;

				std::vector<size_t> neighbors;
				this->getElementNeighbors_Unlocked(elem_idx, neighbors);
				this->_color_manager.setColor_Unlocked(elem_idx, neighbors);
			}

			//update any boundary faces
			std::vector<size_t> boundary_faces;
			this->getBoundaryFaces_Unlocked(elem_idx, boundary_faces);
			for (size_t f_idx : boundary_faces) {
				Face_t &FACE   = this->_boundary[f_idx];
				FACE.is_active = false;

				assert(!FACE.children.empty());
				for (size_t cf_idx : FACE.children) {
					Face_t &CHILDFACE = this->_boundary[cf_idx];
					if (CHILDFACE.depth == ELEM.depth) {
						CHILDFACE.is_active  = true;
						//TODO: check if we need to color the boundary faces
					}
				}
			}
		} else {
			_elements_to_split.insert(elem_idx);
		}
	}

	
	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start) {
		Element_t &ELEM = this->_elements[elem_idx];
		VTK_ELEMENT<Vertex_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Vertex_t>(ELEM);

		//initialize storage for the values that will be needed to create the children elements
		std::vector<Vertex_t> child_vertex_coords;
		std::vector<size_t>   child_node_idx(vtk_n_nodes_when_split(ELEM.vtkID));
		
		//handle parent nodes/vertices that will be re-used
		size_t j;
		for (j=0; j<ELEM.nodes.size(); j++) {
			child_vertex_coords.push_back(this->_nodes[ELEM.nodes[j]].vertex);
			child_node_idx[j] = ELEM.nodes[j];
		}

		//get the verticices of the remaining nodes that the children will need
		vtk_elem->split(child_vertex_coords);

		//find any nodes that already exist and collect which nodes need to be added to _nodes
		std::vector<size_t> local_node;
		for (;j<child_vertex_coords.size(); j++) {
			Node_t NODE(child_vertex_coords[j]);
			// size_t n_idx = this->_nodes.push_back_async(std::move(NODE));
			size_t n_idx = this->_nodes.find(NODE);
			assert(n_idx<this->_nodes.size());
			// this->_nodes[n_idx].index = n_idx;
			child_node_idx[j] = n_idx;
		}

		//now all nodes have been created to create the children
		ELEM.is_active = false;

		for (size_t k=0; k<vtk_n_children(ELEM.vtkID); k++) {
			//get the indices of the nodes that define child k in the correct order
			std::vector<size_t> childNodes;
			vtk_elem->getChildNodes(childNodes, k, child_node_idx);

			//create the child
			const size_t global_child_index = child_idx_start + k;
			assert(global_child_index<this->_elements.size());
			Element_t newElem(childNodes, ELEM.vtkID);
			this->insertElement_Unlocked(newElem, global_child_index);
			Element_t &CHILD = this->_elements[global_child_index];
			CHILD.index      = global_child_index;
			CHILD.parent     = ELEM.index;
			CHILD.depth      = ELEM.depth+1;
			ELEM.children.push_back(CHILD.index);
			// std::cout << "new child:\n" << CHILD << std::endl;
		}

		//clean up memory
		delete vtk_elem;
	}



	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::splitBoundaryFace_Unlocked(const size_t face_idx) {

	}


	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	void HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>::processSplit() {
		//This method refines all elements in _elements_to_split by color batches
		//Each batch consists of all elements with a specific color
		//No two elements in the same color batch share a node
		//Each color batch is processed in parallel
		//There are some race conditions when using balanced coloring, so the optimal color may not be chosen for the children elements
		//The chosen color for the children elements is guarenteed to be valid and only new colors are generated when strictly necessary

		if (_elements_to_split.empty()) {return;}
		assert(this->colorsValid_Unlocked());

		//partition the specified that are active by their color
		//if any element has been split already and the children are already stored in the mesh, activate them
		std::vector<std::vector<size_t>> colored_elements_to_split;
		{
			colored_elements_to_split.resize(this->_color_manager.nColors());
			for (size_t e_idx: _elements_to_split) {
				assert(e_idx<this->_elements.size());
				Element_t &ELEM = this->_elements[e_idx];
				
				//elements should heve been filtered before being put into _elements_to_split
				assert(ELEM.is_active);
				assert(ELEM.children.empty());

				//element is to be split and the children must be computed
				assert(ELEM.color<colored_elements_to_split.size());
				colored_elements_to_split[ELEM.color].push_back(ELEM.index);

				//create the required nodes
				// generateNodesForSplit(ELEM);
			}
		}

		//clear _elements_to_split
		_elements_to_split.clear();
		
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			std::vector<size_t> &this_color_elems = colored_elements_to_split[color];

			if (this_color_elems.empty()) {continue;} //no elements of this color to split

			std::vector<size_t> child_element_index_start;
			std::vector<size_t> boundary_face_index_start;

			//the indices of where to store each child element must be known ahead of time
			//the local child j of element k will be stored at _elements[child_element_index_start[k] + j]
			const size_t nStartingElements = this->_elements.size();
			const size_t nStartingBoundary = this->_boundary.size();

			child_element_index_start.push_back(nStartingElements);
			boundary_face_index_start.push_back(nStartingBoundary);

			size_t nNewElements = vtk_n_children(this->_elements[this_color_elems[0]].vtkID);
			size_t maxNewNodes  = vtk_n_nodes_when_split(this->_elements[this_color_elems[0]].vtkID);
			size_t nNewBoundaryFaces = 0;

			std::vector<size_t> boundary_faces;
			this->getBoundaryFaces_Unlocked(this_color_elems[0], boundary_faces);
			for (size_t f_idx : boundary_faces) {
				nNewBoundaryFaces += vtk_n_children(this->_boundary[f_idx].vtkID);
			}


			//loop through elements and compute child offsets
			for (size_t i=1; i<this_color_elems.size(); i++) {
				const Element_t &CUR_ELEM     = this->_elements[this_color_elems[i]];
				const Element_t &PREV_ELEM    = this->_elements[this_color_elems[i-1]];
				const size_t prev_children    = vtk_n_children(PREV_ELEM.vtkID);
				const size_t prev_index_start = child_element_index_start[i-1];
				
				child_element_index_start.push_back(prev_index_start + prev_children);
				boundary_face_index_start.push_back(nNewBoundaryFaces);

				nNewElements += vtk_n_children(CUR_ELEM.vtkID);
				maxNewNodes  += vtk_n_nodes_when_split(CUR_ELEM.vtkID) - vtk_n_nodes(CUR_ELEM.vtkID);

				boundary_faces.clear();
				this->getBoundaryFaces_Unlocked(this_color_elems[0], boundary_faces);
				for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
					nNewBoundaryFaces += vtk_n_children(this->_boundary[boundary_faces[f_idx]].vtkID);
				}
			}

			//reserve space and default-initialize the new elements
			this->_elements.resize(nStartingElements+nNewElements);
			this->_nodes.resize(this->_nodes.size()+maxNewNodes);
			// this->_nodes.reserve_buffer(maxNewNodes/10);
			this->_boundary.reserve(nStartingBoundary+nNewBoundaryFaces);

			//decrement the _colorCount by the number of elements that will be deactivated
			this->_color_manager.decrementCount(color,this_color_elems.size());


			//generate the required nodes
			#pragma omp parallel for
			for (size_t i=0; i<this_color_elems.size(); i++) {
				Element_t &ELEM = this->_elements[this_color_elems[i]];
				generateNodesForSplit(ELEM);
			}
			this->_nodes.flush();

			//split the elements of this color
			#pragma omp parallel for
			for (size_t i=0; i<this_color_elems.size(); i++) {
				//split the element
				splitElement_Unlocked(this_color_elems[i], child_element_index_start[i]);
				Element_t &ELEM = this->_elements[this_color_elems[i]];

				//split any boundary faces
				// std::vector<size_t> parent_boundary_faces;
				// this->getBoundaryFaces_Unlocked(ELEM.index, parent_boundary_faces);
				// for (size_t pf_idx : parent_boundary_faces) {
				// 	#pragma omp critical //TODO: make this thread safe if needed.
				// 	{
				// 		assert(pf_idx<this->_boundary.size());
				// 		//ensure that _boundary does not re-size and invalidate references
				// 		while (this->_boundary.capacity() < this->_boundary.size() + vtk_n_children(this->_boundary[pf_idx].vtkID)) {
				// 			this->_boundary.reserve(2*this->_boundary.capacity());
				// 		}

				// 		Face_t &FACE    = this->_boundary[pf_idx];
				// 		FACE.is_active  = false;

				// 		VTK_ELEMENT<Vertex_t>* vtk_face = _VTK_ELEMENT_FACTORY<Vertex_t>(FACE);
				// 		//find the vertices for the split face
				// 		std::vector<Vertex_t> child_vertex_coords;
				// 		std::vector<size_t>  child_node_idx(vtk_n_nodes_when_split(FACE.vtkID));
				// 		size_t i;
				// 		for (i=0; i<FACE.nodes.size(); i++) {
				// 			child_vertex_coords.push_back(this->_nodes[FACE.nodes[i]].vertex);
				// 			child_node_idx[i] = FACE.nodes[i];
				// 		}
				// 		vtk_face->split(child_vertex_coords);

				// 		//get any new nodes
				// 		for (;i<child_vertex_coords.size(); i++) {
				// 			Node_t NODE(child_vertex_coords[i]);
				// 			child_node_idx[i] = this->_nodes.find(NODE); //the node must have been generated by splitting the element
				// 			if (child_node_idx[i] >= this->_nodes.size()) {
				// 				throw std::runtime_error("Couldn't find the node in boundary face: " + std::to_string(FACE.index));
				// 			}
				// 		}


				// 		//create the children faces
				// 		for (size_t j=0; j<vtk_n_children(FACE.vtkID); j++) {
				// 			std::vector<size_t> childNodes;
				// 			vtk_face->getChildNodes(childNodes, j, child_node_idx);
				// 			Element_t CHILDFACE(childNodes, FACE.vtkID);
				// 			CHILDFACE.parent = (size_t) -1;

				// 			//determine the child element that this split face is a child of
				// 			for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
				// 				const Element_t &CHILD = this->_elements[ELEM.children[c_idx]];
				// 				const VTK_ELEMENT<Vertex_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Vertex_t>(CHILD);
				// 				for (int k=0; k<vtk_n_faces(CHILD.vtkID); k++) {
				// 					if (CHILDFACE == vtk_elem->getFace(k)) {
				// 						CHILDFACE.parent = CHILD.index;
				// 						CHILDFACE.color  = CHILD.color;
				// 						break;
				// 					}
				// 				}
				// 				delete vtk_elem;
				// 			}

				// 			//add the face to the _boundary
				// 			assert(CHILDFACE.parent<this->_elements.size());
				// 			CHILDFACE.index = this->_boundary.size();
				// 			FACE.children.push_back(CHILDFACE.index);
				// 			this->_boundary.push_back(std::move(CHILDFACE));
				// 		}

				// 		//clean up memory
				// 		delete vtk_face;
				// 	}
				// }
			}
			// this->_nodes.flush();
		}
		this->_nodes.shrink_to_fit();
	}




	template<BasicMeshNode                    Node_t,
			 HierarchicalColorableMeshElement Element_t,
			 HierarchicalMeshElement          Face_t,
			 ColorMethod                      COLOR_METHOD,
			 size_t                           MAX_COLORS>
	std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS> &mesh) {
		const ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS> &base_mesh = mesh;
		os << base_mesh;
		// operator<<(os, static_cast<const ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>&>(mesh));
		return os;
	}
}

