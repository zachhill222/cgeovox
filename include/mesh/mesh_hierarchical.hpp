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
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement ElementStruct_t = BasicElement,
			ColorMethod                      COLOR_METHOD    = ColorMethod::GREEDY,
			size_t                           MAX_COLORS      = 64
			>
	class HierarchicalMesh : public ColoredMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t,COLOR_METHOD,MAX_COLORS> {
	private:
		using BaseClass = ColoredMesh<space_dim,ref_dim,Scalar_t,ElementStruct_t,COLOR_METHOD,MAX_COLORS>;
		
		mutable std::shared_mutex   _el_split_rw_mtx;   //mutex to lock _elements_to_split
		std::unordered_set<size_t>	_elements_to_split; //indices of elements that are to be refined

	public:
		//aliases
		using typename BaseClass::Index_t;
		using typename BaseClass::DomainBox_t;
		using typename BaseClass::RefBox_t;
		using typename BaseClass::RefPoint_t;
		using typename BaseClass::Point_t;
		using typename BaseClass::Vertex_t;
		using typename BaseClass::VertexList_t;
		using typename BaseClass::ElementIterator_t;
		using typename BaseClass::BoundaryIterator_t;

		//elements and faces have the same storage struct type, but it's nice to see the distinction in the code
		using typename BaseClass::Element_t;
		using typename BaseClass::Face_t;


		/////////////////////////////////////////////////
		/// Pass constructors to BaseClass
		/////////////////////////////////////////////////
		HierarchicalMesh() : BaseClass() {}
		HierarchicalMesh(const DomainBox_t &domain) :  BaseClass(domain) {}

		HierarchicalMesh(const RefBox_t &domain) requires (ref_dim<space_dim) : BaseClass(domain) {}
		
		HierarchicalMesh(const RefBox_t &domain, const Index_t &N, const bool useIsopar=false) : BaseClass(domain, N, useIsopar) {}


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
		/// New vertices will most likely be created and old vertices updated during this process.
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
		template<int space_dim_u, int ref_dim_u,
			 Scalar                           Scalar_u,
			 HierarchicalColorableMeshElement Element_u,
			 ColorMethod                      color_method,
			 size_t                           max_colors>
		friend std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<space_dim_u,ref_dim_u,Scalar_u,Element_u,color_method,max_colors> &mesh);

	private:
		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start, size_t child_face_start);

		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		void splitBoundaryFace_Unlocked(const size_t face_idx, const size_t child_idx_start);


		/////////////////////////////////////////////////
		/// Subroutine for processSplit()
		/////////////////////////////////////////////////
		template<BasicMeshElement Element_u>
		std::vector<size_t> generateNodesForSplit(const Element_u &ELEM) {
			auto* vtk_elem = _VTK_ELEMENT_FACTORY<space_dim,ref_dim,Scalar_t>(ELEM);

			//initialize storage for the values that will be needed to create the children elements
			std::vector<Point_t> child_vertex_coords;
			std::vector<size_t>   split_node_numbers(vtk_n_vertices_when_split(ELEM.vtkID));
			
			//handle parent vertices/vertices that will be re-used
			size_t j;
			for (j=0; j<ELEM.vertices.size(); j++) {
				child_vertex_coords.push_back(this->_vertices[ELEM.vertices[j]].coord);
				split_node_numbers[j] = ELEM.vertices[j];
			}

			//get the verticices of the remaining vertices that the children will need
			vtk_elem->split(child_vertex_coords);

			//add any new vertices
			std::vector<size_t> local_node;
			for (;j<child_vertex_coords.size(); j++) {
				Vertex_t VERTEX(child_vertex_coords[j]);
				size_t n_idx = this->_vertices.push_back_async(std::move(VERTEX));
				this->_vertices[n_idx].index = n_idx;
				split_node_numbers[j] = n_idx;
			}

			delete vtk_elem;
			return split_node_numbers;
		}
	};
	

	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::getElementDescendents_Unlocked(
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


	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::getBoundaryFaceDescendents_Unlocked(
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

	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::joinDescendents(const size_t elem_idx) {
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


	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::splitElement(const size_t elem_idx) {
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

	
	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::splitElement_Unlocked(const size_t elem_idx, const size_t child_idx_start, const size_t child_face_start) {
		Element_t &ELEM = this->_elements[elem_idx];
		std::vector<size_t> split_node_numbers = generateNodesForSplit(ELEM);

		auto* vtk_elem = _VTK_ELEMENT_FACTORY<space_dim,ref_dim,Scalar_t>(ELEM);

		//now all vertices have been created to create the children
		assert(ELEM.is_active);
		ELEM.is_active = false;

		

		for (size_t k=0; k<vtk_n_children(ELEM.vtkID); k++) {
			//get the indices of the vertices that define child k in the correct order
			std::vector<size_t> childNodes;
			vtk_elem->getChildVertices(childNodes, k, split_node_numbers);

			//create the child
			const size_t global_child_index = child_idx_start + k;
			assert(global_child_index<this->_elements.size());
			assert(!this->_elements[global_child_index].is_active);

			Element_t newElem(childNodes, ELEM.vtkID);
			this->insertElement_Unlocked(newElem, global_child_index);
			Element_t &CHILD = this->_elements[global_child_index];
			CHILD.index      = global_child_index;
			CHILD.parent     = ELEM.index;
			CHILD.depth      = ELEM.depth+1;
			ELEM.children.push_back(CHILD.index);
		}


		//split boundary faces
		std::vector<size_t> faces;
		this->getBoundaryFaces_Unlocked(elem_idx, faces);
		int n_faces=0;
		for (size_t f_idx : faces) {
			Face_t &FACE = this->_boundary[f_idx];
			FACE.is_active = false;
			const FaceTracker &TRACKER = this->_boundary_track[f_idx];

			auto* vtk_face = _VTK_ELEMENT_FACTORY<space_dim,ref_dim-1,Scalar_t>(FACE);

			std::vector<size_t> face_split_vertices;
			vtk_elem->getSplitFaceVertices(face_split_vertices, TRACKER.elem_face, split_node_numbers);

			//split the face
			std::vector<size_t> faceChildNodes;
			for (size_t k=0; k<vtk_n_children(FACE.vtkID); k++) {
				vtk_face->getChildVertices(faceChildNodes, k, face_split_vertices);

				//get the index for the new face
				const size_t global_face_child_index = child_face_start + n_faces;
				n_faces++;

				//create the new face
				Face_t newFace(faceChildNodes, FACE.vtkID);

				//determine which child element the new face belongs to
				FaceTracker newTracker {(size_t)-1,-1};
				for (size_t c_idx : ELEM.children) {
					const Element_t &CHILD = this->_elements[c_idx];
					auto* vtk_child = _VTK_ELEMENT_FACTORY<space_dim,ref_dim,Scalar_t>(CHILD);
					for (int cf_idx=0; cf_idx<vtk_n_faces(CHILD.vtkID); cf_idx++) {
						if (newFace == vtk_child->getFace(cf_idx)) {
							newTracker.elem_idx = CHILD.index;
							newTracker.elem_face = cf_idx;
							break;
						}
					}
					delete vtk_child;
					if (newTracker.elem_face!=-1) {break;}
				}
				assert(newTracker.elem_face!=-1);


				//insert the face
				newFace.parent = FACE.index;
				newFace.depth  = FACE.depth+1;
				newFace.index  = global_face_child_index;
				FACE.children.push_back(newFace.index);
				this->insertBoundaryFace_Unlocked(newFace, global_face_child_index, newTracker);
			}
			delete vtk_face;
		}

		//clean up memory
		delete vtk_elem;
	}



	

	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	void HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS>::processSplit() {
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
			}
		}

		
		//split all the elements by color (parallel in each color)
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			

			std::vector<size_t> &this_color_elems = colored_elements_to_split[color];
			if (this_color_elems.empty()) {continue;} //no elements of this color to split
			
			std::vector<size_t> child_element_index_start;
			std::vector<size_t> child_face_index_start;
			//the indices of where to store each child element must be known ahead of time
			//the local child j of element k will be stored at _elements[child_element_index_start[k] + j]
			const size_t nStartingElements = this->_elements.size();
			const size_t nStartingFaces    = this->_boundary.size();

			child_element_index_start.push_back(nStartingElements);
			child_face_index_start.push_back(nStartingFaces);

			size_t nNewFaces    = 0;
			size_t nNewElements = 0;
			size_t maxNewNodes  = 0;

			for (size_t e_idx : this_color_elems) {
				assert(this->_elements[e_idx].color == color);
				assert(this->_elements[e_idx].is_active);

				const Element_t &ELEM = this->_elements[e_idx];
				nNewElements += vtk_n_children(ELEM.vtkID);
				maxNewNodes  += vtk_n_vertices_when_split(ELEM.vtkID) - vtk_n_vertices(ELEM.vtkID);

				std::vector<size_t> faces;
				this->getBoundaryFaces_Unlocked(e_idx, faces);
				for (size_t f_idx : faces) {
					nNewFaces += vtk_n_children(this->_boundary[f_idx].vtkID);
				}

				child_element_index_start.push_back(nStartingElements + nNewElements);
				child_face_index_start.push_back(nStartingFaces + nNewFaces);
			}


			//reserve space and default-initialize the new elements
			this->_elements.resize(nStartingElements+nNewElements);
			this->_boundary.resize(nStartingFaces+nNewFaces);
			this->_boundary_track.resize(nStartingFaces+nNewFaces);
			this->_vertices.resize(this->_vertices.size()+maxNewNodes);

			//decrement the _colorCount by the number of elements that will be deactivated
			this->_color_manager.decrementCount(color,this_color_elems.size());

			//split the elements of this color
			#pragma omp parallel for
			for (size_t i=0; i<this_color_elems.size(); i++) {
				assert(this->_elements[this_color_elems[i]].color == color);
				splitElement_Unlocked(this_color_elems[i], child_element_index_start[i], child_face_index_start[i]);
			}
			this->_vertices.flush(); //ensure vertices are up-to-date before starting the next color
		}
		
		//clean up data structures
		_elements_to_split.clear();
		this->_vertices.shrink_to_fit();
	}




	template<
			int                              space_dim,
			int                              ref_dim,
			Scalar                           Scalar_t,
			HierarchicalColorableMeshElement Element_t,
			ColorMethod                      COLOR_METHOD,
			size_t                           MAX_COLORS
			>
	std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS> &mesh) {
		const ColoredMesh<space_dim,ref_dim,Scalar_t,Element_t,COLOR_METHOD,MAX_COLORS> &base_mesh = mesh;
		os << base_mesh;
		os << mesh._color_manager;
		return os;
	}
}

