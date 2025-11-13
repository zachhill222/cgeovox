#pragma once

#include "mesh/mesh_util.hpp"
#include "mesh/basic_mesh.hpp"
#include "mesh/colored_mesh.hpp"

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

namespace gv::mesh
{
	/////////////////////////////////////////////////
	/// This class extends the BasicMesh class to color the elements. The coloring is handled by the _color_manager. Each element has
	/// an ELEM.color field of type size_t. Any two elements in the mesh with the same color are guaranteed to have no nodes in common.
	/// When the ColorMethod::GREEDY is used, each element will recieve the first (lowest) valid color value. When the ColorMethod::BALANCED
	/// is used, each element will recieve the valid color with associated with the least number of elements.
	/// 
	/// @tparam Node_t       The type of node to use. Usually BasicNode<gv::util::Point<3,double>>.
	/// @tparam Element_t    The type of element to use. This is usually set by the class that inherits from this class.
	/// @tparam Face_t       The type of boundary element to use. This is usually set by the class that inherits from this class.
	/// @tparam COLOR_METHOD The method used to color the elements. Either greedy (ColorMethod::GREEDY) or balanced (ColorMethod::BALANCED).
	/// @tparam MAX_COLORS   The maximum number of colors that the mesh can have. Colors are stored in an std::array<std::atomic<size_t>> structure that is not resized.
	/////////////////////////////////////////////////
	template<BasicMeshNode                    Node_t       = BasicNode<gv::util::Point<3,double>>,
			 HierarchicalColorableMeshElement Element_t    = ColoredElement,
			 HierarchicalMeshElement          Face_t       = BasicElement,
			 ColorMethod                      COLOR_METHOD = ColorMethod::GREEDY,
			 size_t                           MAX_COLORS   = 32>
	class HierarchicalMesh : public ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS> {
	private:
		using BaseClass = ColoredMesh<Node_t,Element_t,Face_t,COLOR_METHOD,MAX_COLORS>;

		std::unordered_set<size_t>	_element_refine_list; //indices of elements that are to be refined
	public:
		/////////////////////////////////////////////////
		/// Pass constructors to BaseClass
		/////////////////////////////////////////////////
		HierarchicalMesh() : BaseClass() {}
		HierarchicalMesh(const Box_t<3> &domain, const Index_t<3> &N, const bool useIsopar=false) : BaseClass(domain, N, useIsopar) {}
		HierarchicalMesh(const Box_t<2> &domain, const Index_t<2> &N, const bool useIsopar=false) : BaseClass(domain, N, useIsopar) {}


		/////////////////////////////////////////////////
		/// Get the total number of active elements in the mesh.
		/////////////////////////////////////////////////
		size_t nElems_Locked() const override {
			std::shared_lock<std::shared_mutex> lock(this->_rw_mtx);
			size_t nElems = 0;
			for (const Element_t &ELEM : this->_elements) {
				if (ELEM.is_active) {nElems++;}
			}
			return nElems;
		}


		/////////////////////////////////////////////////
		/// Get the total number of boundary faces in the mesh.
		/////////////////////////////////////////////////
		size_t nBoundaryFaces_Locked() const override {
			std::shared_lock<std::shared_mutex> lock(this->_rw_mtx);
			size_t nFaces = 0;
			for (const Face_t &FACE : this->_boundary) {
				if (FACE.is_active) {nFaces++;}
			}
			return nFaces;
		}


		/////////////////////////////////////////////////
		/// Process the elements in the refinement list
		/////////////////////////////////////////////////
		void process_refinement();


		/////////////////////////////////////////////////
		/// A method to get the descendent elements of the specified element.
		///
		/// @param elem_idx The index of the requested element (i.e., _elements[elem_idx]).
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get only the active descendents
		/////////////////////////////////////////////////
		void get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;
		

		/////////////////////////////////////////////////
		/// A method to get the descendent faces of the specified boundary face.
		///
		/// @param face_idx The index of the requested element (i.e., _boundary[face_idx]).
		/// @param descendents A reference to an existing vector where the result will be stored
		/// @param activeOnly Optionally, the user can get only the active descendents
		/////////////////////////////////////////////////
		void get_boundary_face_descendents(const size_t face_idx, std::vector<size_t> &descendents=false, const bool activeOnly=true) const;


		/////////////////////////////////////////////////
		/// A method to mark an element to be split/refined. The element that is split will have the new elements added as children, and the new elements that are created
		/// will have the element that was split as a parent. The new elements are of the same type as the original.
		/// New nodes will most likely be created and old nodes updated during this process.
		/// For certain elements (i.e., voxels or hexahedrons) there will likely be more than one new node created and there is no guarentee that the mesh will be conformal.
		/// If the specified element has already been split and re-joined (i.e., the children exist), then the children are simply activated and no new elements are created in memory.
		/// If this _elements[elem_idx].is_active is false, then the method returns without making any changes.
		///
		/// An element is a valid refinement target if it is currently active.
		/// If _elements[elem_idx].is_active is false, then the method immediately returns.
		/// If _elements[elem_idx].children is populated, then the children are activated and re-colored (while the specified element is deactivated).
		/// If _elements[elem_idx].children is empty, then the creation of the children elements is delayed until they are needed and then run in parallel. This is done by inserting elem_idx
		///    into _element_refine_list.
		///
		/// If elem_idx>=_elements.size(), then _element_refine_list is processed (if it is not empty) and then the elem_idx element is re-examined. If it is still out of bounds,
		///    an exception is thrown.
		///
		/// @param elem_idx The element to be split.
		///
		/// @todo Add support for more element types.
		/////////////////////////////////////////////////
		void split_element(const size_t elem_idx);
		

		/////////////////////////////////////////////////
		/// A method to join/unrefine previously split elements. If _elements[elem_idx] exists, then all of the descendents of that element are de-activated and the element is activated.
		/// The de-activated elements are not deleted. The element is re-colored.
		///
		/// @param elem_idx The element whose descendents are to be joined.
		/////////////////////////////////////////////////
		void join_descendents(const size_t elem_idx);
		

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
		template <Float U>
		friend std::ostream& operator<<(std::ostream& os, const HierarchicalMesh<U> &mesh);
	};
	

	template <Float T>
	void HierarchicalMesh<T>::get_element_descendents(const size_t elem_idx, std::vector<size_t> &descendents, const bool activeOnly) const {
		
		const Element_t &ELEM = _elements[elem_idx];

		//loop through the children
		for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
			const size_t child_idx = ELEM.children[c_idx];
			const Element_t &CHILD = _elements[child_idx];

			//add the relevent children
			if (!activeOnly or CHILD.is_active) {descendents.push_back(child_idx);}

			//recurse if needed
			if (CHILD.children.size()>0) {get_element_descendents(child_idx, descendents, activeOnly);}
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::join_descendents(const size_t elem_idx) {
		//get the active descendents of the element
		std::vector<size_t> descendents;
		get_element_descendents(elem_idx, descendents, true);

		//de-activate the descendents and any boundary faces
		for (size_t e_idx=0; e_idx<descendents.size(); e_idx++) {
			Element_t &ELEM = _elements[descendents[e_idx]];
			ELEM.is_active = false;
			_colorCount[ELEM.color] -= 1;
		}

		//activate the element
		_elements[elem_idx].is_active = true;

		//recolor the element
		recolor(elem_idx);

		//get the descendents of any boundary faces
		std::vector<size_t> boundary_faces;
		get_boundary_faces(elem_idx, boundary_faces);
		for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
			std::vector<size_t> descendents;
			get_boundary_face_descendents(boundary_faces[f_idx], descendents, true);

			//deactivate the descendent boundary elements
			for (size_t d_idx=0; d_idx<descendents.size(); d_idx++) {
				_boundary[descendents[d_idx]].is_active = false;
			}

			//activate the boundary faces of the coarse element
			_boundary[boundary_faces[f_idx]].is_active = true;

			//update the color of the boundary element
			_boundary[boundary_faces[f_idx]].color = _elements[elem_idx].color;
		}
	}


	template <Float T>
	void HierarchicalMesh<T>::split_element(const size_t elem_idx) {
		

		//any changes to _elements[elem_idx] are not protected by a unique mutex lock
		//calling split_element(k) for the same value of k in different threads will lead to undefined behavior
		//calling split_element(k) for different values of k in defferent threads is safe
		std::shared_lock<std::shared_mutex> read_lock_elements(_rw_mtx_elements);
		const size_t nElements = _elements.size();

		//if the specified does not exist, process any elements that need to be refined
		if (elem_idx >= nElements) {
			std::shared_lock<std::shared_mutex> read_lock_el_ref(_rw_mtx_el_ref_list);
			bool can_process_elements = !_element_refine_list.empty();
			read_lock_el_ref.unlock();
			if (can_process_elements) {process_refinement();}
		}
		

		//ensure that the element exists
		if (elem_idx >= nElements) {
			throw std::runtime_error("Index " + std::to_string(elem_idx) + " is out of bounds (_elements.size()= " + std::to_string(nElements) + ")");
			return;
		}


		//ensure that the element is active
		Element_t &ELEM = _elements[elem_idx];
		if (!ELEM.is_active) {return;}


		//if the element has been previously refined, activate and re-color its children
		if (!ELEM.children.empty()) {
			ELEM.is_active = false;
			std::unique_lock<std::shared_mutex> write_lock_color(_rw_mtx_color);
			_colorCount[ELEM.color] -= 1;

			//activate and color the children
			for (size_t e_idx: ELEM.children) {
				Element_t &CHILD = _elements[e_idx];
				CHILD.is_active  = true;
				recolor(CHILD.index);
			}

			//update any boundary faces
			std::vector<size_t> boundary_faces;
			get_boundary_faces(ELEM.index, boundary_faces);
			for (size_t i=0; i<boundary_faces.size(); i++) {
				Element_t &FACE = _boundary[boundary_faces[i]];
				FACE.is_active = false;

				assert(!FACE.children.empty());
				for (size_t f_idx : FACE.children) {
					Element_t &CHILDFACE = _boundary[f_idx];
					CHILDFACE.is_active  = true;
					CHILDFACE.color = _elements[CHILDFACE.parent].color; //the parent of a face is the element to which it is a face of (unique for boundary faces)
				}
			}
		}


		//the children of the element must be created. delay this until many elements can be processed in parallel.
		std::unique_lock<std::shared_mutex> write_lock_el_ref(_rw_mtx_el_ref_list);
		_element_refine_list.insert(elem_idx);
	}


	template <Float T>
	void HierarchicalMesh<T>::process_refinement() {
		//This method refines all elements in _element_refine_list by color batches
		//Each batch consists of all elements with a specific color
		//No two elements in the same color batch share a node
		//Each color batch is processed in parallel
		//There are some race conditions when using balanced coloring, so the optimal color may not be chosen for the children elements
		//The chosen color for the children elements is guarenteed to be valid and only new colors are generated when strictly necessary

		if (_element_refine_list.empty()) {return;}

		//partition the specified that are active by their color
		//if any element has been split already and the children are already stored in the mesh, activate them
		std::vector<std::vector<size_t>> colored_elements_to_split;
		{
			std::shared_lock<std::shared_mutex> read_lock_color(_rw_mtx_color);
			colored_elements_to_split.resize(_colorCount.size());
			for (size_t e_idx: _element_refine_list) {
				assert(e_idx<_elements.size());
				Element_t &ELEM = _elements[e_idx];
				if (!ELEM.is_active) {continue;}

				//element is to be split and the children must be computed
				assert(ELEM.index == e_idx);
				assert(ELEM.color<colored_elements_to_split.size());
				if (ELEM.children.size()==0) {colored_elements_to_split[ELEM.color].push_back(ELEM.index); continue;}
			}
		}

		
		for (size_t color=0; color<colored_elements_to_split.size(); color++) {
			std::vector<size_t> &this_color_elems = colored_elements_to_split[color];

			if (this_color_elems.size()==0) {continue;} //no elements of this color to split

			std::vector<size_t> child_element_index_start;
			std::vector<size_t> boundary_face_index_start;

			//the indices of where to store each child element must be known ahead of time
			//the local child j of element k will be stored at _elements[child_element_index_start[k] + j]
			const size_t nStartingElements = _elements.size();
			const size_t nStartingBoundary = _boundary.size();

			child_element_index_start.push_back(nStartingElements);
			boundary_face_index_start.push_back(nStartingBoundary);

			size_t nNewElements = vtk_n_children(_elements[this_color_elems[0]].vtkID);
			size_t maxNewNodes = vtk_n_nodes_when_split(_elements[this_color_elems[0]].vtkID);
			size_t nNewBoundaryFaces = 0;

			std::vector<size_t> boundary_faces;
			get_boundary_faces(this_color_elems[0], boundary_faces);
			for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
				nNewBoundaryFaces += vtk_n_children(_boundary[boundary_faces[f_idx]].vtkID);
			}


			//loop through elements and compute child offsets
			for (size_t i=1; i<this_color_elems.size(); i++) {
				const Element_t &CUR_ELEM  = _elements[this_color_elems[i]];
				const Element_t &PREV_ELEM = _elements[this_color_elems[i-1]];

				const size_t prev_children = vtk_n_children(PREV_ELEM.vtkID);
				const size_t prev_index_start = child_element_index_start[i-1];
				child_element_index_start.push_back(prev_index_start + prev_children);
				
				boundary_face_index_start.push_back(nNewBoundaryFaces);

				nNewElements += vtk_n_children(CUR_ELEM.vtkID);
				maxNewNodes  += vtk_n_nodes_when_split(CUR_ELEM.vtkID) - vtk_n_nodes(CUR_ELEM.vtkID);

				boundary_faces.clear();
				get_boundary_faces(this_color_elems[0], boundary_faces);
				for (size_t f_idx=0; f_idx<boundary_faces.size(); f_idx++) {
					nNewBoundaryFaces += vtk_n_children(_boundary[boundary_faces[f_idx]].vtkID);
				}
			}

			//reserve space and default-initialize the new elements
			_elements.resize(nStartingElements+nNewElements);
			_nodes.reserve(_nodes.size()+maxNewNodes);
			_boundary.reserve(nStartingBoundary+nNewBoundaryFaces);


			//decrement the _colorCount by the number of elements that will be deactivated
			_colorCount[color] -= this_color_elems.size();


			#pragma omp parallel
			{
				//get color count for each thread and initialize storage for any changes
				std::vector<size_t> t_color_count;
			    std::vector<size_t> t_color_delta;
			    
			    {
			        std::shared_lock<std::shared_mutex> read_lock_color(_rw_mtx_color);
			        t_color_count = _colorCount;
			        t_color_delta.resize(t_color_count.size());
			    }


				#pragma omp for
				for (size_t i=0; i<this_color_elems.size(); i++) {
					//create helper references and logical element
					Element_t &ELEM = _elements[this_color_elems[i]];
					assert(ELEM.index == this_color_elems[i]);
					VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(ELEM);

					//initialize storage for the values that will be needed to create the children elements
					std::vector<Point_t> child_vertex_coords;
					std::vector<size_t>  child_node_idx(vtk_n_nodes_when_split(ELEM.vtkID));
					
					//handle parent nodes/vertices that will be re-used
					size_t j;
					for (j=0; j<ELEM.nNodes; j++) {
						child_vertex_coords.push_back(_nodes[ELEM.nodes[j]].vertex);
						child_node_idx[j] = ELEM.nodes[j];
					}

					//get the verticices of the remaining nodes that the children will need
					vtk_elem->split(child_vertex_coords);

					//find any nodes that already exist and collect which nodes need to be added to _nodes
					std::vector<Node_t> nodes_to_create;
					std::vector<size_t> local_node;
					for (;j<child_vertex_coords.size(); j++) {
						Node_t NODE(child_vertex_coords[j]);
						size_t node_idx = _nodes.find(NODE);
						if (node_idx < _nodes.size()) {child_node_idx[j] = node_idx;}
						else {
							nodes_to_create.push_back(NODE);
							local_node.push_back(j); //track which nodes need to be corrected after NODE is inserted into _nodes and its index is known
						}
					}

					//create the nodes. insertion into _nodes must be serial, but the order of insertion does not matter
					for (size_t k=0; k<nodes_to_create.size(); k++) {
						size_t node_idx = (size_t) -1;
						[[maybe_unused]] int flag = _nodes.push_back(nodes_to_create[k], node_idx); //this has a mutex to make it thread-safe
						assert(flag==1); //the node must have been inserted. all elements are of the same color, so no other element will have added this node.
						assert(node_idx<_nodes.size());
						child_node_idx[local_node[k]] = node_idx; //now that the node has been created, track its index to create the children
					}

					//now all nodes have been created to create the children
					ELEM.is_active = false;

					for (size_t k=0; k<vtk_n_children(ELEM.vtkID); k++) {
						//get the indices of the nodes that define child k in the correct order
						std::vector<size_t> childNodes;
						vtk_elem->getChildNodes(childNodes, k, child_node_idx);

						//create the child
						const size_t global_child_index = child_element_index_start[i] + k;
						assert(global_child_index<_elements.size());
						_elements[global_child_index] = Element_t(childNodes, ELEM.vtkID);
						Element_t &CHILD = _elements[global_child_index];
						CHILD.index = global_child_index;
						CHILD.parent = ELEM.index;

						//update the nodes
						for (size_t n=0; n<childNodes.size(); n++) {
							_nodes[childNodes[n]].elems.push_back(global_child_index);
						}

						//update the parent element
						ELEM.children.push_back(global_child_index);

						//color the child
						CHILD.color = getValidElementColor(CHILD.index, t_color_count);
						if (CHILD.color >= t_color_count.size()) {
							//a color must be created
							t_color_count.push_back(1);
							t_color_delta.push_back(1);
						} else {
							t_color_count[CHILD.color] += 1;
							t_color_delta[CHILD.color] += 1;
						}
					}

					//clean up memory
					delete vtk_elem;

					//split any boundary faces
					std::vector<size_t> parent_boundary_faces;
					get_boundary_faces(ELEM.index, parent_boundary_faces);
					for (size_t f_idx=0; f_idx<parent_boundary_faces.size(); f_idx++) {
						#pragma omp critical //TODO: make this thread safe if needed.
						{
							assert(parent_boundary_faces[f_idx]<_boundary.size());
							//ensure that _boundary does not re-size and invalidate references
							while (_boundary.capacity() < _boundary.size() + vtk_n_children(_boundary[parent_boundary_faces[f_idx]].vtkID)) {_boundary.reserve(2*_boundary.capacity());}

							Element_t &FACE = _boundary[parent_boundary_faces[f_idx]];
							FACE.is_active = false;

							VTK_ELEMENT<Point_t>* vtk_face = _VTK_ELEMENT_FACTORY<Point_t>(FACE);
							//find the vertices for the split face
							std::vector<Point_t> child_vertex_coords;
							std::vector<size_t> child_node_idx(vtk_n_nodes_when_split(FACE.vtkID));
							size_t i;
							for (i=0; i<FACE.nNodes; i++) {
								child_vertex_coords.push_back(_nodes[FACE.nodes[i]].vertex);
								child_node_idx[i] = FACE.nodes[i];
							}
							vtk_face->split(child_vertex_coords);

							//get any new nodes
							for (;i<child_vertex_coords.size(); i++) {
								Node_t NODE(child_vertex_coords[i]);
								child_node_idx[i] = _nodes.find(NODE); //the node must have been generated by splitting the element
								assert(child_node_idx[i]<_nodes.size());
							}


							//create the children faces
							for (size_t j=0; j<vtk_n_children(FACE.vtkID); j++) {
								std::vector<size_t> childNodes;
								vtk_face->getChildNodes(childNodes, j, child_node_idx);
								Element_t CHILDFACE(childNodes, FACE.vtkID);
								CHILDFACE.parent = (size_t) -1;

								//determine the child element that this split face is a child of
								for (size_t c_idx=0; c_idx<ELEM.children.size(); c_idx++) {
									const Element_t &CHILD = _elements[ELEM.children[c_idx]];
									const VTK_ELEMENT<Point_t>* vtk_elem = _VTK_ELEMENT_FACTORY<Point_t>(CHILD);
									for (int k=0; k<vtk_n_faces(CHILD.vtkID); k++) {
										if (CHILDFACE == vtk_elem->getFace(k)) {
											CHILDFACE.parent = CHILD.index;
											CHILDFACE.color  = CHILD.color;
											break;
										}
									}
									delete vtk_elem;
								}

								//add the face to the _boundary
								assert(CHILDFACE.parent<_elements.size());
								CHILDFACE.index = _boundary.size();
								FACE.children.push_back(CHILDFACE.index);
								_boundary.push_back(std::move(CHILDFACE));
							}

							//clean up memory
							delete vtk_face;
						}
					}
				}

				#pragma omp critical
				{
					for (size_t c=0; c<t_color_delta.size(); c++) {
						if (c<_colorCount.size()) {
							_colorCount[c] += t_color_delta[c];
						} else {
							_colorCount.push_back(t_color_delta[c]);
						}
					}
				}
			}
		}



	}


