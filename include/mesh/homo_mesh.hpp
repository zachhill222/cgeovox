#pragma once

#include "util/point.hpp"
#include "util/point_octree.hpp"
#include "util/box.hpp"

#include <vector>
#include <stdexcept>

#include <sstream>
#include <iostream>
#include <fstream>

#include <omp.h>

namespace gv::mesh
{
	/// CLASS DEFINITION: mesh for a single element type
	template <typename Element_t>
	class HomoMesh
	{
	protected:
		//common typedefs
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3>;
		using PointList_t = gv::util::PointOctree<3,32>;

		PointList_t _nodes;
		std::vector<size_t> _elem2node;
		std::vector<size_t> _node2elem_start_idx; //each node belongs to an unknown number of elements. track where node2element begins for each node.
		std::vector<size_t> _node2elem;
		std::vector<std::vector<size_t>> _boundary; //assign which nodes belong to the boundary. allow for multiple boundaries.

	public:
		HomoMesh() : _nodes() {}
		HomoMesh(const Box_t& bbox) : _nodes(bbox) {}

		///type of element
		static const bool isHomogeneous = true;
		static const Element_t referenceElement;

		///array for storing element markers (e.g., use for marking properties or sub-domains).
		std::vector<int> elem_marker;

		///get a default element. usefull for external tools.
		Element_t new_element() const {Element_t element; return element;}

		///clear current mesh
		void clear();

		///set bounding box for the mesh
		void set_bbox(const Box_t& bbox) {clear(); _nodes.set_bbox(bbox);}

		///reserve space for _elem2node
		void reserve(size_t nNewElems)
		{
			assert(nNewElems>=nElems());
			if (nNewElems>_nodes.size()) {_nodes.reserve(nNewElems);} //approximation
			_elem2node.reserve(referenceElement.nNodes*nNewElems);
		}

		///copy node global indices for element "idx" into array "element"
		void get_element(const size_t idx, size_t (&element)[referenceElement.nNodes]) const;

		///add a single element via its node phycial locations. new nodes are added to _nodes if needed.
		void add_element(const gv::util::Point<3,double> (&element)[referenceElement.nNodes]);

		///add a single element via its global node indices in _nodes. each node must have previously been added to _nodes and index tracked externally.
		void add_element(const size_t (&element)[referenceElement.nNodes]);

		///add a single node
		void add_node(const gv::util::Point<3,double> &node)  {_nodes.push_back(node);}

		///get index for a node
		size_t node_idx(const gv::util::Point<3,double> &node) const  {return _nodes.find(node);}

		///get node location (return const reference)
		const gv::util::Point<3,double>& nodes(size_t const &idx) const  {return _nodes[idx];}

		///add node to a boundary group
		void add_to_boundary(const size_t node_idx, const size_t boundary_idx = 0)  {_boundary[boundary_idx].push_back(node_idx);}

		///create a new boundary group
		size_t create_new_boundary()  {_boundary.push_back(std::vector<size_t> {}); return _boundary.size()-1;}

		///get boundary group
		const std::vector<size_t>& boundary(size_t idx=0) const  {return _boundary[idx];}

		///get number of elements
		size_t nElems() const  {return _elem2node.size()/referenceElement.nNodes;}

		///get number of nodes
		size_t nNodes() const  {return _nodes.size();}

		///access elem2node (handles conversion from double to single index)
		size_t elem2node(size_t elem, size_t localnode) const  {return _elem2node[elem*referenceElement.nNodes+localnode];}
		
		///get number of elements that contain a specified node
		size_t local_nElem(size_t node) const;

		///get global element index for the n-th ("localelem") local element that contains the specified node. use local_nElem to get the total number of local elements first.
		size_t node2elem(size_t node, size_t localelem) const;

		///compute mesh connectivity (node2elem)
		void compute_node2elem();

		///mesh sub-domain and put into out_mesh
		void mesh_subdomain(const int mkr, HomoMesh<Element_t> &out_mesh) const;

		///print mesh structure to ostream
		void vtkprint(std::ostream &stream) const;

		///print mesh connectivity for error checking
		void print_connectivity() const;

		///save mesh to file
		void save_as(std::string filename) const;

		///append data to mesh file
		template<typename Array_t>
		void _append_node_scalar_data(const std::string filename, const Array_t &data, const std::string data_description, const bool make_header=true) const;

		///append data to mesh file
		template<typename Array_t>
		void _append_element_scalar_data(const std::string filename, const Array_t &data, const std::string data_description, const bool make_header=true) const;
	};


/// CLASS METHOD IMPLEMENTATION ///

///clear current mesh
template <typename Element_t>
void HomoMesh<Element_t>::clear()
{
	_nodes.clear();
	_elem2node.clear();
	_node2elem_start_idx.clear();
	_node2elem.clear();
	_boundary.clear();
	elem_marker.clear();
}


///copy node global indices for element "idx" into array "element"
template <typename Element_t>
void HomoMesh<Element_t>::get_element(const size_t idx, size_t (&element)[referenceElement.nNodes]) const
{
	size_t start = idx*referenceElement.nNodes;
	for (size_t i=0; i<referenceElement.nNodes; i++) {element[i] = _elem2node[start+i];}
}


///add a single element via its node phycial locations. new nodes are added to _nodes if needed.
template <typename Element_t>
void HomoMesh<Element_t>::add_element(const gv::util::Point<3,double> (&element)[referenceElement.nNodes])
{
	for (size_t i=0; i<referenceElement.nNodes; i++)
	{
		size_t global_idx;
		int flag = _nodes.push_back(element[i], global_idx);
		assert(flag!=-1);
		_elem2node.push_back(global_idx);
	}
}

///add a single element via its global node indices in _nodes. each node must have previously been added to _nodes and index tracked externally.
template <typename Element_t>
void HomoMesh<Element_t>::add_element(const size_t (&element)[referenceElement.nNodes])
{
	for (size_t i=0; i<referenceElement.nNodes; i++)
	{
		_elem2node.push_back(element[i]);
	}
}


///get number of elements that contain a specified node
template <typename Element_t>
size_t HomoMesh<Element_t>::local_nElem(size_t node) const
{
	if (node==nNodes()-1) {return _node2elem.size()-_node2elem_start_idx[node];}
	return _node2elem_start_idx[node+1]-_node2elem_start_idx[node];
}


///get global element index for the n-th ("localelem") local element that contains the specified node. use local_nElem to get the total number of local elements first.
template <typename Element_t>
size_t HomoMesh<Element_t>::node2elem(size_t node, size_t localelem) const
{
	//return max size_t as an error
	if (localelem >= local_nElem(node)) {return (size_t) -1;}
	return _node2elem[_node2elem_start_idx[node] + localelem];
}


///compute mesh connectivity (node2elem)
template <typename Element_t>
void HomoMesh<Element_t>::compute_node2elem()
{
	//compute number of elements for each node
	std::vector<size_t> node_count(nNodes(), 0);
	for (size_t el=0; el<nElems(); el++)
	{
		for (size_t i=0; i<referenceElement.nNodes; i++)
		{
			size_t node = elem2node(el,i);
			node_count[node] += 1;
		}
	}

	//convert number of elements to start index
	_node2elem_start_idx.resize(nNodes(),0);
	_node2elem_start_idx[0]=0;

	for (size_t n=1; n<nNodes(); n++)
	{
		_node2elem_start_idx[n] = _node2elem_start_idx[n-1] + node_count[n-1];
	}

	//compute node2element
	_node2elem.resize(_node2elem_start_idx[nNodes()-1] + node_count[nNodes()-1]);
	std::fill(node_count.begin(), node_count.end(), 0); //use node_count for tracking current index for each node
	for (size_t el=0; el<nElems(); el++)
	{
		for (size_t i=0; i<referenceElement.nNodes; i++)
		{
			size_t node = elem2node(el,i);
			size_t idx = _node2elem_start_idx[node] + node_count[node];
			_node2elem[idx] = el;
			node_count[node] += 1;
		}
	}
}


///mesh sub-domain and put into out_mesh
template <typename Element_t>
void HomoMesh<Element_t>::mesh_subdomain(const int mkr, HomoMesh<Element_t> &out_mesh) const
{
	//TODO: make external and internal boundaries?

	//check that all elements are marked
	if (elem_marker.size() != nElems())
	{
		throw std::runtime_error("Not all elements of the original mesh are marked.");
	}

	//count number of elements
	size_t n_elems_out = 0;
	for (size_t el=0; el<nElems(); el++)
	{
		if (elem_marker[el]==mkr) {n_elems_out+=1;}
	}

	if (n_elems_out==0) {return;}

	//prepare out_mesh
	out_mesh.clear();
	out_mesh.reserve(n_elems_out);
	out_mesh.set_bbox(this->_nodes.bbox());

	//add elements
	for (size_t el=0; el<nElems(); el++)
	{
		if (elem_marker[el] == mkr)
		{
			//construct new element
			gv::util::Point<3,double> new_elem[referenceElement.nNodes];
			for (size_t i=0; i<referenceElement.nNodes; i++)
			{
				new_elem[i] = this->_nodes[this->elem2node(el,i)];
			}

			//add new element to out_mesh
			out_mesh.add_element(new_elem);
			out_mesh.elem_marker.push_back(mkr);
		}
	}
}


///print mesh connectivity for error checking
template <typename Element_t>
void HomoMesh<Element_t>::print_connectivity() const
{
	std::cout << "ELEMENT2NODE:\n";
	for (size_t el=0; el<nElems(); el++)
	{
		std::cout << "element " << el << "\t: ";
		for (size_t n=0; n<referenceElement.nNodes; n++)
		{
			std::cout << elem2node(el,n) << " ";
		}
		std::cout << std::endl;
	}


	std::cout << "\nNODE2ELEMENT:\n";
	for (size_t n=0; n<nNodes(); n++)
	{
		std::cout << "node " << n << "\t: ";
		for (size_t el=0; el<local_nElem(n); el++)
		{
			std::cout << node2elem(n,el) << " ";
		}
		std::cout << std::endl;
	}
}


///print mesh structure in vtk format (any ostream)
template <typename Element_t>
void HomoMesh<Element_t>::vtkprint(std::ostream &stream) const
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
	for (size_t i=0; i<nNodes(); i++) { buffer << _nodes[i] << "\n";}
	buffer << "\n";
	stream << buffer.rdbuf();
	buffer.str("");

	//ELEMENTS
	buffer << "CELLS " << nElems() << " " << (1+referenceElement.nNodes)*nElems() << "\n";
	for (size_t i=0; i<nElems(); i++)
	{
		buffer << referenceElement.nNodes << " ";
		for (size_t j=0; j<referenceElement.nNodes; j++)
		{
			buffer << _elem2node[referenceElement.nNodes*i + j] << " ";
		}
		buffer << "\n";
	}
	buffer << "\n";
	stream << buffer.rdbuf();
	buffer.str("");

	//VTK IDs
	buffer << "CELL_TYPES " << nElems() << "\n";
	for (size_t i=0; i<nElems(); i++) {buffer << referenceElement.vtkID << " ";}
	buffer << "\n\n";
	stream << buffer.rdbuf();
	buffer.str("");

	// //ELEMENT MARKERS
	// if (elem_marker.size() == nElems())
	// {
	// 	buffer << "CELL_DATA " << nElems() << "\n";
	// 	buffer << "SCALARS elem_marker integer\n";
	// 	buffer << "LOOKUP_TABLE default\n";
	// 	for (size_t el=0; el<nElems(); el++) {buffer << elem_marker[el] << " ";}
	// 	buffer << "\n\n";
	// }
	// stream << buffer.rdbuf();
	// buffer.str("");
}


//save mesh to file (vtk format)
template <typename Element_t>
void HomoMesh<Element_t>::save_as(std::string filename) const
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


//append data to mesh. set make_header to false if previous data of the same type (e.g. scalar node data) has been previously added to the file.
template <typename Element_t>
template <typename Array_t>
void HomoMesh<Element_t>::_append_node_scalar_data(const std::string filename, const Array_t &data, const std::string data_description, const bool make_header) const
{
	//////////////// OPEN FILE ////////////////
	std::ofstream meshfile(filename, std::ios::app);

	if (not meshfile.is_open()){
		std::cout << "Couldn't write to " << filename << std::endl;
		meshfile.close();
		return;
	}

	//append data
	std::stringstream buffer;

	if (make_header)
	{
		buffer << "POINT_DATA " << nNodes() << "\n";
	}
	
	buffer << "SCALARS " << data_description << " float\n";
	buffer << "LOOKUP_TABLE default\n";
	for (size_t i=0; i<nNodes(); i++) {buffer << data[i] << " ";}
	buffer << "\n\n";
	meshfile << buffer.rdbuf();
	buffer.str("");

	//////////////// CLOSE FILE /////////////
	meshfile.close();
}


//append data to mesh. set make_header to false if previous data of the same type (e.g. scalar element data) has been previously added to the file.
template <typename Element_t>
template <typename Array_t>
void HomoMesh<Element_t>::_append_element_scalar_data(const std::string filename, const Array_t &data, const std::string data_description, const bool make_header) const
{
	//////////////// OPEN FILE ////////////////
	std::ofstream meshfile(filename, std::ios::app);

	if (not meshfile.is_open()){
		std::cout << "Couldn't write to " << filename << std::endl;
		meshfile.close();
		return;
	}

	//append data
	std::stringstream buffer;

	if (make_header)
	{
		buffer << "CELL_DATA " << nElems() << "\n";
	}
	
	buffer << "SCALARS " << data_description << " float\n";
	buffer << "LOOKUP_TABLE default\n";
	for (size_t i=0; i<nElems(); i++) {buffer << data[i] << " ";}
	buffer << "\n\n";
	meshfile << buffer.rdbuf();
	buffer.str("");

	//////////////// CLOSE FILE /////////////
	meshfile.close();
}




}

