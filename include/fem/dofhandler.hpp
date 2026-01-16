#pragma once

#include "mesh/mesh_basic.hpp"
#include "mesh/vtk_defs.hpp"

#include <string>
#include <vector>
#include <memory>
#include <cassert>

namespace gv::fem
{
	//////////////////////////////////////////////////
	/// Struct to track each element that a DOF is associated with
	/// and the local index of the DOF in that element.
	//////////////////////////////////////////////////
	struct ElemLocalDOF
	{
		size_t elem_idx = (size_t) -1;
		int local_idx   = -1;
	};


	struct ElemType
	{
		const bool iso; //e.g. iso=true and degree=1 for Q1-iso-Q2 elements
		const int degree;
	};

	struct InfoElem
	{
		ElemType type {false, 1} ; //(default is Q1)
		std::vector<size_t> dofs{}; //link to _dofs container

		InfoElem(bool iso, int deg) : type({iso, deg}), dofs{} {}
	};


	//////////////////////////////////////////////////
	/// Struct to hold all information for a given DOF.
	//////////////////////////////////////////////////
	enum class FeatureType
	{
		VERTEX, //always 0D
		FACE, //1D or 2D
		ELEMENT, //2D or 3D
		NONE
	};

	struct InfoDOF
	{
		FeatureType feature {FeatureType::NONE};
		size_t feature_index = (size_t) -1;
		std::vector<ElemLocalDOF> elements{}; //link to _elem2dof container
		bool is_active = false;
	};


	//////////////////////////////////////////////////
	/// Interface for handling all DOFs for a given problem.
	/// This interface assumes that all elements are using the same dof types.
	/// 
	//////////////////////////////////////////////////
	template<gv::mesh::BasicMeshType Mesh_t, int deg=1>
	class HandlerDOF
	{
	protected:
		std::shared_ptr<const Mesh_t> _mesh = nullptr;
		std::vector<InfoDOF> _dofs;
		std::vector<InfoElem> _elem;
		
		std::vector<size_t> _handlerElem2meshElem; //element indices always need to be tracked
	public:
		using Element_t  = typename Mesh_t::Element_t;
		using Vertex_t   = typename Mesh_t::Vertex_t;
		using Point_t    = typename Mesh_t::Point_t;
		using RefPoint_t = typename Mesh_t::RefPoint_t;
	
		/// Access methods
		inline const InfoDOF& get_dof(const size_t idx) const {assert(idx<_dofs.size()); return _dofs[idx];}
		inline InfoDOF& get_dof(const size_t idx)  {assert(idx<_dofs.size()); return _dofs[idx];}
		inline const InfoElem& get_elem(const size_t idx) const {assert(idx<_elem.size()); return _elem[idx];}
		inline InfoElem& get_elem(const size_t idx) {assert(idx<_elem.size()); return _elem[idx];}
		std::vector<ElemLocalDOF> get_dof_support(const size_t idx) const; //for evaluating, getting the local dof number is necessary

		/// Utility methods
		void link_mesh(const Mesh_t& mesh) {_mesh = std::make_shared<Mesh_t>(&mesh);}

		/// Construct lagrange dofs. This requires that
		/// the mesh is in a conformal state.
		void distribute_lagrange_dofs();
	};


	template<gv::mesh::BasicMeshType Mesh_t>
	std::vector<ElemLocalDOF> HandlerDOF<Mesh_t>::get_dof_support(const size_t idx) const
	{
		assert(idx<_dofs.size());
		assert(_elem2dof.size() == _mesh.get()->nElements());
		
		std::vector<size_t> result;
		for (const ElemLocalDOF& el : _dofs[idx].elements)
		{
			assert(el.elem_idx < _elem2dof.size());
			assert(el.local_idx >= 0);
			result.push_back(el);
		}

		result.shrink_to_fit();
		return result;
	}


	////////////////////////////////////////
	/// Distributing lagrange dofs based on the element type. (Q1 or Q2)
	////////////////////////////////////////
	template<gv::mesh::BasicMeshType Mesh_t>
	void HandlerDOF<Mesh_t>::distribute_lagrange_dofs()
	{
		_dofs.clear();
		_dofs.resize(_mesh->nVertices());
		size_t current_dof_idx = 0;

		_elem.clear();
		_elem.reserve(_mesh->nElements());

		_handlerElem2meshElem.clear();
		_handlerElem2meshElem.reserve(_mesh->nElements());

		for (const auto el = _mesh.get()->begin(); el!=_mesh.get()->end(); ++el) {
			const Element_t& ELEM = *el;

			//dof and mesh element indices can be missmatched
			//if the mesh is hierarchical
			_handlerElem2meshElem.emplace_back(ELEM.index);
			if (ELEM.vtkID == LINE_VTK_ID or
				ELEM.vtkID == PIXEL_VTK_ID or 
				ELEM.vtkID == QUAD_VTK_ID or 
				ELEM.vtkID == VOXEL_VTK_ID or 
				ELEM.vtkID == HEXAHEDRON_VTK_ID) {
				_elem.emplace_back(false, 1)
			}
			else if (ELEM.vtkID == QUADRATIC_EDGE_VTK_ID
					 ELEM.vtkID == BIQUADRATIC_QUAD_VTK_ID or
					 ELEM.vtkID == TRIQUADRATIC_HEXAHEDRON_VTK_ID) {
				_elem.emplace_back(false, 2)
			}
			else {
				throw std::runtime_error("HandlerDOF::distribute_lagrange_dofs - unknown element type with vtkID= " + std::to_string(ELEM.vtkID));
			}

			//make dofs and link to elements
			const size_t n_dofs = gv::mesh::vtk_n_vertices(ELEM.vtkID);
			_elem.back()->dofs.reserve(n_dofs);
			for (size_t i=0; i<n_dofs; ++i) {
				_elem.back()->dofs.push_back
			}
		}
	}

}


