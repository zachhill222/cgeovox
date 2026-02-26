#pragma once

#include "fem/dofs.hpp"
#include "mesh/mesh_basic.hpp"
#include "mesh/mesh_util.hpp"
#include "mesh/vtk_defs.hpp"

#include "gutil.hpp"

#include <type_traits>
#include <vector>
#include <array>
#include <string>
#include <cassert>

#include <omp.h>

namespace gv::fem
{
	//DOF handler for a single DOF type
	template<gv::mesh::BasicMeshType Mesh_t, typename DOF_t, typename Coef_t = double>
	class UniformDOFhandler
	{
	private:
		void distribute_nodal();

	protected:
		const Mesh_t& mesh;
		std::vector<DOF_t>  dofs;
		std::vector<Coef_t> coefs; //best tracked in a problem and only write here for fileio
		std::vector<size_t> boundary_dofs;

	public:
		inline size_t n_dofs() const {return dofs.size();}
		inline const DOF_t& dof(const size_t idx) const {assert(idx<dofs.size()); return dofs[idx];}
		inline const Coef_t& coef(const size_t idx) const {assert(idx<coefs.size()); return coefs[idx];}
		inline Coef_t& coef(const size_t idx) {assert(idx<coefs.size()); return coefs[idx];}
		void reserve(const size_t idx) {dofs.reserve(idx); coefs.reserve(idx);}
		void clear() {dofs.clear(); coefs.clear(); boundary_dofs.clear();}

		//construct handler and link to the mesh
		UniformDOFhandler(const Mesh_t& mesh) : mesh(mesh) {}

		//create the dofs for the current mesh
		void distribute();

		//interpolate the coefs to evaluate the field at the given point
		Coef_t interpolate(const )

		//save the mesh and nodal evaluations to a file
		void save_as(const std::string filename, const bool use_ascii=false) const;
	};


	///dispatch the distribute to the correct type
	template<gv::mesh::BasicMeshType Mesh_t, typename DOF_t, typename Coef_t>
	void UniformDOFhandler<Mesh_t,DOF_t,Coef_t>::distribute()
	{
		clear();

		if constexpr (std::is_same_v<DOF_t,VoxelQ1>) {distribute_nodal();}
		else if constexpr (std::is_same_v<DOF_t,HexQ1>) {distribute_nodal();}
		else {throw std::logic_error("this DOF_t is not supported yet");}
	}


	///distribute lagrange nodal dofs (it is assumed that the mesh is in a conformal state)
	template<gv::mesh::BasicMeshType Mesh_t, typename DOF_t, typename Coef_t>
	void UniformDOFhandler<Mesh_t,DOF_t,Coef_t>::distribute_nodal()
	{
		reserve(mesh.nVertices());
		dofs.resize(mesh.nVertices());

		#pragma omp parallel for
		for (size_t n=0; n<mesh.nVertices(); ++n) {
			const auto& VERTEX = mesh.getVertex(n);
			std::array<size_t,8> support;
			std::array<size_t,8> local_idx;

			int i;
			for (i=0; i<VERTEX.elems.size(); ++i) {
				support[i] = VERTEX.elems[i];
				const auto& ELEM = mesh.getElement(support[i]);

				//ensure that the mesh and DOF are compatible
				if constexpr (std::is_same_v<DOF_t,VoxelQ1>) {
					if (ELEM.vtkID != VOXEL_VTK_ID) {
						throw std::runtime_error("attempting to distribute VoxelQ1 DOF to a non-voxel element");
					}
				}
				else if constexpr (std::is_same_v<DOF_t,HexQ1>) {
					if (ELEM.vtkID != HEXAHEDRON_VTK_ID) {
						throw std::runtime_error("attempting to distribute HexQ1 DOF to a non-hex element");
					}
				}

				//get local index within the support element
				for (size_t m=0; m<ELEM.vertices.size(); ++m) {
					if (ELEM.vertices[m]==n) {
						local_idx[i] = m;
						break;
					}
				}
				assert(ELEM.vertices[local_idx[i]] == n);
			}

			//mark unused support elements if the node was on the boundary
			for (;i<DOF_t::max_support; ++i) {
				support[i] = (size_t) -1;
				local_idx[i] = (size_t) -1;
			}

			//create the DOF
			dofs[n] = DOF_t(n, support, local_idx);

			//mark DOF as part of the boundary
			if (VERTEX.boundary_faces.size()>0) {
				#pragma omp critical
				{
					boundary_dofs.push_back(n);
				}
			}
		}
	}
}


