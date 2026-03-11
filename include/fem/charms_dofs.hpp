#pragma once

#include "fem/dofs.hpp"
#include "mesh/mesh_util.hpp"

#include <vector>

//This class introduces CHARMS DOF specializations that incorporate
//the ability create the child DOFs

namespace gv::fem
{
	template<typename DERRIVED>
	struct CharmsDOF
	{
		bool active = true;

		template<gv::mesh::HierarchicalMeshType Mesh_t>
		inline std::vector<DERRIVED> make_children(const Mesh_t& mesh) const {
			return static_cast<const DERRIVED*>(this)->make_children_impl(mesh);
		}
	};



	struct CharmsVoxelQ1 : public CharmsDOF<CharmsVoxelQ1>, public VoxelQ1
	{
		using VoxelQ1::VoxelQ1; //use the VoxelQ1 constructors

		template<gv::mesh::HierarchicalMeshType Mesh_t>
		std::vector<CharmsVoxelQ1> make_children_impl(const Mesh_t& mesh) const {
			//
			//      O --- o --- O --- o --- O
			//		|     |     |     |     |
			//		o --- x --- x --- x --- o
			//      |     |     |     |     |
			//      O --- x --- X --- x --- O
			//      |     |     |     |     |
			//      o --- x --- x --- x --- o
			//      |     |     |     |     |
			//      O --- o --- O --- o --- O
			//
			// O: old vertices
			// o: new vertices
			// X: old dof (+new dof at same vertex with smaller support)
			// x: new dofs (these are the only ones that get activated if the refinement is truly hierarchical)
			//

			std::vector<CharmsVoxelQ1> result;
			result.reserve(27);

			//Construct support of parent dof as a single box
			std::vector<typename Mesh_t::Vertex_t> vertex_list;
			for (size_t )

			return result;
		}
	};
}