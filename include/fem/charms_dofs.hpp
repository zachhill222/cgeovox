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
		bool active = false;

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
			/////////////////////////////////////////////////////////////////////////////////////////////////////////
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
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

			
			//Construct box that contains the vertices of all new dof locations
			//and does not contain any locations of vertices not getting new dofs
			//if we are interior to the domain, there should be exactly 27 such vertices
			//if we are on the boundary, there are at most 27 such vertices

			//get center coordinate
			const auto& ELEM0 = mesh.getElement(this->support_idx[0]);
			const size_t idx = this->local_idx[0];
			const auto& CENTER_V = mesh.getVertex(ELEM.vertices[idx]);
			const typename Mesh_t::DomainPoint_t& center = CENTER_V.coord;

			//get second coordinate to compute the box dimension
			const auto& OTHER_V = mesh.getVertex(ELEM.vertices[idx==0 ? 1 : 0]);
			const auto h = 0.75*gutil::norminfty(center - OTHER_V.coord);
			const typename Mesh_t::DomainPoint_t H(h);
			typename Mesh_t::DomainBox_t box(center-H, center+H);

			//get the index of each vertex in the box
			std::set<size_t> vertex_idx;
			vertex_idx.reserve(27);
			for (size_t e_idx : this->support_idx) {
				const auto& ELEM = mesh.getElement(e_idx);
				for (size_t c_idx : ELEM.children) {
					const auto& CHILD_ELEM = mesh.getElement(c_idx);
					if (!CHILD_ELEM.is_active) {continue;}
					for (size_t v_idx : CHILD_ELEM.vertices) {
						const auto& VERTEX = mesh.getVertex(v_idx);
						if (box.contains(VERTEX.coord)) {
							vertex_idx.insert(v_idx);
						}
					}
				}
			}

			//make sure that we have the correct number of vertices
			#ifndef NDEBUG
				if (CENTER_V.onBoundary()) {assert(vertex_idx.size()<27);}
				else {assert(vertex_idx.size()==27);}
			#endif


			//create the new DOFs
			std::vector<CharmsVoxelQ1> result;
			std::array<size_t,8> child_support_idx;
			std::array<size_t,8> child_local_idx;
			result.reserve(vertex_idx.size());
			for (size_t v_idx : vertex_idx) {
				const auto& VERTEX = mesh.getVertex(v_idx);
				std::fill(child_support_idx.begin(), child_support_idx.end(), (size_t) -1);
				std::fill(child_local_idx.begin(), child_local_idx.end(), (size_t) -1);

				for (size_t i=0; i<VERTEX.elements.size(); ++i) {
					//in the mesh, a single vertex may belong to many elements, especially in a hierarchical mesh
					//however, if the mesh consists of voxels with active elements non-overllapping,
					//it can belong to at most 8 active elements
					const size_t e_idx = VERTEX.elements[i];
					const auto& ELEM = mesh.getElement(e_idx);
					if (!ELEM.is_active) {continue;}
					child_support_idx[i] = e_idx;

					//get the local index for this vertex in this support element
					for (size_t j=0; j<8; j++) {
						//voxels always have exactly 8 vertices
						if (ELEM.vertices[j] == v_idx) {
							child_local_idx[i] = j;
						}
					}
				}
			
				//create the new DOF (inactive by default)
				result.emplace_back(v_idx, child_support_idx, child_local_idx);
			}


			return result;
		}
	};
}