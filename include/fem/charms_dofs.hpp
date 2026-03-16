#pragma once

#include "fem/dofs.hpp"
#include "mesh/mesh_util.hpp"

#include <vector>
#include <array>
#include <unordered_set>

//This class introduces CHARMS DOF specializations that incorporate
//the ability create the child DOFs

namespace gv::fem
{
	template<typename DERIVED>
	struct CharmsDOF
	{
		bool active = false;
		size_t depth = (size_t) -1;

		template<gv::mesh::HierarchicalMeshType Mesh_t>
		inline auto make_children(const Mesh_t& mesh) const {
			return static_cast<const DERIVED*>(this)->make_children_impl(mesh);
		}

		inline double get_child_coef(size_t child_no) const {
			return static_cast<const DERIVED*>(this)->get_child_coef_impl(child_no);
		}
	};

	

	//Concept to check if a DOF is a CHARMS type.
	template<typename T>
	struct derives_from_charms_dof
	{
		template<typename D>
		static std::true_type test(const CharmsDOF<D>*);
		static std::false_type test(...);
		static constexpr bool value = decltype(test(std::declval<T*>()))::value;
	};
	
	template<typename T>
	concept IsCharmsDOF = derives_from_charms_dof<T>::value  && requires {
                       { T::nchildren } -> std::convertible_to<int>;
                   };

    template<IsCharmsDOF DOF_t>
	bool operator==(const DOF_t& left, const DOF_t& right) {
		if (left.depth != right.depth) {return false;}
		if (left.global_idx != right.global_idx) {return false;}
		return true;
	}


	struct CharmsVoxelQ1 : public CharmsDOF<CharmsVoxelQ1>, public VoxelQ1<CharmsVoxelQ1>
	{
		static constexpr int nchildren = 27;
		using VoxelQ1<CharmsVoxelQ1>::VoxelQ1; //use the VoxelQ1 constructors

		template<gv::mesh::HierarchicalMeshType Mesh_t>
		std::array<CharmsVoxelQ1, nchildren> make_children_impl(const Mesh_t& mesh) const {
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
			const auto& CENTER_V = mesh.getVertex(ELEM0.vertices[idx]);
			const typename Mesh_t::Point_t& center = CENTER_V.coord;

			//get second coordinate to compute the box dimension (clearly voxel specific)
			assert(ELEM0.depth == this->depth);
			const auto& OTHER_V = mesh.getVertex(ELEM0.vertices[idx==0 ? 1 : 0]);
			const auto h = 0.5*gutil::norminfty(center - OTHER_V.coord);
			const typename Mesh_t::Point_t H(1.25*h);
			typename Mesh_t::DomainBox_t box(center-H, center+H);

			//get the index of each vertex in the box
			std::unordered_set<size_t> vertex_idx_set;
			vertex_idx_set.reserve(27);
			for (size_t e_idx : this->support_idx) {
				if (e_idx == (size_t) -1) {continue;}
				const auto& ELEM = mesh.getElement(e_idx);
				assert(ELEM.depth == this->depth);

				for (size_t c_idx : ELEM.children) {
					const auto& CHILD_ELEM = mesh.getElement(c_idx);
					if (!CHILD_ELEM.is_active) {continue;}
					for (size_t v_idx : CHILD_ELEM.vertices) {
						const auto& VERTEX = mesh.getVertex(v_idx);
						if (box.contains(VERTEX.coord)) {
							vertex_idx_set.insert(v_idx);
						}
					}
				}
			}

			//convert to a vector
			const std::vector<size_t> vertex_idx(vertex_idx_set.begin(), vertex_idx_set.end());

			//make sure that we have the correct number of vertices
			#ifndef NDEBUG
				if (CENTER_V.onBoundary()) {assert(vertex_idx.size()<27);}
				else {assert(vertex_idx.size()==27);}
			#endif


			//make sure the children get created in the proper order
			std::array<int,nchildren> standard2vertex_idx;
			std::fill(standard2vertex_idx.begin(), standard2vertex_idx.end(), -1);
			for (int kk=-1; kk<2; ++kk) {
				for (int jj=-1; jj<2; ++jj) {
					for (int ii=-1; ii<2; ++ii) {
						const int standard_idx = (ii+1) + 3*(jj+1) + 9*(kk+1);
						const typename Mesh_t::Point_t step{ii,jj,kk};
						const typename Mesh_t::Point_t childCoord = center + h*step;
						for (size_t v_idx=0; v_idx<vertex_idx.size(); ++v_idx) {
							if (mesh.getVertex(vertex_idx[v_idx]).coord == childCoord) {
								standard2vertex_idx[standard_idx] = v_idx;
								break;
							}
						}
					}
				}
			}


			//create the new DOFs
			std::array<size_t,8> child_support_idx;
			std::array<size_t,8> child_local_idx;
			std::array<CharmsVoxelQ1,nchildren> result;
			for (int c_idx=0; c_idx<nchildren; ++c_idx) {
				if (standard2vertex_idx[c_idx] < 0) {continue;}
				const size_t v_idx = vertex_idx[standard2vertex_idx[c_idx]];
				const auto& VERTEX = mesh.getVertex(v_idx);
				child_support_idx.fill((size_t) -1);
				child_local_idx.fill((size_t) -1);

				int spt_cursor=0;
				for (size_t i=0; i<VERTEX.elems.size(); ++i) {
					//in the mesh, a single vertex may belong to many elements, especially in a hierarchical mesh
					//however, if the mesh consists of voxels with active elements non-overllapping,
					//it can belong to at most 8 active elements
					const size_t e_idx = VERTEX.elems[i];
					const auto& ELEM = mesh.getElement(e_idx);
					if (ELEM.depth != (this->depth + 1)) {continue;}
					child_support_idx[spt_cursor] = e_idx;

					//get the local index for this vertex in this support element
					for (size_t j=0; j<8; j++) {
						//voxels always have exactly 8 vertices
						if (ELEM.vertices[j] == v_idx) {
							child_local_idx[spt_cursor] = j;
							break;
						}
					}

					spt_cursor++;
				}
				
				//create the new DOF (inactive by default)
				result[c_idx] = CharmsVoxelQ1(v_idx, child_support_idx, child_local_idx);
				result[c_idx].depth = this->depth + 1;
			}


			return result;
		}



		double get_child_coef_impl(size_t child_no) const {
			// the child ording must be consistent with the ordering loop in make_children_impl
			switch (child_no) {
			
			//extreme corners
			case 0: case 2: case 6: case 8: case 18: case 20: case 24: case 26:
				return 0.125;

			//edge midpoints
			case 1: case 3: case 5: case 7: case 9: case 11: case 15: case 17: case 19: case 21: case 23: case 25:
				return 0.25;

			//face midpoints
			case 4: case 10: case 12: case 14: case 16: case 22:
				return 0.5;

			//center (original coordinate)
			case 13:
				return 1.0;

			default:
				assert(false);
			}
		}
	};
}