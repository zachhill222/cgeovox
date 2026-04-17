#pragma once

#include "voxel_mesh/fem/dofhandler.hpp"
#include "voxel_mesh/fem/kernel.hpp"
#include "voxel_mesh/mesh/voxel_mesh.hpp"


namespace GV
{
	
	//A base problem class to handle constructing matrices from bilinear forms
	//and coordinating mesh refinement with several dofhandlers.
	template<typename VoxelMeshType Mesh_type, typename Kernel_type, typename DofHandler_ts>
	class BaseProblem
	{
	public:
		using Mesh_t = Mesh_type;
		using Elem_t = typename Mesh_t::VoxelElement;
		using Kernel_t = Kernel_type;

		static constexpr uint64_t MAX_DEPTH = Mesh_t::MAX_DEPTH;

		//the problem class is responsible for coordinating changes with the mesh,
		//but it might be useful to do pre-processing to the mesh before passing it 
		//to the problem class
		Mesh_t& 	mesh;
		Kernel_t 	kernel;

		
		
	private:
		std::tuple<DofHandler_ts&...> dof_handlers;
	};


}