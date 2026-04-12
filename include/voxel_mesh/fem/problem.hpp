#pragma once

#include "voxel_mesh/fem/dofhandler.hpp"
#include "voxel_mesh/mesh/voxel_mesh.hpp"

#include <tuple>

namespace gv::fem
{
	//struct for passing multiple kernels
	template<typename... Kernel_ts>
	struct KernelContainer
	{
		std::tuple<Kernel_ts...> kernels;
		KernelContainer(Kernel_ts&&... ks) : kernels(std::forward<Kernel_ts>(ks)...) {}

		template<int I>
		auto& get() {return std::get<T>(kernels);}

		template<int I>
		const auto& get() {return std::get<T>(kernels);}

		static constexpr int size() {return sizeof...(Kernel_ts);}
	};

	//struct for passing multiple dof types.
	//pass one for each variable
	template<VoxelMeshType Mesh_t, typename... DOF_ts>
	struct DofHandlerContainer
	{
		std::tuple<DofHandler<Mesh_t, DOF_ts>...> handlers;

		DofHandlerContainer(const Mesh_t& mesh) :
			handlers(DofHandler<Mesh_t,DOF_ts>(mesh)...) {}

		template<int I>
		auto& get() {return std::get<I>(handlers);}

		template<int I>
		const auto& get() {return std::get<T>(handlers);}
	};

	//forward declare to help with parameter unpacking
	template<typename Mesh_type,
			 typename DofContainer_type,
			 typename KernelContainer_type>
	class BaseProblem;
}