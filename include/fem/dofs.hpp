#pragma once

#include "gutil.hpp"
#include "mesh/mesh_util.hpp"
#include "mesh/vtk_elements.hpp"

#include <array>
#include <concepts>
#include <type_traits>

#include <omp.h>

namespace gv::fem
{
	//Base class for all FEM DOFs
	//Pass a reference to the derived class (e.g. Q1 basis functions on a voxel) to avoid
	//vtables (Curiously Recurring Template Pattern)
	template<int FEATURE_DIM, int REF_DIM, int MAX_SUPPORT, typename DERIVED>
		requires ( (0<=FEATURE_DIM and FEATURE_DIM<=3) and (REF_DIM==2 or REF_DIM==3) )
	struct DOF
	{
		using RefPoint_t = gutil::Point<REF_DIM,double>;
		//index of the mesh feature that this DOF "lives" in
		//this could be an index into the vertex, element, face, or edge arrays.
		//which array is to be index is determined by the feature_dim and ref_dim.
		//if feature_dim=0, then this is an index into the vertex array
		//if feature_dim=1, then this is an index into the edge array and so on.
		//when ref_dim=2, the edge array and face_array should be the same.
		size_t global_idx;

		//indices of support elements to be used in evaluation
		//in CHARMS, these elements must be on the same refinement level as the basis function
		std::array<size_t, MAX_SUPPORT> support_idx;

		//the local index of this basis function in each of the support elements.
		//this is necessary for the evaluation methods.
		std::array<size_t, MAX_SUPPORT> local_idx;

		//track dimensions of the reference element and the mesh feature.
		//e.g., for piecewise polynomial lagrange basis functions on triangle, ref_dim=2 and feature_dim=0
		static constexpr int ref_dim     = REF_DIM;
		static constexpr int feature_dim = FEATURE_DIM;

		//track the maximum number of support elements
		static constexpr int max_support = MAX_SUPPORT;

		//evaluation and gradient of the basis function in the reference coordinates
		//the gradient must have the jacobian of the element map applied to it before it is used,
		//but that should be handled by the dofhandler.
		//eval_impl and grad_impl must be defined in the derived class
		//spt is the index of the support element that point is in
		inline double eval(const RefPoint_t& point, const int spt) const {return static_cast<const DERIVED*>(this)->eval_impl(point, spt);}
		inline RefPoint_t grad(const RefPoint_t& point, const int spt) const {return static_cast<const DERIVED*>(this)->grad_impl(point, spt);}

		//evaluate the DOF at the specified point in the mesh
		//this will be slower than eval() because of several lookups and the transformation to the reference coordinates
		template<gv::mesh::BasicMeshType Mesh_t>
		double eval_at(const typename Mesh_t::Point_t& point, const Mesh_t& mesh) const
		{
			//determine if the point is inside the support
			//the ordering of the support elements is not assumed to be known
			std::vector<typename Mesh_t::Point_t> elem_vertices;
			for (int spt=0; spt<MAX_SUPPORT; ++spt) {
				const size_t e_idx = this->support_idx[spt];
				if (e_idx == (size_t) -1) {continue;}
				const auto& ELEM = mesh.getElement(e_idx);
				
				//collect vertex coordinates and construct element
				auto vtk_elem = gv::mesh::VTK_ELEMENT_POLY<Mesh_t>(ELEM.vtkID);
				vtk_elem.set_element(mesh, ELEM.index);

				//check containment
				if (vtk_elem.contains(point)) {
					//it is contained, 
					const RefPoint_t ref_coord = vtk_elem.geo2ref(point);

					return eval(ref_coord, spt);
				}
			}

			return 0.0;
		}

		//evaluate the DOF at the specified point in the mesh
		//this will be slower than eval() because of several lookups and the transformation to the reference coordinates
		template<gv::mesh::BasicMeshType Mesh_t>
		typename Mesh_t::Point_t eval_grad_at(const typename Mesh_t::Point_t& point, const Mesh_t& mesh) const
		{
			//determine if the point is inside the support
			//the ordering of the support elements is not assumed to be known
			std::vector<typename Mesh_t::Point_t> elem_vertices;
			for (int spt=0; spt<MAX_SUPPORT; ++spt) {
				const size_t e_idx = this->support_idx[spt];
				if (e_idx == (size_t) -1) {continue;}
				const auto& ELEM = mesh.getElement(e_idx);
				
				//collect vertex coordinates and construct element
				auto vtk_elem = gv::mesh::VTK_ELEMENT_POLY<Mesh_t>(ELEM.vtkID);
				vtk_elem.set_element(mesh, ELEM.index);

				//check containment
				if (vtk_elem.contains(point)) {
					//it is contained, 
					const RefPoint_t ref_coord = vtk_elem.geo2ref(point);
					auto jac = vtk_elem.jacobian(ref_coord);
					return static_cast<typename Mesh_t::Point_t>(jac.tr()/grad(ref_coord, spt));
				}
			}
			return typename Mesh_t::Point_t(0.0);
		}


		//when evaluating at all of the quadrature points, it will be nice to do so all at once.
		//use a result parameter to help avoid excessive allocations, but a return by value is provided for debug and convenience
		template<int N_PTS>
		void batch_eval(std::array<double, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			#pragma omp simd
			for (int i=0; i<N_PTS; ++i) {
				result[i] = eval(points[i], spt);
			}
		}

		template<int N_PTS>
		std::array<double, N_PTS> batch_eval(const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			std::array<double, N_PTS> result;
			batch_eval(result, points, spt);
			return result;
		}

		template<int N_PTS>
		void batch_grad(std::array<RefPoint_t, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			#pragma omp simd
			for (int i=0; i<N_PTS; ++i) {
				result[i] = grad(points[i], spt);
			}
		}

		template<int N_PTS>
		std::array<RefPoint_t, N_PTS> batch_grad(const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			std::array<RefPoint_t, N_PTS> result;
			batch_grad(result, points, spt);
			return result;
		}

		~DOF() = default;

		DOF() : global_idx((size_t) -1), support_idx{}, local_idx{} {}
		
		DOF(const size_t idx, const std::array<size_t,MAX_SUPPORT> spt, const std::array<size_t,MAX_SUPPORT> lcl) :
			global_idx(idx), support_idx(spt), local_idx(lcl) {}
	};


	//check if two DOFs are the same. For this to work correctly, the support_idx (and local_idx) arrays
	//MUST be populated in a consistent manner.
	template<int FEATURE_DIM, int REF_DIM, int MAX_SUPPORT, typename DERIVED>
		requires ( (0<=FEATURE_DIM and FEATURE_DIM<=3) and (REF_DIM==2 or REF_DIM==3) )
	bool operator==(const DOF<FEATURE_DIM,REF_DIM,MAX_SUPPORT,DERIVED>& left,
					const DOF<FEATURE_DIM,REF_DIM,MAX_SUPPORT,DERIVED>& right
					)
	{
		if (left.global_idx != right.global_idx) {return false;}
		if (left.support_idx != right.support_idx) {return false;}

		//with the same mesh feature and same support, the local indices must be the same
		assert(left.local_idx == right.local_idx);
		return true;
	}


	//Base class for all LagrangeDOFs
	template<int REF_DIM, int MAX_SUPPORT, typename DERIVED> requires (REF_DIM==2 or REF_DIM==3)
	struct LagrangeDOF : public DOF<0, REF_DIM, MAX_SUPPORT, DERIVED>
	{
		using Base = DOF<0, REF_DIM, MAX_SUPPORT, DERIVED>;
		using RefPoint_t = typename Base::RefPoint_t;
		using Base::Base;

	};


	//Concept to check if a DOF is a Lagrange type.
	template<typename T>
	struct derives_from_lagrange_dof
	{
		template<int R, int M, typename D>
		static std::true_type test(const LagrangeDOF<R,M,D>*);
		static std::false_type test(...);
		static constexpr bool value = decltype(test(std::declval<T*>()))::value;
	};
	
	template<typename T>
	concept IsLagrangeDOF = derives_from_lagrange_dof<T>::value;

	//Q1 elements on voxels
	//use DERIVED template so that CHARMS variants can inject themselves here
	template<typename DERIVED=void>
	struct VoxelQ1 : public LagrangeDOF<3, 8, 
			std::conditional_t<std::is_void_v<DERIVED>, VoxelQ1<void>, DERIVED>>
	{
		using ActualDerived = std::conditional_t<std::is_void_v<DERIVED>, VoxelQ1<void>, DERIVED>;
	    using Base = LagrangeDOF<3, 8, ActualDerived>;
		using RefPoint_t = typename Base::RefPoint_t;
		using Base::Base;

		//store pre-computed information to help evaluate the basis functions
		// PHI_i(X) = COEF * PROD( 1 + CORNERS[i][j]*X[j])
		static constexpr double COEF = 0.125;
		static constexpr double CORNERS[8][3] {
			{-1, -1, -1},
			{ 1, -1, -1},
			{-1,  1, -1},
			{ 1,  1, -1},
			{-1, -1,  1},
			{ 1, -1,  1},
			{-1,  1,  1},
			{ 1,  1,  1},
		};


		double eval_impl(const RefPoint_t& point, const int spt) const
		{
			const int i = this->local_idx[spt];
			return COEF * (1.0d + CORNERS[i][0]*point[0]) * (1.0d + CORNERS[i][1]*point[1]) * (1.0d + CORNERS[i][2]*point[2]);
		}

		
		RefPoint_t grad_impl(const RefPoint_t& point, const int spt) const
		{
			const int i = this->local_idx[spt];
			RefPoint_t result;

			result[0] = COEF * CORNERS[i][0] * (1.0d + CORNERS[i][1]*point[1]) * (1.0d + CORNERS[i][2]*point[2]);
			result[1] = COEF * CORNERS[i][1] * (1.0d + CORNERS[i][2]*point[2]) * (1.0d + CORNERS[i][0]*point[0]);
			result[2] = COEF * CORNERS[i][2] * (1.0d + CORNERS[i][0]*point[0]) * (1.0d + CORNERS[i][1]*point[1]);

			return result;
		}


		template<int N_PTS>
		void batch_eval(std::array<double, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			const int i = this->local_idx[spt];

			const double C0 = CORNERS[i][0];
			const double C1 = CORNERS[i][1];
			const double C2 = CORNERS[i][2];

			#pragma omp simd
			for (int j=0; j<N_PTS; ++j) {
				result[j] = COEF * (1.0d + C0*points[j][0]) * (1.0d + C1*points[j][1]) * (1.0d + C2*points[j][2]);
			}
		}


		template<int N_PTS>
		void batch_grad(std::array<RefPoint_t, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			const int i = this->local_idx[spt];
			
			const double C0 = CORNERS[i][0];
			const double C1 = CORNERS[i][1];
			const double C2 = CORNERS[i][2];

			#pragma omp simd
			for (int j=0; j<N_PTS; ++j) {
				const double T0 = 1.0d + C0*points[j][0];
				const double T1 = 1.0d + C1*points[j][1];
				const double T2 = 1.0d + C2*points[j][2];

				result[j][0] = COEF * C0 * T1 * T2;
				result[j][1] = COEF * C1 * T2 * T0;
				result[j][2] = COEF * C2 * T0 * T1;
			}
		}
	};


	//Q1 elements on hexahedrons (different corner indexing, likely iso-parametric mapping)
	//use DERIVED template so that CHARMS variants can inject themselves here
	template<typename DERIVED=void>
	struct HexQ1 : public LagrangeDOF<3, 8, 
			std::conditional_t<std::is_void_v<DERIVED>, HexQ1<void>, DERIVED>>
	{
		using ActualDerived = std::conditional_t<std::is_void_v<DERIVED>, HexQ1<void>, DERIVED>;
	    using Base = LagrangeDOF<3, 8, ActualDerived>;
		using RefPoint_t = typename Base::RefPoint_t;
		using Base::Base;

		//store pre-computed information to help evaluate the basis functions
		// PHI_i(X) = COEF * PROD( 1 + CORNERS[i][j]*X[j])
		static constexpr double COEF = 0.125;
		static constexpr double CORNERS[8][3] {
			{-1, -1, -1},
			{ 1, -1, -1},
			{ 1,  1, -1},
			{-1,  1, -1},
			{-1, -1,  1},
			{ 1, -1,  1},
			{ 1,  1,  1},
			{-1,  1,  1},
		};


		double eval_impl(const RefPoint_t& point, const int spt) const
		{
			const int i = this->local_idx[spt];
			return COEF * (1.0d + CORNERS[i][0]*point[0]) * (1.0d + CORNERS[i][1]*point[1]) * (1.0d + CORNERS[i][2]*point[2]);
		}

		
		RefPoint_t grad_impl(const RefPoint_t& point, const int spt) const
		{
			const int i = this->local_idx[spt];
			RefPoint_t result;

			result[0] = COEF * CORNERS[i][0] * (1.0d + CORNERS[i][1]*point[1]) * (1.0d + CORNERS[i][2]*point[2]);
			result[1] = COEF * CORNERS[i][1] * (1.0d + CORNERS[i][2]*point[2]) * (1.0d + CORNERS[i][0]*point[0]);
			result[2] = COEF * CORNERS[i][2] * (1.0d + CORNERS[i][0]*point[0]) * (1.0d + CORNERS[i][1]*point[1]);

			return result;
		}


		template<int N_PTS>
		void batch_eval(std::array<double, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			const int i = this->local_idx[spt];

			const double C0 = CORNERS[i][0];
			const double C1 = CORNERS[i][1];
			const double C2 = CORNERS[i][2];

			#pragma omp simd
			for (int j=0; j<N_PTS; ++j) {
				result[j] = COEF * (1.0d + C0*points[j][0]) * (1.0d + C1*points[j][1]) * (1.0d + C2*points[j][2]);
			}
		}


		template<int N_PTS>
		void batch_grad(std::array<RefPoint_t, N_PTS>& result, const std::array<RefPoint_t, N_PTS>& points, const int spt) const
		{
			const int i = this->local_idx[spt];
			
			const double C0 = CORNERS[i][0];
			const double C1 = CORNERS[i][1];
			const double C2 = CORNERS[i][2];

			#pragma omp simd
			for (int j=0; j<N_PTS; ++j) {
				const double T0 = 1.0d + C0*points[j][0];
				const double T1 = 1.0d + C1*points[j][1];
				const double T2 = 1.0d + C2*points[j][2];

				result[j][0] = COEF * C0 * T1 * T2;
				result[j][1] = COEF * C1 * T2 * T0;
				result[j][2] = COEF * C2 * T0 * T1;
			}
		}
	};

}


