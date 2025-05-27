#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"


namespace gv::util
{
	///Octree for points in space.
	template <int dim=3, size_t n_data=32>
	class PointOctree : public gv::util::BasicOctree<gv::util::Point<dim,double>, dim, false, n_data>
	{
	public:
		PointOctree() : gv::util::BasicOctree<gv::util::Point<dim,double>, dim, false, n_data>() {}
		PointOctree(const gv::util::Box<dim> &bbox) : gv::util::BasicOctree<gv::util::Point<dim,double>, dim, false, n_data>(bbox) {}

	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const gv::util::Point<dim,double> &data) const override {return box.contains(data);}
	};


	///Octree for boxes in space (e.g., good for voxel meshes)
	// template <int dim=3, size_t n_data=16>
	// class BoxOctree : public gv::util::BasicOctree<gv::util::Box<dim>, dim, true, n_data>
	// {
	// public:
	// 	BoxOctree() : gv::util::BasicOctree<gv::util::Box<dim>, dim, true, n_data>() {}
	// 	BoxOctree(const gv::util::Box<dim> &bbox) : gv::util::BasicOctree<gv::util::Box<dim>, dim, true, n_data>(const gv::util::Box<dim> &bbox) {}

	// private:
	// 	bool is_data_valid(const gv::util::Box<dim> &box, const gv::util::Box<dim> &data) const override {return data.instersects(box);}
	// }
}