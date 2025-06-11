#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"
#include "util/octree_util.hpp"

namespace gv::util
{
	///Octree for points in space.
	template <int dim=3, size_t n_data=32>
	class PointOctree : public gv::util::BasicOctree<gv::util::Point<dim,double>, dim, n_data>
	{
	public:
		friend class gv::util::OctreeInspector<PointOctree>;
		
		PointOctree(const size_t capacity=1024) : //if bounding box is unknown ahead of time
			gv::util::BasicOctree<gv::util::Point<dim,double>, dim, n_data>(capacity) {}

		PointOctree(const gv::util::Box<dim> &bbox, const size_t capacity=1024) :
			gv::util::BasicOctree<gv::util::Point<dim,double>, dim, n_data>(bbox, capacity) {}

	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const gv::util::Point<dim,double> &data) const override {return box.contains(data);}
	};
}