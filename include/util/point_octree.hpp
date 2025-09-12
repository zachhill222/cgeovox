#pragma once

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

namespace gv::util
{
	///Octree for points in space.
	template<int dim=3, int n_data=32>
	class PointOctree : public gv::util::BasicOctree_Point<gv::util::Point<dim,double>, dim, n_data>
	{
	public:
		using Data_t = gv::util::Point<dim,double>;
		PointOctree() : //if bounding box is unknown ahead of time
			gv::util::BasicOctree_Point<Data_t, dim, n_data>(1024) {}

		PointOctree(const gv::util::Box<dim> &bbox, const size_t capacity=1024) :
			gv::util::BasicOctree_Point<Data_t, dim, n_data>(bbox, capacity) {}

		//copy constructor with (possible) change of bounding box
		constexpr PointOctree(const PointOctree &other, const gv::util::Box<dim>& new_bbox) :
			gv::util::BasicOctree_Point<gv::util::Point<dim,double>, dim, n_data>(new_bbox, other.capacity())
		{
			for (size_t idx=0; idx<other.size(); idx++) {this->push_back(other[idx]);}
		}

	private:
		bool is_data_valid(const gv::util::Box<dim> &box, const Data_t &data) const override {return box.contains(data);}
	};
}