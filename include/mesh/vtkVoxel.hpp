#pragma once

#include "util/point.hpp"

namespace gv::mesh{
	class Voxel {
	public:
		Voxel() {}
		static const int vtkID = 11;
		static const size_t nNodes = 8;

		//get signs for basis functions. All bases are the product of three functions of the form 0.5*(1+C*x) where C is stored below.
		static constexpr const int _basis_signs[24] {
			-1,-1,-1, \
			-1, 1,-1, \
			 1, 1,-1, \
			 1, 1,-1, \
			-1,-1, 1, \
			 1,-1, 1, \
			-1, 1, 1, \
			 1, 1, 1
			};
		
		int basis_sign(const size_t i, const size_t axis) const {return _basis_signs[3*i + axis];}

		//evaluate basis function i
		double eval_basis(const size_t i, const gv::util::Point<3,double> &point) const
		{
			return 0.125 * (1+basis_sign(i,0)*point[0]) * (1+basis_sign(i,1)*point[1]) * (1+basis_sign(i,2)*point[2]);
		}

		//evaluate gradient of basis function i
		gv::util::Point<3,double> eval_grad_basis(const size_t i, const gv::util::Point<3,double> &point) const
		{
			gv::util::Point<3,double> result;

			result[0] = 0.125 * basis_sign(i,0)              * (1+basis_sign(i,1)*point[1]) * (1+basis_sign(i,2)*point[2]);
			result[1] = 0.125 * (1+basis_sign(i,0)*point[0]) * basis_sign(i,1)              * (1+basis_sign(i,2)*point[2]);
			result[2] = 0.125 * (1+basis_sign(i,0)*point[0]) * (1+basis_sign(i,1)*point[1]) * basis_sign(i,2);
			return result;
		}

		//evaluate integral of product of two basis functions over element with size H
		double integrate_mass(const size_t i, const size_t j, const gv::util::Point<3,double> &H) const
		{
			double value = 1;
			for (size_t axis=0; axis<3; axis++)
			{
				if (basis_sign(i,axis)==basis_sign(j,axis)) {value *= 0.66666666666666666666666667 * H[axis];}
				else {value *= 0.333333333333333333333333333 * H[axis];}
			}
			return value;
		}
	};
}
