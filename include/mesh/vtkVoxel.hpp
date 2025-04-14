#pragma once

#include "util/point.hpp"

#include <armadillo>

namespace gv::mesh{
	class Voxel {
	public:
		Voxel() {}
		static const int vtkID = 11;
		static const size_t nNodes = 8;

		static constexpr const double gauss_quad_3_location[3] {-0.7745966692414834, 0,                    0.7745966692414834};
		static constexpr const double gauss_quad_3_weight[3]   { 0.8888888888888889, 0.555555555555555556, 0.8888888888888889};

		static constexpr const double gauss_quad_4_location[4] {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
		static constexpr const double gauss_quad_4_weight[4]   { 0.3478548451374538,  0.6521451548625461, 0.6521451548625461, 0.3478548451374538};

		static constexpr const double gauss_quad_5_location[5] {-0.9061798459386640, -0.5384693101056831, 0,                  0.5384693101056831, 0.9061798459386640};
		static constexpr const double gauss_quad_5_weight[5]   { 0.2369268850561891,  0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};

		//pointers to node locations
		const gv::util::Point<3,double>* nodes[8];

		//get signs for basis functions. All bases are the product of three functions of the form 0.5*(1+C*x) where C is stored below.
		static constexpr const int _basis_signs_list[24] {
			-1,-1,-1, \
			 1,-1,-1, \
			-1, 1,-1, \
			 1, 1,-1, \
			-1,-1, 1, \
			 1,-1, 1, \
			-1, 1, 1, \
			 1, 1, 1
			};
		
		int basis_sign(const size_t i, const size_t axis) const {return _basis_signs_list[3*i + axis];}

		//evaluate basis function i at a point in the reference element
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
		double integrate_mass(const size_t basis1, const size_t basis2) const
		{
			gv::util::Point<3,double> H = *(nodes[7]) - *(nodes[0]);
			double jac = 0.125 * H[0] * H[1] * H[2];
			// double jac = 0.125; //testing

			//loop through quadrature points
			// double value = 0.0;
			// for (int i=0; i<4; i++)
			// {
			// 	H[0] = gauss_quad_4_location[i];
			// 	for (int j=0; j<4; j++)
			// 	{
			// 		H[1] = gauss_quad_4_location[j];
			// 		for (int k=0; k<4; k++)
			// 		{
			// 			H[2] = gauss_quad_4_location[k];
			// 			value += eval_basis(basis1,H)*eval_basis(basis2,H);
			// 		}
			// 	}
			// }

			double value = 0.037037037037037035; // 1/27
			for (int axis=0; axis<3; axis++)
			{
				if (basis_sign(basis1,axis)==basis_sign(basis2,axis)) {value *= 2.0;}
			}

			return value*jac;
		}

		// //get jacobian matrix
		// arma::dmat jacobian(const gv::util::Point<3,double> &point) const
		// {
		// 	gv::util::Point<3,double> rows[3];
		// 	for (int i=0; i<8; i++) //basis function
		// 	{
		// 		gv::util::Point<3,double> grad = eval_grad_basis(point);
		// 		for (j=0; j<3; j++) //rows
		// 		{
		// 			row[j] += nodes[i][j] * grad;
		// 		}
		// 	}

		// 	arma::dmat result(3,3, arma::fill:zeros);
		// 	result.row(0) = rows[0];
		// 	result.row(1) = rows[1];
		// 	result.row(2) = rows[2];

		// 	return result;
		// }

		// //get jacobian determinant
		// arma::dmat jacobian_det(const gv::util::Point<3,double> &point) const
		// {
		// 	// return gv::util::abs(arma::det(jacobian(point)));
		// 	return arma::det(jacobian(point));
		// }
	};
}
