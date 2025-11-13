#pragma once

#include "util/matrix.hpp"
#include "util/point.hpp"

#include "concepts.hpp"

#include <functional>


#define LINESEARCH_A0   1.0
#define LINESEARCH_RHO  0.5
#define LINESEARCH_C    0.75
#define LINESEARCH_MAXITER 128

#define NEWTON_MAXITER 16
#define GRADTOL 1E-10

namespace gv::optimization
{
	template<int dim, Scalar Scalar_t>
	using Point = gv::util::Point<dim,Scalar_t>;

	template<int dim, Scalar Scalar_t>
	using Matrix = gv::util::Matrix<dim, dim, Scalar_t>;

	template<int dim, Scalar Scalar_t>
	using ScalarFun_t = std::function<double(Point<dim,Scalar_t>)>;

	template<int dim, Scalar Scalar_t>
	using GradFun_t = std::function<Point<dim,Scalar_t>(Point<dim,Scalar_t>)>;


	template<int dim, Scalar Scalar_t>
	Scalar_t linesearch(const Point<dim,Scalar_t> &x0, const ScalarFun_t<dim,Scalar_t>& f, const Point<dim,Scalar_t> &grad0, const Point<dim,Scalar_t> &direction)
	{
		Scalar_t alpha = (Scalar_t) LINESEARCH_A0;
		Scalar_t f0    = f(x0);
		Scalar_t CgradDir = LINESEARCH_C * gv::util::dot(grad0,direction);

		for (int i=0; i<LINESEARCH_MAXITER; i++)
		{
			if (f(x0+alpha*direction) <= f0 + alpha*CgradDir) {return alpha;}
			alpha *= LINESEARCH_RHO;
		}

		std::cout << "WARNING: Linesearch did not converge" << std::endl;
		return alpha;
	}


	template<int dim, Scalar Scalar_t>
	Point<dim,Scalar_t> quasiNewtonDFP(const ScalarFun_t<dim,Scalar_t>&f, const GradFun_t<dim,Scalar_t>&grad, const Point<dim,Scalar_t> &x0)
	{
		//initialize estimate of the inverse hessian
		Matrix<dim,Scalar_t> D;
		D.eye();

		//initialize last-iteration variables
		Point<dim,Scalar_t> last_grad = grad(x0);
		Point<dim,Scalar_t> last_x = x0;
		Point<dim,Scalar_t> next_x;
		//main loop
		for (int k=0; k<NEWTON_MAXITER; k++)
		{
			//get search direction and stepsize
			Point<dim,Scalar_t> direction = -(D*last_grad);
			Scalar_t lambda = linesearch(f, last_x, last_grad, direction);
			Point<dim,Scalar_t> step = lambda*direction;

			//update and check for convergence
			next_x    = last_x + step;
			Point<dim,Scalar_t> next_grad = grad(next_x);
			if (gv::util::norminfty(next_grad) < GRADTOL) {return next_x;}

			//update estimate of the inverse hessian
			Point<dim,Scalar_t> delta_grad = next_grad - last_grad;
			D += (((Scalar_t) 1)/gv::util::dot(step,delta_grad)) * gv::util::outer(step,step);
			Point<dim,Scalar_t> D_grad = D*delta_grad;
			Scalar_t c = gv::util::dot(delta_grad, D_grad);
			D -= ((Scalar_t) 1/c) * gv::util::outer(D_grad, D_grad);

			//update last-iteration variables
			last_x    = next_x;
			last_grad = next_grad;
		}

		std::cout << "WARNING: Newton (DFP) did not converge" << std::endl;
		return next_x;
	}


	//solve min(0.5*|x-x0|^2) such that g(x)=0
	template<int dim, Scalar Scalar_t>
	Point<dim,Scalar_t> minimum_distanceNewtonBFGS(
		const ScalarFun_t<dim,Scalar_t>&g,
		const GradFun_t<dim,Scalar_t>&grad, 
		const Point<dim,Scalar_t> &x0, 
		const Point<dim,Scalar_t> &initial_guess)
	{
		//initialize estimate of the hessian of g
		Matrix<dim,Scalar_t> B;
		B.eye();

		//initialize 4x4 matrix (hessian + constraint)
		Matrix<dim+1,Scalar_t> M;

		//initialize last-iteration variables
		Point<dim,Scalar_t> last_grad = grad(initial_guess);
		Point<dim,Scalar_t> last_x = initial_guess;
		Point<dim,Scalar_t> next_x;
		Scalar_t lambda = 0;
		//main loop
		for (int k=0; k<NEWTON_MAXITER; k++)
		{
			//assemble right hand side and check for convergence
			Point<dim+1,Scalar_t> rhs;
			for (int i=0; i<dim; i++)
			{
				rhs[i] = x0[i] - last_x[i] + lambda*last_grad[i];
			}
			rhs[dim] = g(last_x);

			if (gv::util::norminfty(rhs) < GRADTOL) {return last_x;}
			if (gv::util::norminfty(last_grad) < GRADTOL or gv::util::norminfty(last_grad)>1E10) {return last_x;}

			//assemble hessian+constraint matrix
			for (int i=0; i<dim; i++)
			{
				//hessian block
				for (int j=0; j<dim; j++)
				{
					M(i,j) = lambda*B(i,j);
					if (i==j) {M(i,j) -= (Scalar_t) 1;}
				}

				//gradient blocks
				M(i,dim) = last_grad[i];
				M(dim,i) = last_grad[i];

				//M(dim,dim)=0
			}


			//solve system
			Point<dim+1,Scalar_t> step = -(M/rhs);
			Point<dim,Scalar_t> dir_x;
			for (int i=0; i<dim; i++) {dir_x[i] = step[i];}


			//update x
			// Scalar_t t = linesearch(last_x, g, last_grad, dir_x);
			// Point<dim,Scalar_t> step_x = t*dir_x;
			Point<dim,Scalar_t> step_x = dir_x;
			next_x    = last_x + step_x;
			Point<dim,Scalar_t> next_grad = grad(next_x);
			
			//update lambda
			lambda += step[dim];
			// double max_grad = gv::util::norminfty(next_grad);
			// int max_grad_ind = 0;
			// for (int i=0; i<dim; i++)
			// {
			// 	if (gv::util::abs(next_grad[i]) == max_grad)
			// 	{
			// 		max_grad_ind = i;
			// 	}
			// }
			// lambda = (x0[max_grad_ind]-next_x[max_grad_ind])/next_grad[max_grad_ind];


			//update estimate of the inverse hessian (BFGS update rule)
			Point<dim,Scalar_t> delta_grad = next_grad - last_grad;
			if (gv::util::norminfty(delta_grad)<GRADTOL) {return next_x;}

			B += (((Scalar_t) 1)/gv::util::dot(step_x,delta_grad)) * gv::util::outer(delta_grad,delta_grad);
			Point<dim,Scalar_t> B_step_x = B*step_x;
			Scalar_t c = gv::util::dot(step_x, B_step_x);
			B -= ((Scalar_t) 1/c) * gv::util::outer(B_step_x, B_step_x);

			//update last-iteration variables
			last_x    = next_x;
			last_grad = next_grad;
		}

		// std::cout << "WARNING: constrained Newton (BFGS) did not converge" << std::endl;
		return next_x;
	}
}
