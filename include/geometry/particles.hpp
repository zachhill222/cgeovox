#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "util/box.hpp"
#include "util/matrix.hpp"

#include "optimization/newton.hpp"

#include <cmath>
#include <functional>

namespace gv::geometry{
	//base shape for particles. includes rotatable prism bounding box.
	class Particle
	{
	public:
		using Point_t  = gv::util::Point<3,double>;
		using Box_t    = gv::util::Box<3,double>;
		using Quat_t   = gv::util::Quaternion<double>;
		using Hessian_t = gv::util::Matrix<3,3,double>;

		Particle() {}
		Particle(const Point_t &radii, const Point_t &center, const Quat_t &quaternion, double eps0, double eps1) :
			_radii(radii),
			_center(center),
			_quaternion(quaternion), //controls rotation from global to local coordinates
			_eps0(eps0),
			_eps1(eps1)
			{}
		
		//parameters that ANY particle might need
		//keep all particles the same size in memory for contiguous allocation of multiple types
		Point_t _radii {1,1,1};
		Point_t _center {0,0,0};
		Quat_t  _quaternion {1,0,0,0};
		double _eps0 = 1;
		double _eps1 = 1;


		//access methods.
		Point_t center() const {return _center;}

		//return point in local normalized coordinates
		Point_t tolocal(const Point_t &point) const {return (_quaternion.rotate(point-_center));}

		//convert point in local normalized coordinates to global coordinates
		Point_t toglobal(const Point_t &point) const {return _quaternion.conj().rotate(point)+_center;}

		//convert a direction vector from global coordinates to local
		Point_t to_local_direction(const Point_t &direction) const {return _quaternion.rotate(direction);}

		//check if point in global coordinates is in the bounding box aligned with the local axes
		bool is_in_bbox(const Point_t &point) const {return gv::util::norminfty<3,double>(tolocal(point)) <= 1;}

		//evaluate level set function at specified point in global coordinates
		double eval_level_set(const Point_t &point) const {return _eval_level_set(tolocal(point));}

		//get smallest axis-alligned bounding box (alligned to global axes)
		Box_t bbox() const
		{
			Point_t low, high;
			for (int i=0; i<3; i++)
			{
				Point_t direction {0,0,0};
				
				direction[i] = -1;
				low[i] = support(direction)[i];
				
				direction[i] = 1;
				high[i] = support(direction)[i];
			}
			return Box_t {low, high};
		}

		//check if point is inside particle
		bool contains(const Point_t &point) const
		{
			return eval_level_set(point) <= 1;
		}

		//get the point on the surface closest to the specified point
		virtual Point_t closest_point(const Point_t& globalpoint) const
		{
			std::function<double(Point_t)> fun = [this](const Point_t& point) -> double {return this->_eval_level_set(point)-1.0;};
			std::function<Point_t(Point_t)> gradient = [this](const Point_t& point) -> Point_t {return this->_grad(point);};
			Point_t x0 = tolocal(globalpoint);
			Point_t x  = toglobal(gv::optimization::minimum_distanceNewtonBFGS(fun, gradient, x0, x0));
			
			// std::cout << this->grad(x) / (globalpoint - x) << std::endl;

			return x;
		}


		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t support(const Point_t &direction) const = 0;

		//compute and convert the gradient of the level set from local coordinates back to global
		Point_t grad(const Point_t& globalpoint) const
		{
			Point_t localgrad = _grad(tolocal(globalpoint));
			return _quaternion.conj().rotate(localgrad);
		}

		//get the signed distance from the specified point to the surface
		//if the specified point is inside the particle, the result is negative
		//otherwise the result is positive
		double signed_distance(const Point_t& globalpoint) const
		{
			Point_t closestpoint = closest_point(globalpoint);
			if (contains(globalpoint)) {return -gv::util::norm2(globalpoint-closestpoint);}
			return gv::util::norm2(globalpoint-closestpoint);
		}


	protected:
		//evaluate level set function at specified point in normalized local coordinates
		virtual double _eval_level_set(const Point_t &localpoint) const = 0;

		//compute gradient of the level set at a point in the local coordinate system
		virtual Point_t _grad(const Point_t &localpoint) const = 0;
	};


	//prism class
	class Prism : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3,double>;
		using Quat_t  = gv::util::Quaternion<double>;

		Prism() : Particle() {}
		Prism(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = to_local_direction(direction);
			Point_t localpoint = _radii;
			for (int i=0; i<3; i++)
				{
					if (rotated_direction[i]<0)
					localpoint[i] *= -1;
				}
			return toglobal(localpoint);
		}

		

	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::norminfty(localpoint/_radii);}

		Point_t _grad(const Point_t &localpoint) const override
		{
			if (gv::util::squaredNorm(localpoint)==0) {return Point_t{0,0,0};}

			//get maximum component and indices
			Point_t normalized = localpoint/_radii;
			double max_val = gv::util::norminfty(normalized);
			Point_t result {0,0,0};

			//the level set only increases only when moving in the direction with largest coordinate value
			for (int i=0; i<3; i++) {
				if (gv::util::abs(normalized[i])==max_val) {
					result[i] = gv::util::sgn(localpoint[i])/_radii[i];
				}
			}

			return result;
		}
	};


	//sphere class
	class Sphere : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3,double>;
		using Quat_t  = gv::util::Quaternion<double>;

		Sphere() : Particle() {}
		//same constructor signature. only radii[0] is used.
		Sphere(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			return _radii[0] * gv::util::normalize(direction);
		}

		Point_t closest_point(const Point_t& globalpoint) const override
		{
			Point_t direction = globalpoint - _center;
			return support(direction);
		}


	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::squaredNorm(localpoint/_radii);}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			return 2.0*localpoint/_radii[0];
		}
	};



	//ellipsoid class
	class Ellipsoid : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3,double>;
		using Quat_t  = gv::util::Quaternion<double>;

		Ellipsoid() : Particle() {}
		Ellipsoid(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = to_local_direction(direction);
			Point_t localpoint = _radii*_radii*rotated_direction;
			double c = gv::util::norm2(_radii*rotated_direction);
			return toglobal(localpoint/c);
		}

		

	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::squaredNorm(localpoint/_radii);}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			return 2.0*localpoint/(_radii*_radii);
		}
	};


	//right-circular cyllinder class. height axis parallel to z-axis in local coordinates.
	class Cylinder : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3,double>;
		using Quat_t  = gv::util::Quaternion<double>;

		Cylinder() : Particle() {}
		Cylinder(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = to_local_direction(direction);
			Point_t normalized {0,0,1};
			if (rotated_direction[2] < 0) {normalized[2] = -1;}

			double R = std::sqrt(rotated_direction[0]*rotated_direction[0] + rotated_direction[1]*rotated_direction[1]);
			if (R>0)
			{
				normalized[0] = rotated_direction[0]/R;
				normalized[1] = rotated_direction[1]/R;
			}

			return this->toglobal(normalized*_radii);
		}

		

	protected:
		double _eval_level_set(const Point_t &localpoint) const override
		{
			Point_t normalized = localpoint/_radii;
			
			double R2 = normalized[0]*normalized[0]+normalized[1]*normalized[1];
			double H  = gv::util::abs(normalized[2]);
			return std::max(R2, H*H);
		}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			
			Point_t normalized = localpoint/_radii;
			Point_t result {0,0,0};
			
			double R2 = normalized[0]*normalized[0] + normalized[1]*normalized[1];
			double H  = gv::util::abs(normalized[2]);

			if (R2>=H)
			{
				result[0] = 2.0*localpoint[0]/(_radii[0]*_radii[0]);
				result[1] = 2.0*localpoint[1]/(_radii[1]*_radii[1]);
			}

			if (H>=R2)
			{
				result[2] = 2.0*localpoint[2]/(_radii[2]*_radii[2]);
			}
			return result;
		}
	};


	//super-ellipsoid class.
	class SuperEllipsoid : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t   = gv::util::Box<3,double>;
		using Quat_t  = gv::util::Quaternion<double>;

		SuperEllipsoid() : Particle() {}
		SuperEllipsoid(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = to_local_direction(direction)*_radii;
			
			//get omega
			double c = gv::util::sgn(rotated_direction[0])*std::pow(gv::util::abs(rotated_direction[0]), 1.0/(2.0-_eps1));
			double s = gv::util::sgn(rotated_direction[1])*std::pow(gv::util::abs(rotated_direction[1]), 1.0/(2.0-_eps1));
			double omega = std::atan2(s, c); //in [-pi,pi]

			//get eta
			c = std::pow(gv::util::abs(rotated_direction[0]), 1.0/(2.0-_eps0));
			s = gv::util::sgn(rotated_direction[2]) * std::pow( gv::util::abs( rotated_direction[2]*cos_pow(omega,2.0-_eps1)), 1.0/(2.0-_eps0));

			double eta = atan2(s, c); //in [-pi/2,pi/2] because x >= 0

			//get normal in global coordinates
			Point_t localpoint = _parametric(eta, omega);
			return this->toglobal(localpoint);
		}

		


	protected:
		//signed cos(theta)^eps
		static const double cos_pow(const double theta, const double eps)
		{
			double C = cos(theta);
			return gv::util::sgn(C)*std::pow(gv::util::abs(C), eps);
		}
		//signed sin(theta)^eps
		static const double sin_pow(const double theta, const double eps)
		{
			double S = sin(theta);
			return gv::util::sgn(S)*std::pow(gv::util::abs(S), eps);
		}

		//evaluate level set function
		double _eval_level_set(const Point_t &localpoint) const override
		{
			Point_t normalized = localpoint/_radii;

			double a = std::pow(normalized[0]*normalized[0], 1.0/_eps1) + std::pow(normalized[1]*normalized[1], 1.0/_eps1);
			return std::pow(a, _eps1/_eps0) + std::pow(normalized[2]*normalized[2], 1.0/_eps0);
		}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			Point_t normalized = localpoint/_radii;

			double C  = 2.0/_eps0;
			double a  = std::pow(normalized[0]*normalized[0], 1.0/_eps1) + std::pow(normalized[1]*normalized[1], 1.0/_eps1);
			double ETA = std::pow(a, (_eps1-_eps0)/_eps0);
			Point_t result {C*ETA*std::pow(gv::util::abs(normalized[0]), (2.0-_eps1)/_eps1),
							C*ETA*std::pow(gv::util::abs(normalized[1]), (2.0-_eps1)/_eps1),
							C*std::pow(gv::util::abs(normalized[2]), (2.0-_eps1)/_eps1)
						};

			for (int i=0; i<3; i++) {result[i] *= gv::util::sgn(normalized[i]);} //correct for taking the absolute value before evaluating the exponential

			return result/_radii;
		}

		//get point in local coordinates from the parametric representation of the particle surface
		Point_t _parametric(const double eta, const double omega) const{
			//compute sines and cosines
			double C_eta   = cos_pow(eta, _eps0);
			double S_eta   = sin_pow(eta, _eps0);
			double C_omega = cos_pow(omega, _eps1);
			double S_omega = sin_pow(omega, _eps1);

			Point_t localpoint = Point_t {C_eta*C_omega, C_eta*S_omega, S_eta};
			return localpoint*_radii;
		}
	};


	//check equality of two particles. needed for octrees.
	bool operator==(const Particle &left, const Particle &right)
	{
		if (left._center!=right._center) {return false;}
		if (left._radii!=right._radii) {return false;}
		if (left._quaternion!=right._quaternion) {return false;}
		if (left._eps0!=right._eps0) {return false;}
		if (left._eps1!=right._eps1) {return false;}
		return true;
	}
}