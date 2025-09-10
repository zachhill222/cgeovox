#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "util/box.hpp"

#include <cmath>


namespace gv::geometry{
	//base shape for particles. includes rotatable prism bounding box.
	class Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;
		using Quat_t = gv::util::Quaternion<double>;

		Particle() {}
		Particle(const Point_t &radii, const Point_t &center, const Quat_t &quaternion, double eps0, double eps1) :
			_radii(radii),
			_center(center),
			_quaternion(quaternion),
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
		Point_t tolocal(const Point_t &point) const {return (_quaternion.rotate(point-_center))/_radii;}

		//convert point in local normalized coordinates to global coordinates
		Point_t toglobal(const Point_t &point) const {return _quaternion.conj().rotate(point*_radii)+_center;}

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
			// std::cout << (Box_t {low, high}).tostr();
			return Box_t {low, high};
		}

		//check if point is inside particle
		bool contains(const Point_t &point) const
		{
			Point_t localpoint = tolocal(point);
			if (gv::util::norminfty<3,double>(localpoint) <= 1) {return _eval_level_set(localpoint) <= 1;}
			return false;
		}


		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t support(const Point_t &direction) const {assert(false); return Point_t {0,0,0};}


	protected:
		//evaluate level set function at specified point in normalized local coordinates
		virtual double _eval_level_set(const Point_t &localpoint) const {assert(false); return 0;}
	};


	//prism class
	class Prism : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;
		using Quat_t = gv::util::Quaternion<double>;

		Prism() : Particle() {}
		Prism(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			Point_t localpoint {1,1,1};
			for (int i=0; i<3; i++)
				{
					if (rotated_direction[i]<0)
					localpoint[i] = -1;
				}
			return toglobal(localpoint);
		}

	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::norminfty<3,double>(localpoint);}
	};



	//ellipsoid class
	class Ellipsoid : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;
		using Quat_t = gv::util::Quaternion<double>;

		Ellipsoid() : Particle() {}
		Ellipsoid(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			Point_t localpoint = gv::util::normalize(rotated_direction);
			return this->toglobal(localpoint);
		}

	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::squaredNorm(localpoint);}
	};


	//right-circular cyllinder class. height axis parallel to z-axis in local coordinates.
	class Cylinder : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;
		using Quat_t = gv::util::Quaternion<double>;

		Cylinder() : Particle() {}
		Cylinder(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			Point_t localpoint {0,0,1};
			if (rotated_direction[2] < 0) {localpoint[2] = -1;}
			
			double R = std::sqrt(rotated_direction[0]*rotated_direction[0] + rotated_direction[1]*rotated_direction[1]);
			if (R>0)
			{
				localpoint[0]/=R;
				localpoint[1]/=R;
			}

			return this->toglobal(localpoint);
		}

	protected:
		double _eval_level_set(const Point_t &localpoint) const override{return std::max(localpoint[0]*localpoint[0]+localpoint[1]*localpoint[1], gv::util::abs(localpoint[2]));}
	};


	//super-ellipsoid class.
	class SuperEllipsoid : public Particle
	{
	public:
		using Point_t = gv::util::Point<3,double>;
		using Box_t = gv::util::Box<3,double>;
		using Quat_t = gv::util::Quaternion<double>;

		SuperEllipsoid() : Particle() {}
		SuperEllipsoid(const Point_t &radii, const Point_t &center, const Quat_t quaternion = Quat_t {1,0,0,0}, const double eps0=1, const double eps1=1):
			Particle(radii, center, quaternion, eps0, eps1) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		Point_t support(const Point_t &direction) const override
		{
			Point_t rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			//get omega
			double x = gv::util::sgn(rotated_direction[0])*std::pow(gv::util::abs(rotated_direction[0]), 1.0/(2.0-_eps1));
			double y = gv::util::sgn(rotated_direction[1])*std::pow(gv::util::abs(rotated_direction[1]), 1.0/(2.0-_eps1));
			double omega = std::atan2(y, x); //in [-pi,pi]

			//get eta
			x = std::pow(gv::util::abs(rotated_direction[0]), 1.0/(2.0-_eps0));
			y = gv::util::sgn(rotated_direction[2]) * std::pow( gv::util::abs( rotated_direction[2]*cos_pow(omega,2.0-_eps1) ) , 1.0/(2.0-_eps0));

			double eta = atan2(y, x); //in [-pi/2,pi/2] because x >= 0

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
			double a = std::pow(localpoint[0]*localpoint[0], 1.0/_eps1) + std::pow(localpoint[1]*localpoint[1], 1.0/_eps1);
			return std::pow(a, _eps1/_eps0) + std::pow(localpoint[2]*localpoint[2], 1.0/_eps0);
		}

		//get point in local coordinates from the parametric representation of the particle surface
		Point_t _parametric(const double eta, const double omega) const{
			//compute sines and cosines
			double C_eta   = cos_pow(eta, _eps0);
			double S_eta   = sin_pow(eta, _eps0);
			double C_omega = cos_pow(omega, _eps1);
			double S_omega = sin_pow(omega, _eps1);

			Point_t localpoint = Point_t {C_eta*C_omega, C_eta*S_omega, S_eta};
			return localpoint;
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