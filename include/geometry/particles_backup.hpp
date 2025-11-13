#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "util/box.hpp"

#include <cmath>


namespace gv::geometry{
	//base shape for particles. includes rotatable prism bounding box.
	class Prism
	{
	public:
		Prism() {}
		Prism(const gv::util::Point<3,double> &radii, const gv::util::Point<3,double> &center, const gv::util::Quaternion<double> quaternion = gv::util::Quaternion<double> {1,0,0,0}) : _quaternion(quaternion), _center(center), _radii(radii) {}
		Prism(const gv::util::Point<3,double> &radii, double eps[2], const gv::util::Point<3,double> &center, const gv::util::Quaternion<double> quaternion = gv::util::Quaternion<double> {1,0,0,0}) : _quaternion(quaternion), _center(center), _radii(radii) {}
		//return point in local normalized coordinates
		inline gv::util::Point<3,double> tolocal(const gv::util::Point<3,double> &point) const {return (_quaternion.rotate(point-_center))/_radii;}

		//convert point in local normalized coordinates to global coordinates
		inline gv::util::Point<3,double> toglobal(const gv::util::Point<3,double> &point) const {return _quaternion.conj().rotate(point*_radii)+_center;}

		//check if point in global coordinates is in the bounding box
		inline bool is_in_bbox(const gv::util::Point<3,double> &point) const {return gv::util::norminfty<3,double>(tolocal(point)) <= 1;}

		//evaluate level set function at specified point in global coordinates
		inline double eval_level_set(const gv::util::Point<3,double> &point) const {return _eval_level_set(tolocal(point));}

		//get center
		inline gv::util::Point<3,double> center() const {return _center;}

		//get radii
		inline gv::util::Point<3,double> radii() const {return _radii;}

		//get quaternion
		inline gv::util::Quaternion<double> quaternion() const {return _quaternion;}

		//get smallest axis-alligned bounding box
		gv::util::Box<3> bbox() const
		{
			gv::util::Point<3,double> low, high;
			for (int i=0; i<3; i++)
			{
				gv::util::Point<3,double> direction {0,0,0};
				
				direction[i] = -1;
				low[i] = support(direction)[i];
				
				direction[i] = 1;
				high[i] = support(direction)[i];
			}
			// std::cout << (gv::util::Box<3> {low, high}).tostr();
			return gv::util::Box<3> {low, high};
		}

		//check if point is inside particle
		bool contains(const gv::util::Point<3,double> &point) const
		{
			gv::util::Point<3,double> localpoint = tolocal(point);
			if (gv::util::norminfty<3,double>(localpoint) <= 1) {return _eval_level_set(localpoint) <= 1;}
			return false;
		}


		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual gv::util::Point<3,double> support(const gv::util::Point<3,double> &direction) const
		{
			gv::util::Point<3,double> rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			gv::util::Point<3,double> localpoint {1,1,1};
			for (int i=0; i<3; i++)
				{
					if (rotated_direction[i]<0)
					localpoint[i] = -1;
				}
			return toglobal(localpoint);
		}


	protected:
		//quaternion to rotate from global coordinate system to particle coordinate system
		gv::util::Quaternion<double> _quaternion;

		//origin of particle coordinate system
		gv::util::Point<3,double> _center;

		//major radii in particle coordinate system
		gv::util::Point<3,double> _radii;

		//evaluate level set function at specified point in normalized local coordinates
		virtual double _eval_level_set(const gv::util::Point<3,double> &localpoint) const {return gv::util::norminfty<3,double>(localpoint);}
	};


	//ellipsoid class
	class Ellipsoid : public Prism
	{
	public:
		Ellipsoid() : Prism() {}
		Ellipsoid(const gv::util::Point<3,double> &radii, const gv::util::Point<3,double> &center, const gv::util::Quaternion<double> quaternion = gv::util::Quaternion<double> {1,0,0,0}) : Prism(radii, center, quaternion) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		gv::util::Point<3,double> support(const gv::util::Point<3,double> &direction) const override
		{
			gv::util::Point<3,double> rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			gv::util::Point<3,double> localpoint = gv::util::normalize(rotated_direction);
			return this->toglobal(localpoint);
		}

	protected:
		double _eval_level_set(const gv::util::Point<3,double> &localpoint) const override {return gv::util::squaredNorm(localpoint);}
	};


	//right-circular cyllinder class. height axis parallel to z-axis in local coordinates.
	class Cylinder : public Prism
	{
	public:
		Cylinder() : Prism() {}
		Cylinder(const gv::util::Point<3,double> &radii, const gv::util::Point<3,double> &center, const gv::util::Quaternion<double> quaternion = gv::util::Quaternion<double> {1,0,0,0}) : Prism(radii, center, quaternion) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		gv::util::Point<3,double> support(const gv::util::Point<3,double> &direction) const override
		{
			gv::util::Point<3,double> rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			gv::util::Point<3,double> localpoint {0,0,1};
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
		double _eval_level_set(const gv::util::Point<3,double> &localpoint) const override{return std::max(localpoint[0]*localpoint[0]+localpoint[1]*localpoint[1], gv::util::abs(localpoint[2]));}
	};


	//super-ellipsoid class.
	class SuperEllipsoid : public Prism
	{
	public:
		SuperEllipsoid() : Prism() {}
		SuperEllipsoid(const gv::util::Point<3,double> &radii, double eps[2], const gv::util::Point<3,double> &center, const gv::util::Quaternion<double> quaternion = gv::util::Quaternion<double> {1,0,0,0}) : \
					Prism(radii, center, quaternion), _eps {eps[0], eps[1]}, _powers {1.0/eps[0], 1.0/eps[1], eps[1]/eps[0]}, _invpowers {1.0/(2.0-eps[0]), 1.0/(2.0-eps[1])} {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		gv::util::Point<3,double> support(const gv::util::Point<3,double> &direction) const override
		{
			gv::util::Point<3,double> rotated_direction = this->_quaternion.rotate(direction)*this->_radii;
			//get omega
			double x = gv::util::sgn(rotated_direction[0])*std::pow(gv::util::abs(rotated_direction[0]), _invpowers[1]);
			double y = gv::util::sgn(rotated_direction[1])*std::pow(gv::util::abs(rotated_direction[1]), _invpowers[1]);
			double omega = std::atan2(y, x); //in [-pi,pi]

			//get eta
			x = std::pow(gv::util::abs(rotated_direction[0]), _invpowers[0]);
			y = gv::util::sgn(rotated_direction[2]) * std::pow( gv::util::abs( rotated_direction[2]*cos_pow(omega,2.0-_eps[1]) ) , _invpowers[0]);

			double eta = atan2(y, x); //in [-pi/2,pi/2] because x >= 0

			//get normal in global coordinates
			gv::util::Point<3,double> localpoint = _parametric(eta, omega);
			
			return this->toglobal(localpoint);
		}

		inline double eps(const int idx) const {return _eps[idx];}

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


		//shape parameters
		double _eps[2] {1,1};

		//common exponents
		double _powers[3] {1,1,1};
		double _invpowers[2] {1,1};

		//evaluate level set function
		virtual double _eval_level_set(const gv::util::Point<3,double> &localpoint) const
		{
			double a = std::pow(localpoint[0]*localpoint[0], _powers[1]) + std::pow(localpoint[1]*localpoint[1], _powers[1]);
			return std::pow(a, _powers[2]) + std::pow(localpoint[2]*localpoint[2], _powers[0]);
		}

		//get point in local coordinates from the parametric representation of the particle surface
		gv::util::Point<3,double> _parametric(const double eta, const double omega) const{
			//compute sines and cosines
			double C_eta = cos_pow(eta, _eps[0]);
			double S_eta = sin_pow(eta, _eps[0]);
			double C_omega = cos_pow(omega, _eps[1]);
			double S_omega = sin_pow(omega, _eps[1]);

			gv::util::Point<3,double> localpoint = gv::util::Point<3,double> {C_eta*C_omega, C_eta*S_omega, S_eta};
			return localpoint;
		}
	};


	//check equality of two particles. needed for octrees.
	bool operator==(const Prism &left, const Prism &right)
	{
		return left.center()==right.center() and left.radii()==right.radii() and left.quaternion()==right.quaternion();
	}

	bool operator==(const Ellipsoid &left, const Ellipsoid &right)
	{
		return left.center()==right.center() and left.radii()==right.radii() and left.quaternion()==right.quaternion();
	}

	bool operator==(const SuperEllipsoid &left, const SuperEllipsoid &right)
	{
		if (left.center()==right.center() and left.radii()==right.radii() and left.quaternion()==right.quaternion())
		{
			return left.eps(0)==right.eps(0) and left.eps(1)==right.eps(1);
		}
		return false;
	}

}