#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"

#include <cmath>


namespace gv::geometry{
	template <typename T=double>
	using Point_t = gv::util::Point<3,T>;

	template <typename T=double>
	using Quat_t = gv::util::Quaternion<T>;

	//base shape for particles. includes rotatable prism bounding box.
	template <typename T=double>
	class Prism
	{
	public:
		Prism() {}
		Prism(const Point_t<T> &radii, const Point_t<T> &center, const Quat_t<T> quaternion = Quat_t<T> {1,0,0,0}) : _quaternion(quaternion), _center(center), _radii(radii) {}

		//return point in local normalized coordinates
		inline Point_t<T> tolocal(const Point_t<T> &point) const {return (_quaternion.rotate(point-_center))/_radii;}

		//convert point in local normalized coordinates to global coordinates
		inline Point_t<T> toglobal(const Point_t<T> &point) const {return _quaternion.conj().rotate(point*_radii)+_center;}

		//check if point in global coordinates is in the bounding box
		inline bool is_in_bbox(const Point_t<T> &point) const {return gv::util::norminf<3,T>(tolocal(point)) <= 1;}

		//evaluate level set function at specified point in global coordinates
		inline double eval_level_set(const Point_t<T> &point) const {return _eval_level_set(tolocal(point));}

		//check if point is inside particle
		bool contains(const Point_t<T> &point) const
		{
			Point_t<T> localpoint = tolocal(point);
			if (gv::util::norminf<3,T>(localpoint) <= 1) {return _eval_level_set(localpoint) <= 1;}
			return false;
		}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t<T> support(const Point_t<T> &direction) const
		{
			Point_t<T> rotated_direction = _quaternion.rotate(direction);
			Point_t<T> localpoint {1,1,1};
			for (int i=0; i<3; i++)
				{
					if (direction[i]<0)
					localpoint[i] = -1;
				}
			return toglobal(localpoint);
		}

	protected:
		//quaternion to rotate from global coordinate system to particle coordinate system
		Quat_t<T> _quaternion;

		//origin of particle coordinate system
		Point_t<T> _center;

		//major radii in particle coordinate system
		Point_t<T> _radii;

		//evaluate level set function at specified point in normalized local coordinates
		virtual double _eval_level_set(const Point_t<T> &localpoint) const {return gv::util::norminf<3,T>(localpoint);}
	};


	//ellipsoid class
	template <typename T=double>
	class Ellipsoid : public Prism<T>
	{
	public:
		Ellipsoid() : Prism<T>() {}
		Ellipsoid(const Point_t<T> &radii, const Point_t<T> &center, const Quat_t<T> quaternion = Quat_t<T> {1,0,0,0}) : Prism<T>(radii, center, quaternion) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t<T> support(const Point_t<T> &direction) const
		{
			Point_t<T> rotated_direction = this->_quaternion.rotate(direction);
			Point_t<T> localpoint = rotated_direction.normalized();
			return toglobal(localpoint);
		}

	protected:
		virtual double _eval_level_set(const Point_t<T> &localpoint) const {return localpoint.normSquared();}
	};


	//right-circular cyllinder class. height axis parallel to z-axis in local coordinates.
	template <typename T=double>
	class Cylinder : public Prism<T>
	{
	public:
		Cylinder() : Prism<T>() {}
		Cylinder(const Point_t<T> &radii, const Point_t<T> &center, const Quat_t<T> quaternion = Quat_t<T> {1,0,0,0}) : Prism<T>(radii, center, quaternion) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t<T> support(const Point_t<T> &direction) const
		{
			Point_t<T> rotated_direction = this->_quaternion.rotate(direction);
			Point_t<T> localpoint {0,0,1};
			if (rotated_direction[2] < 0) {localpoint[2] = -1;}
			
			T R = std::sqrt(rotated_direction[0]*rotated_direction[0] + rotated_direction[1]*rotated_direction[1]);
			if (R>0)
			{
				localpoint[0]/=R;
				localpoint[1]/=R;
			}

			return toglobal(localpoint);
		}

	protected:
		virtual double _eval_level_set(const Point_t<T> &localpoint) const {return gv::util::max(localpoint[0]*localpoint[0]+localpoint[1]*localpoint[1], gv::util::abs(localpoint[2]));}
	};


	//super-ellipsoid class.
	template <typename T=double>
	class SuperEllipsoid : public Prism<T>
	{
	public:
		SuperEllipsoid() : Prism<T>() {}
		SuperEllipsoid(const Point_t<T> &radii, T eps[2], const Point_t<T> &center, const Quat_t<T> quaternion = Quat_t<T> {1,0,0,0}) : \
					Prism<T>(radii, center, quaternion), _eps(eps), _powers({1.0/eps[0], 1.0/eps[1], eps[2]/eps[1]}), _invpowers({1.0/(2.0-eps[0]), 1.0/(2.0-eps[1])}) {}

		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t<T> support(const Point_t<T> &direction) const
		{
			Point_t<T> rotated_direction = this->_quaternion.rotate(direction);
			//get omega
			T x = sgn(rotated_direction[0])*std::pow(gv::util::abs(rotated_direction[0]), _invpowers[1]);
			T y = sgn(rotated_direction[1])*std::pow(gv::util::abs(rotated_direction[1]), _invpowers[1]);
			T omega = std::atan2(y, x); //in [-pi,pi]

			//get eta
			x = std::pow(gv::util::abs(rotated_direction[0]), _invpowers[0]);
			y = sgn(rotated_direction[2]) * std::pow( gv::util::abs( rotated_direction[2]*cos_pow(omega,2.0-_eps[1]) ) , _invpowers[0]);

			T eta = atan2(y, x); //in [-pi/2,pi/2] because x >= 0

			//get normal in global coordinates
			Point_t<T> localpoint = parametric(eta, omega);
			
			return toglobal(localpoint);
		}

	protected:
		//signed cos(theta)^eps
		static const T cos_pow(const T theta, const T eps)
		{
			T C = cos(theta);
			return sgn(C)*std::pow(gv::util::abs(C), eps);
		}
		//signed sin(theta)^eps
		static const T sin_pow(const T theta, const T eps)
		{
			T S = sin(theta);
			return sgn(S)*std::pow(gv::util::abs(S), eps);
		}


		//shape parameters
		const T _eps[2] {1,1};

		//common exponents
		const T _powers[3] {1,1,1};
		const T _invpowers[2] {1,1};

		//evaluate level set function
		virtual double _eval_level_set(const Point_t<T> &localpoint) const
		{
			T a = std::pow(localpoint[0]*localpoint[0], _powers[1]) + std::pow(localpoint[1]*localpoint[1], _powers[1]);
			return std::pow(a, _powers[2]) + std::pow(localpoint[2]*localpoint[2], _powers[0]);
		}

		//get point in local coordinates from the parametric representation of the particle surface
		Point_t<T> _parametric(const T eta, const T omega) const{
			//compute sines and cosines
			T C_eta = cos_pow(eta, _eps[0]);
			T S_eta = sin_pow(eta, _eps[0]);
			T C_omega = cos_pow(omega, _eps[1]);
			T S_omega = sin_pow(omega, _eps[1]);

			Point_t<T> localpoint = Point_t<T> {C_eta*C_omega, C_eta*S_omega, S_eta};
			return localpoint;
		}
	};



}