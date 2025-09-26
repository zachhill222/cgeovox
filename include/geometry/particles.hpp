#pragma once

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "util/box.hpp"
#include "util/matrix.hpp"

#include <cmath>


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
			return Box_t {low, high};
		}

		//check if point is inside particle
		bool contains(const Point_t &point) const
		{
			// Point_t localpoint = tolocal(point);
			// if (gv::util::norminfty<3,double>(localpoint) <= 1) {return _eval_level_set(localpoint) <= 1;}
			// return false;

			return eval_level_set(point) <= 1;
		}

		//get the point on the surface that intersects a line segment from a point inside the particle to a point outside
		Point_t segment2surface(Point_t inside, Point_t outside) const
		{
			assert(contains(inside));
			assert(!contains(outside));

			Point_t current;
			for (int i=0; i<100; i++)
			{
				current = 0.5*(inside+outside);
				if (contains(current)) {inside=current;}
				else {outside=current;}

				if (gv::util::norminfty(outside-inside) < 1E-10) {break;}
			}

			return inside;
		}

		Point_t line2surface(Point_t p1, Point_t p2) const
		{
			double L1 = eval_level_set(p1);
			double L2 = eval_level_set(p2);

			if (L1>L2)
			{
				double tmp = L1;
				L1=L2; L2=tmp;

				Point_t TMP = p1;
				p1=p2; p1=TMP;
			}


			//now L1<=L2
			if (!contains(p2) and contains(p1)) {return segment2surface(p1,p2);}
			if (contains(p1) and contains(p2))
			{
				Point_t dir = gv::util::normalize(p2-p1);
				p2 = p1 + 2.0*gv::util::norminfty(_radii)*dir;
				assert(!contains(p2));
				return segment2surface(p1,p2);
			}

			//both outside. find closest point on the line to the center
			Point_t dir = p2-p1;
			double t = gv::util::dot(_center-p1,dir) / gv::util::squaredNorm(dir);
			Point_t inner = p1 + t*dir;
			if (!contains(inner)) {return p1;}
			return segment2surface(inner,p1);
		}

		//get the point on the surface closest to the specified point
		Point_t closest_point(const Point_t& globalpoint) const
		{
			// if (contains(globalpoint)) {return globalpoint;}

			// Point_t current = segment2surface(_center, globalpoint);
			// Point_t support_point = support(globalpoint - current);

			// // std::cout << "START:";
			// int i;
			// for (i=0; i<16; i++)
			// {
			// 	// std::cout << "\t" << gv::util::norm2(current-globalpoint) << std::flush;
			// 	Point_t next = segment2surface(0.5*current + 0.5*support_point, globalpoint);
			// 	if (gv::util::norminfty(current-next) < 1E-10) {break;}
			// 	support_point = support(globalpoint - current);
			// }

			// std::cout << std::endl;
			// std::cout << "END: " << grad(current)/(globalpoint-current) << std::endl;

			// return current;

			
			//get initial point on the surface
			Point_t current = line2surface(_center, globalpoint);
			// if (contains(globalpoint))
			// {
			// 	Point_t direction = globalpoint - _center;
			// 	current = globalpoint + 2.0 * gv::util::norminfty(_radii) * direction;
			// 	assert(!contains(current));
			// 	current = segment2surface(globalpoint,current);
			// }
			// else {current = segment2surface(_center,globalpoint);}


			//refine current guess
			int i;
			for (i=0; i<64; i++)
			{
				Point_t normal = gv::util::normalize(grad(current));
				Point_t ideal_direction = globalpoint - current;

				// std::cout << "i=" << i << std::endl;

				//get component of the ideal direction tangent to the surface
				Point_t normal_component = gv::util::dot(normal, ideal_direction) * normal;
				Point_t tangent_component = ideal_direction - normal_component;

				if (gv::util::norminfty(tangent_component) < 1E-10) {break;}
				current += tangent_component;
				current = line2surface(current, globalpoint);
			}

			// std::cout << "END: " << grad(current)/(globalpoint-current) << std::endl;
			//ensure that the gradient is parallel to the direction to the specified point
			Point_t constant = grad(current)/(globalpoint-current);
			constant -= gv::util::norminfty(constant) * Point_t{1,1,1};
			// std::cout << i << "\t" << gv::util::norminfty(constant) << std::endl;
			// assert(gv::util::norminfty(constant) < 1E-10);
			return current;
		}


		//get a supporting point of the supporting hyperplane in specified direction in global coordinates. this maximizes dot(x,direction) over x in the particle.
		virtual Point_t support(const Point_t &direction) const = 0;

		//compute and convert the gradient of the level set from local coordinates back to global
		Point_t grad(const Point_t& globalpoint) const
		{
			Point_t localgrad = _grad(tolocal(globalpoint));
			return _quaternion.conj().rotate(localgrad*_radii);
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
			Point_t rotated_direction = this->_quaternion.rotate(direction)/this->_radii;
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

		Point_t _grad(const Point_t &localpoint) const override
		{
			if (gv::util::squaredNorm(localpoint)==0) {return Point_t{0,0,0};}

			//get maximum component and indices
			double max_val = gv::util::norminfty(localpoint);
			Point_t result {0,0,0};

			//the level set only increases only when moving in the direction with largest coordinate value
			for (int i=0; i<3; i++) {
				if (gv::util::abs(localpoint[i])==max_val) {
					result[i] = gv::util::sgn(localpoint[i]);
				}
			}

			return result;
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
			Point_t rotated_direction = this->_quaternion.rotate(direction)/this->_radii;
			Point_t localpoint = gv::util::normalize(rotated_direction);
			return this->toglobal(localpoint);
		}

		

	protected:
		double _eval_level_set(const Point_t &localpoint) const override {return gv::util::squaredNorm(localpoint);}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			return 2.0*localpoint;
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
			Point_t rotated_direction = this->_quaternion.rotate(direction)/this->_radii;
			Point_t localpoint {0,0,1};
			if (rotated_direction[2] < 0) {localpoint[2] = -1;}

			double R = std::sqrt(rotated_direction[0]*rotated_direction[0] + rotated_direction[1]*rotated_direction[1]);
			if (R>0)
			{
				localpoint[0] = rotated_direction[0]/R;
				localpoint[1] = rotated_direction[1]/R;
			}

			return this->toglobal(localpoint);
		}

		

	protected:
		double _eval_level_set(const Point_t &localpoint) const override{return std::max(localpoint[0]*localpoint[0]+localpoint[1]*localpoint[1], gv::util::abs(localpoint[2]));}

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			Point_t result {0,0,0};
			
			double R2 = localpoint[0]*localpoint[0] + localpoint[1]*localpoint[1];
			double H  = gv::util::abs(localpoint[2]);

			if (R2>=H)
			{
				result[0] = 2.0*localpoint[0];
				result[1] = 2.0*localpoint[0];
			}

			if (H>=R2)
			{
				result[2] = gv::util::sgn(localpoint[2]);
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
			Point_t rotated_direction = this->_quaternion.rotate(direction)/this->_radii;
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

		

		//evaluate the hessian in local coordinates
		Hessian_t hess(const Point_t& localpoint) const
		{
			//various constants
			double A = 2.0*(_eps1-_eps0)/(_eps1*_eps0);
			double B = (2.0-_eps1)/_eps1;
			double C = 2.0/_eps0;
			double a  = std::pow(localpoint[0]*localpoint[0], 1.0/_eps1) + std::pow(localpoint[1]*localpoint[1], 1.0/_eps1);
			double ETA = std::pow(a, (_eps1-_eps0)/_eps0);
			double ETA_XY_TEMP =  A*std::pow(a, (_eps1-2.0*_eps0)/_eps0);
			double ETA_X = ETA_XY_TEMP * std::pow(gv::util::abs(localpoint[0]), B) * gv::util::sgn(localpoint[0]);
			double ETA_Y = ETA_XY_TEMP * std::pow(gv::util::abs(localpoint[1]), B) * gv::util::sgn(localpoint[1]);

			//assemble hessian
			Hessian_t H; //all zeros
			H(0,0) = C * ETA_X * std::pow(gv::util::abs(localpoint[0]), B) * gv::util::sgn(localpoint[0]) + ETA * B * std::pow(localpoint[0]*localpoint[0], (1.0-_eps1)/_eps1);
			H(1,1) = C * ETA_Y * std::pow(gv::util::abs(localpoint[1]), B) * gv::util::sgn(localpoint[1]) + ETA * B * std::pow(localpoint[1]*localpoint[1], (1.0-_eps1)/_eps1);
			H(0,1) = C * ETA_Y * std::pow(gv::util::abs(localpoint[0]), B) * gv::util::sgn(localpoint[0]);
			H(1,0) = C * ETA_X * std::pow(gv::util::abs(localpoint[1]), B) * gv::util::sgn(localpoint[1]);
			
			H(2,2) = C * (C-1.0) * std::pow(localpoint[2]*localpoint[2], (1.0-_eps0)/_eps0);

			// assert(H(0,1)==H(1,0));
			if (H(0,1)!=H(1,0)) {std::cout << "hessian at " << localpoint << "\n" << H << std::endl;}

			return H;
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

		//evaluate the gradient in local coordinates
		Point_t _grad(const Point_t &localpoint) const override
		{
			double C  = 2.0/_eps0;
			double a  = std::pow(localpoint[0]*localpoint[0], 1.0/_eps1) + std::pow(localpoint[1]*localpoint[1], 1.0/_eps1);
			double ETA = std::pow(a, (_eps1-_eps0)/_eps0);
			Point_t result {C*ETA*std::pow(gv::util::abs(localpoint[0]), (2.0-_eps1)/_eps1),
							C*ETA*std::pow(gv::util::abs(localpoint[1]), (2.0-_eps1)/_eps1),
							C*std::pow(gv::util::abs(localpoint[2]), (2.0-_eps1)/_eps1)
						};

			for (int i=0; i<3; i++) {result[i] *= gv::util::sgn(localpoint[i]);} //correct for taking the absolute value before evaluating the exponential

			return result;
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