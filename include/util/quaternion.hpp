#pragma once

#include "util/point.hpp"
#include "ostream"
#include <cmath>


namespace gv::util{
	class Quaternion{
	public:
		//// INITIALIZERS
		Quaternion(): _q0(1), _qv(Point3 {0,0,0}) {}
		Quaternion(const double q0, const double q1, const double q2, const double q3): _q0(q0), _qv(Point3 {q1,q2,q3} ) {}
		Quaternion(const double q0, const Point3 qv): _q0(q0), _qv(Point3 {qv[0], qv[1], qv[2]}) {}

		//// ATTRIBUTES
		double operator[](int idx) const;
		double& operator[](int idx);

		double q0() const;
		Point3 qv() const;

		//// ROTATIONS
		Quaternion conj() const;
		Quaternion inv() const;
		double squaredNorm() const;
		double norm() const;
		Quaternion* normalize(); //normalize this quaternion to a rotation quaternion
		Quaternion* setrotation(const double& theta, const Point3& axis);
		Point3 rotate(const Point3& point) const; //rotate point

		///// ARITHMETIC
		Quaternion* operator+=(const Quaternion& other);
		Quaternion operator+(const Quaternion& other) const;
		Quaternion* operator-=(const Quaternion& other);
		Quaternion operator-(const Quaternion& other) const;
		Quaternion* operator*=(const Quaternion& other);
		Quaternion operator*(const Quaternion& other) const;
		Quaternion* operator/=(const Quaternion& other);
		Quaternion operator/(const Quaternion& other) const;
		bool operator==(const Quaternion& other) const;
		inline bool operator!=(const Quaternion& other) const { return !(operator==(other));}

	private:
		double _q0;
		Point3 _qv;
	};


	//// IMPLEMENTATION
	double Quaternion::operator[](int idx) const{
		if (idx==0){
			return _q0;
		}
		return _qv[idx-1];
	}

	double& Quaternion::operator[](int idx){
		if (idx==0){
			return _q0;
		}
		return _qv[idx-1];
	}

	double Quaternion::q0() const {return _q0;}

	Point3 Quaternion::qv() const {return _qv;}

	Quaternion Quaternion::conj() const{
		return Quaternion(_q0, -_qv);
	}
	Quaternion Quaternion::inv() const{
		double C = 1.0/squaredNorm();
		return Quaternion(C*_q0, (-C)*_qv);
	}
	double Quaternion::squaredNorm() const{
		return _q0*_q0 + _qv.squaredNorm();
	}
	double Quaternion::norm() const{
		return std::sqrt(squaredNorm());
	}

	Quaternion* Quaternion::normalize(){
		double C = 1.0/norm();
		_q0*=C;
		_qv*=C;
		return this;
	}
	Quaternion* Quaternion::setrotation(const double& theta, const Point3& axis){
		_q0 = std::cos(0.5*theta);
		_qv = std::sin(0.5*theta)*axis.normalized();
		return this;
	}
	Point3 Quaternion::rotate(const Point3& point) const {
		Quaternion V = Quaternion(0.0, point);
		V = operator*((V*conj()));
		return V.qv();
	}

	///// ARITHMETIC
	Quaternion* Quaternion::operator+=(const Quaternion& other){
		_q0+=other.q0();
		_qv+=other.qv();
		return this;
	}
	Quaternion Quaternion::operator+(const Quaternion& other) const{
		return Quaternion(_q0+other.q0(), _qv+other.qv());
	}
	Quaternion* Quaternion::operator-=(const Quaternion& other){
		_q0-=other.q0();
		_qv-=other.qv();
		return this;
	}
	Quaternion Quaternion::operator-(const Quaternion& other) const{
		return Quaternion(_q0-other.q0(), _qv-other.qv());
	}
	Quaternion* Quaternion::operator*=(const Quaternion& other){
		double Q0 = _q0*other.q0() - dot(_qv,other.qv());
		_qv = _q0*other.qv() + other.q0()*_qv + cross(_qv,other.qv());
		_q0 = Q0;
		return this;
	}
	Quaternion Quaternion::operator*(const Quaternion& other) const{
		double Q0 = _q0*other.q0() - dot(_qv,other.qv());
		Point3  QV = _q0*other.qv() + other.q0()*_qv + cross(_qv,other.qv());
		return Quaternion(Q0, QV);
	}
	Quaternion* Quaternion::operator/=(const Quaternion& other){
		operator*=(other.inv());
		return this;
	}
	Quaternion Quaternion::operator/(const Quaternion& other) const{
		return operator*(other.inv());
	}
	
	bool Quaternion::operator==(const Quaternion& other) const{
		if (_q0 != other.q0()){
			return false;
		}

		if (_qv != other.qv()){
			return false;
		}

		return true;
	}


	///Print to ostream.
	std::ostream& operator<<(std::ostream& os, const Quaternion &quaternion){
		for (int i = 0; i < 4; i++){
			os << quaternion[i] << " ";
		}
		return os;
	}
}

