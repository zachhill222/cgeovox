#pragma once

#include "util/polytope.hpp"
#include "util/point.hpp"
#include "util/plane.hpp"

#define GJK_DBL_TOL 1E-12
#define MAX_GJK_ITERATIONS 16

template <typename T>
using Point3_t = gv::util::Point<3,T>;

template <typename T>
using Polytope3_t = gv::util::Polytope<3,T>;

template <typename T>
using Plane_t = gv::util::Plane<T>;


namespace gv::geometry{


//SA and SB are two classes that represent convex shapes with a methods:
//	Point support(const Point& direction)
//	Point center()

// FOR DETAILS, SEE:
// “Implementing GJK - 2006”
// by Casey Muratori
// and 
// https://cs.brown.edu/courses/cs195u/lectures/04_advancedCollisionsAndPhysics.pdf


//SUPPORT FUNCTION IN MINKOWSKI DIFFERENCE
template<class SA, class SB, typename T=double>
Point3_t<T> support(const SA& S1, const SB& S2, const Point3_t<T>& direction)
{
	return S1.support(direction) - S2.support(-direction);
}

//LINE CASE
template <typename T=double>
bool lineCase(Polytope3_t<T>& simplex, Point3_t<T>& direction);

//TRIANGLE CASE
template <typename T=double>
bool triangleCase(Polytope3_t<T>& simplex, Point3_t<T>& direction);

//FULL SIMPLEX (TETRAHEDRON) CASE
template <typename T=double>
bool tetraCase(Polytope3_t<T>& simplex, Point3_t<T>& direction);

//WRAPPER FUNCTION FOR SPECIAL CASES
template <typename T=double>
bool doSimplex(Polytope3_t<T>& simplex, Point3_t<T>& direction);



//GJK IMPLEMENTATION
template<class SA, class SB, typename T=double> 
bool collides_GJK(const SA& S1, const SB& S2)
{
	Point3_t<T> direction = S1.center() - S2.center();
	Point3_t<T> A = support(S1,S2,direction);
	
	Polytope3_t<T> simplex {A};
	direction = -simplex[0];

	//MAIN LOOP
	for (int i=0; i<MAX_GJK_ITERATIONS; i++){
		A = support(S1,S2,direction);

		if (gv::util::dot(A,direction) < 0){
			return false;
		}

		simplex.push_back(A);

		if (doSimplex(simplex, direction)) {
			return true;
		}
	}

	// std::cout << "GJK failed to converge in " << MAX_GJK_ITERATIONS << " iterations\n";
	return true; //failed to converge, return collision to be safe.
}


//LINE CASE IMPLEMENTATION
template <typename T>
bool lineCase(Polytope3_t<T>& simplex, Point3_t<T>& direction)
{
	Point3_t<T> &A = simplex[1]; //most recent point
	Point3_t<T> &B = simplex[0];

	Point3_t<T> AO = -A;
	Point3_t<T> AB = B-A;
	
	double DOT;

	DOT = gv::util::dot(AB,AO);
	if (DOT>0.0)
	{
		// std::cout << "AB\n";
		direction = gv::util::cross(AB, gv::util::cross(AO,AB));

		//check if line segment contained the origin. AB and AO are co-linear.
		if (gv::util::squaredNorm(direction) <= GJK_DBL_TOL)
		{
			return true;
		}
		// simplex = Polytope3_t<T>({B, A}); //no change to simplex
	}
	else
	{
		direction = AO;
		simplex = Polytope3_t<T>({A});
	}
	return false;
}


//TRIANGLE CASE IMPLEMENTATION
template <typename T>
bool triangleCase(Polytope3_t<T>& simplex, Point3_t<T>& direction){
	Point3_t<T> &A = simplex[2]; //most recent point
	Point3_t<T> &B = simplex[1];
	Point3_t<T> &C = simplex[0];

	Point3_t<T> AO  = -A;
	Point3_t<T> AB  = B-A;
	Point3_t<T> AC  = C-A;

	Point3_t<T> ABC_normal = gv::util::cross(AB,AC); //normal to triangle
	Point3_t<T> AB_normal  = gv::util::cross(AB,ABC_normal); //away from triangle, normal to edge AB, in triangle plane
	Point3_t<T> AC_normal  = gv::util::cross(ABC_normal,AC); //away from triangle, normal to edge AB, in triangle plane

	double DOT;

	DOT = gv::util::dot(AC_normal,AO);
	if (DOT>0.0){
		DOT = gv::util::dot(AC,AO);
		if (DOT>0.0)
		{
			direction = gv::util::cross(AC, gv::util::cross(AO,AC));
			simplex = Polytope3_t<T>({C,A});
		}
		else{
			DOT = gv::util::dot(AB,AO);
			if (DOT>0.0)
			{ //STAR
				direction = gv::util::cross(AB, gv::util::cross(AO,AB));
				simplex = Polytope3_t<T>({B, A});
			}
			else
			{
				direction = AO;
				simplex = Polytope3_t<T>({A});
			}
		}
	}
	else{
		DOT = gv::util::dot(AB_normal,AO);
		if (DOT>0.0){
			DOT = gv::util::dot(AB,AO);
			if (DOT>0.0)
			{ //STAR
				direction = gv::util::cross(AB, gv::util::cross(AO,AB));
				simplex = Polytope3_t<T>({B,A});
			}
			else
			{
				direction = AO;
				simplex = Polytope3_t<T>({A});
			}
		}
		else
		{
			DOT = gv::util::dot(ABC_normal,AO);
			//above, below, or on triangle
			if (DOT>GJK_DBL_TOL )
			{
				direction = ABC_normal;
				// simplex = Polytope3_t<T>({C,B,A}); //no change to simplex
			}
			else if (DOT<-GJK_DBL_TOL)
			{
				direction = -ABC_normal;
				simplex = Polytope3_t<T>({B, C, A}); //orientation matters
			}
			else {return true;}
		}
	}

	return false; //triangle PROBABLY doesn't contain the origin
}


//FULL SIMPLEX (TETRAHEDRON) CASE IMPLEMENTATION
template <typename T>
bool tetraCase(Polytope3_t<T>& simplex, Point3_t<T>& direction)
{
	Point3_t<T> &A = simplex[3]; //most recent point
	Point3_t<T> &B = simplex[2];
	Point3_t<T> &C = simplex[1];
	Point3_t<T> &D = simplex[0];

	Point3_t<T> O = Point3_t<T> {0,0,0};

	Plane_t<T> P;
	T abc, adc, abd;

	//get distance to each plane, we know the orighin is in the negative side of the plane BCD because A is the most recent point
	P   = Plane_t<T>(A,B,C); //normal faces out of tetrahedron
	abc = P.dist(O);

	P   = Plane_t<T>(A,D,C); //normal faces out of tetrahedron
	adc = P.dist(O);

	P   = Plane_t<T>(A,B,D); //normal faces out of tetrahedron
	abd = P.dist(O);

	// if all distances are negative, origin is in side the tetrahedron
	T max_dist = std::max(abc,std::max(adc,abd));
	// std::cout << "max_dist= " << max_dist << std::endl;
	if (max_dist<0.0) {return true;}


	// reduce to triangle case
	T abc_dist = abs(abc);
	T adc_dist = abs(adc);
	T abd_dist = abs(abd);
	T min_dist = std::min(abc_dist, std::min(adc_dist, abd_dist));
	// std::cout << "min_dist= " << min_dist << std::endl;

	if (abc_dist == min_dist)
	{
		// std::cout << "Plane BCA\n";
		simplex = Polytope3_t<T>({B, C, A});
	}
	else if(adc_dist == min_dist)
	{
		// std::cout << "Plane CDA\n";
		simplex = Polytope3_t<T>({C, D, A});
	}
	else
	{
		// std::cout << "Plane DBA\n";
		simplex = Polytope3_t<T>({D, B, A});
	}
	

	// run triangle case
	return triangleCase(simplex, direction);
}


//WRAPPER IMPLEMENATION
template <typename T>
bool doSimplex(Polytope3_t<T>& simplex, Point3_t<T>& direction)
{
	//simplex must contain between 2 and 4 points initially
	//simplex and direction will both be updated for the next iteration

	bool result = false;
	
	//GET NEW SEARCH DIRECTION
	switch (simplex.len()){
	case 2:
		// std::cout << "LINE CASE\n";
		result = lineCase(simplex, direction);
		break;
	case 3:
		// std::cout << "TRIANGLE CASE\n";
		result = triangleCase(simplex, direction);
		break;
	case 4:
		// std::cout << "TETRAHEDRAL CASE\n";
		result = tetraCase(simplex, direction);
		break;
	}

	return result;
}
}
