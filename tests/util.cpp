#include "util/point.hpp"
#include "util/box.hpp"
#include "util/quaternion.hpp"

#include <iostream>
#include <iomanip>
#include <cstring>
#include <typeinfo>
#include <cstdlib>

std::string print_pass(const bool pass)
{
	if (pass) {return "(PASS)";}
	return "(FAIL)";
}


template<int dim, typename Scalar_t>
bool point(const int verbose=1)
{
	bool allpass = true;
	int testno=0;
	using Point_t = gv::util::Point<dim,Scalar_t>;

	//print test type
	if (verbose>0)
	{
		std::string mangled_name = typeid(Point_t).name();
		std::cout << "\n===== Test: Point_t=" << std::flush;
		bool flag = std::system(("c++filt -t " + mangled_name).data()); assert(flag==0);
	}

	//test default constructor
	{
		testno++;
		bool pass = true;
		
		Point_t pt{};
		for (int i=0; i<dim; i++) {if (pt[i]!=0) {pass=false;}}
		allpass = allpass and pass;

		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tdefault constructor {}: " << pt;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test constant constructor
	{
		testno++;
		bool pass = true;

		Point_t ones(1);
		for (int i=0; i<dim; i++) {if (ones[i]!=1) {pass=false;}}

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tconstant contructor (1): " << ones;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test copy constructor
	{
		testno++;
		bool pass = true;

		Point_t ones(1);
		const Point_t first(1);
		Point_t second(first);
		pass = (first==ones) and (second==ones);

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tcopy constructor (const Point_t): " << second;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test move constructor
	{
		testno++;
		bool pass = true;

		Point_t ones(1);
		Point_t first(1);
		Point_t second(std::move(first));
		pass = (second==ones);

		//ensure that the data was moved and not copied
		try {
			first.at(0);
			pass=false;
		}
		catch (const std::runtime_error &e)
		{
			if (strcmp(e.what(), "DATA_MOVED") != 0) {pass = false;}
		}
		catch (...) {pass=false;}

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tmove constructor (std::move(Point_t)): " << second;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test copy assignment
	{
		testno++;
		bool pass = true;

		Point_t ones(1);
		const Point_t first(1);
		Point_t second = first;
		pass = (first==ones) and (second==ones);
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tcopy assignment (Point_t = const Point_t): " << second;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test move assignment
	{
		testno++;
		bool pass = true;

		Point_t ones(1);
		Point_t first(1);
		Point_t second = std::move(first);
		pass = (second==ones);

		//ensure that the data was moved and not copied
		try {
			first.at(0);
			pass=false;
		}
		catch (const std::runtime_error &e)
		{
			if (strcmp(e.what(), "DATA_MOVED") != 0) {pass = false;}
		}
		catch (...) {pass=false;}

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tmove assignment (Point_t = std::move(Point_t)): " << second;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test assignment
	{
		testno++;
		bool pass = true;

		Point_t pt;
		for (int i=0; i<dim; i++) {pt[i] = (Scalar_t) i; if (pt[i]!= (Scalar_t) i) {pass=false;}}
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tcomponent assignment: " << pt;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test vector addition
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;

		for (int i=0; i<dim; i++)
		{
			Scalar_t L = i+10;
			Scalar_t R = i+9;
			left[i]    = L;
			right[i]   = R;
			result[i]  = L+R;
		}
		pass = (left+right)==result;

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\taddition: Point_t{" << left << "} + Point_t{" << right << "} = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test vector subtraction
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;

		for (int i=0; i<dim; i++)
		{
			Scalar_t L = i+10;
			Scalar_t R = i+9;
			left[i]    = L;
			right[i]   = R;
			result[i]  = L-R;
		}
		pass = (left-right)==result;

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tsubtraction: Point_t{" << left << "} - Point_t{" << right << "} = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test scalar multiplication
	{
		testno++;
		bool pass = true;

		Scalar_t left = 2;
		Point_t right;
		Point_t result;

		for (int i=0; i<dim; i++)
		{
			Scalar_t R = i-1;
			right[i]   = R;
			result[i]  = left*R;
		}
		pass = (left*right)==result;

		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tscalar multiplication: " << left << " * Point_t{" << right << "} = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test component-wise multiplication
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;

		for (int i=0; i<dim; i++)
		{
			Scalar_t L = i+1;
			Scalar_t R = i-1;
			left[i]    = L;
			right[i]   = R;
			result[i]  = L*R;
		}
		pass = (left*right)==result;
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tcomponent-wise multiplication: Point_t{" << left << "} * Point_t{" << right << "} = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}

	}

	//test component-wise division
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;

		for (int i=0; i<dim; i++)
		{
			Scalar_t L = i+10;
			Scalar_t R = i+1;
			left[i]    = L;
			right[i]   = R;
			result[i]  = L/R;
		}
		pass = (left/right)==result;
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tcomponent-wise division: Point_t{" << left << "} / Point_t{" << right << "} = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}

	}

	//test squaredNorm
	{
		testno++;
		bool pass = true;

		Point_t pt;
		Scalar_t norm2 = 0;
		for (int i=0; i<dim; i++)
		{
			Scalar_t val = i;
			pt[i] = val;
			norm2 += val*val;
		}
		pass = gv::util::squaredNorm(pt) == norm2;
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tEuclidean norm (squared): gv::util::squaredNorm(Point_t{" << pt << "}) = " << norm2;}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test elmax
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;
		for (int i=0; i<dim; i++)
		{
			Scalar_t L = 1-i;
			Scalar_t R = i;
			left[i]    = L;
			right[i]   = R;
			result[i]  = std::max(L,R);
		}
		pass = gv::util::elmax(left,right) == result;
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tget element-wise maximum: gv::util::elmax(Point_t{" << left << "}, Point_t{" << right << "}) = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}
	}

	//test elmin
	{
		testno++;
		bool pass = true;

		Point_t left;
		Point_t right;
		Point_t result;
		for (int i=0; i<dim; i++)
		{
			Scalar_t L = 1-i;
			Scalar_t R = i;
			left[i]    = L;
			right[i]   = R;
			result[i]  = std::min(L,R);
		}
		pass = gv::util::elmin(left,right) == result;
		
		allpass = allpass and pass;
		if (verbose>0) {std::cout << std::setw(2) << testno << ": " << print_pass(pass);}
		if (verbose>1) {std::cout << "\tget element-wise maximum: gv::util::elmin(Point_t{" << left << "}, Point_t{" << right << "}) = Point_t{" << result << "}";}
		if (verbose>0) {std::cout << std::endl;}
	}

	if (verbose>0) {std::cout << "===== End " << print_pass(allpass) << std::endl;}
	return allpass;
}

template<int dim, typename Scalar_t>
bool box(const int verbose)
{
	return true;
}


bool quaternion()
{
	return true;
}









int main(int argc, char* argv[])
{
	//get verbosity (verbose=-1: no print, verbose=0: no print from individual tests)
	int verbose = -1;
	if (argc>1) {verbose = atoi(argv[1]);}

	//track total and success
	int pass=0;
	int total=0;

	//test point
	{
		int _pass=0;
		int _total=0;

		//comment out lines to disable individual tests
		if (point<2,double>(verbose)) {_pass++;} _total++;
		if (point<3,double>(verbose)) {_pass++;} _total++;
		if (point<2,size_t>(verbose)) {_pass++;} _total++;
		if (point<3,size_t>(verbose)) {_pass++;} _total++;
		if (point<2,float>(verbose)) {_pass++;} _total++;
		if (point<3,float>(verbose)) {_pass++;} _total++;

		//increment "module" counters
		total++;
		if (_pass==_total) {pass++;}

		//print "module" result
		if (verbose==0) {std::cout << std::setw(2) << total << ": point.h (" << _pass << "/" << _total << ")" << std::endl;}
	}

	std::cout << "PASS=" << pass << " FAIL=" << total-pass << std::endl;
	return total-pass;
}