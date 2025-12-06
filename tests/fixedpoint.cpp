#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>

#include "util/scalars/fixed_point.hpp"


using FP = gv::util::FixedPoint<int64_t,0>;
using FI = typename FP::Conversion_t;

void print_constants()
{
	std::cout << "\n==================== COMPILE-TIME CONSTANTS ====================" << std::endl;
	std::cout << "EPSILON   = " << FP::EPSILON   << std::endl;
	std::cout << "MAX_FLOAT = " << FP::MAX_FLOAT << std::endl;
	std::cout << "SCALE     = " << FP::SCALE     << std::endl;
	std::cout << "INV_SCALE = " << FP::INV_SCALE << std::endl;
}

void check_addition(std::vector<typename FP::Float_t> values)
{
	std::cout << "\n=========================== ADDITION ===========================" << std::endl;
	auto opp = [](typename FP::Float_t a, typename FP::Float_t b) {
		//floating point arithmetic
		typename FP::Float_t c = a+b;

		//fixed point arithmetic
		FP A(a), B(b);
		FP C = A+B;
		
		std::cout << std::string(FI::BITS + 10, '-');
		std::cout << "\nFixedPoint";
		std::cout << "\n " << A.to_string() << " (" << std::setprecision(20) << A << ")" 
				  << "\n+" << B.to_string() << " (" << std::setprecision(20) << B << ")" 
				  << "\n=" << C.to_string() << " (" << std::setprecision(20) << C << ")"
				  << "\n";

		std::cout << "\nFloat";
		std::cout << "\n " << FI(a).to_string() << " (" << std::setprecision(20) << a << ")" 
				  << "\n+" << FI(b).to_string() << " (" << std::setprecision(20) << b << ")" 
				  << "\n=" << FI(c).to_string() << " (" << std::setprecision(20) << c << ")"
				  << "\n";


		std::cout << "\n|FixedPoint - Float|= " << std::setprecision(10) << std::fabs(static_cast<typename FP::Float_t>(C) - c) << ")\n";

		A+=B;
		if (A!=C) {std::cout << " IN-PLACE IS INCONSISTENT";}
		std::cout << "\n";
	};

	for (size_t i=0; i<values.size(); i++) {opp(values[i], values[values.size()-i-1]);}
}

void check_subtraction()
{
	std::cout << "\n========================= SUBTRACTION =========================" << std::endl;
	auto opp = [](typename FP::Float_t a, typename FP::Float_t b) {
		FP A(a), B(b);
		FP C = A-B;
		typename FP::Float_t c = a-b;
		
		std::cout << std::right 
				  << std::setw(15) << std::setprecision(5) << A
				  << " - " 
				  << std::setw(15) << std::setprecision(5) << B 
				  << " = " 
				  << std::setw(15) << std::setprecision(5) << C;

		std::cout << "\t (diff= " << std::setprecision(5) << std::fabs(static_cast<typename FP::Float_t>(C) - c) << ")";

		A-=B;
		if (A!=C) {std::cout << " IN-PLACE GAVE: " << A;}
		std::cout << "\n";
	};
	opp(10.3, 20.2);
	opp(-0.25, 0.3);
	opp(0.25, 0.301);
	opp(0, 0.00001);
	opp(1e4,1e-5);
}


int main() {
	std::vector<typename FP::Float_t> values;
	values.push_back(1);
	values.push_back(-1);
	values.push_back(1.0001);
	values.push_back(10.3);
	values.push_back(20.2);
	values.push_back(0.0005);
	values.push_back(10000);
	values.push_back(1000.0000001);
	values.push_back(-0.999999);

	print_constants();
	check_addition(values);
	check_subtraction();

	return 0;
}