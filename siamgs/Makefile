INC_DIR       = -I../include -I/usr/include/eigen3

CXX            = g++ -std=c++20 -fopenmp
CXXFLAGS_BASE = -Wall -O3 -DNDEBUG

standard: CXXFLAGS = $(CXXFLAGS_BASE) -DSTANDARD
standard: PROGRAM_NAME = standard
standard: assembly_program

interface: CXXFLAGS = $(CXXFLAGS_BASE) -DINTERFACE
interface: PROGRAM_NAME = interface
interface: assembly_program

convergence: CXXFLAGS = $(CXXFLAGS_BASE)
convergence: PROGRAM_NAME = convergence
convergence: convergence_program


assembly_program: poisson_charms.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIR) poisson_charms.cpp -o $(PROGRAM_NAME)

convergence_program: poisson_charms_convergence.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIR) poisson_charms_convergence.cpp -o $(PROGRAM_NAME)

.PHONY: clean
clean:
	rm -f standard interface convergence *.o