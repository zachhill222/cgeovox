INC_DIR    = -I../include -I/usr/include/eigen3

#get all .cpp files for make all
SRC        = $(wildcard *.cpp)
TARGETS    = $(SRC:.cpp=)

#default build (debug)
CXX        = g++ -std=c++20
OPT        = -O0
PAR        =
DEBUG      = -pg -g3 -Wall

#redefine for testing optimizations
OPT        = -O3
PAR        = -fopenmp
# DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

# DEFAULT: make all test programs
all: $(TARGETS)

# rules to make tests individually. listing individually helps autocomplete.
# using .PHONY forces re-build when header files have changed but not source files.
.PHONY: assembly charmsQ1 octree util charms_assembly matrix mesh fixedpoint

assembly:
	$(CC) $(INC_DIR) assembly.cpp assembly

charmsQ1:
	$(CC) $(INC_DIR) charmsQ1.cpp -o charmsQ1

octree:
	$(CC) $(INC_DIR) octree.cpp -o octree

util:
	$(CC) $(INC_DIR) util.cpp -o util

charms_assembly:
	$(CC) $(INC_DIR) charms_assembly.cpp -o charms_assembly

matrix:
	$(CC) $(INC_DIR) matrix.cpp -o matrix

mesh:
	$(CC) $(INC_DIR) mesh.cpp -o mesh

fixedpoint:
	$(CC) $(INC_DIR) fixedpoint.cpp -o fixedpoint

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o

.PHONY: remove_tests
remove_tests:
	rm -f gmon.out
	rm -rf *.er
	rm -f *.vtk
	rm -f ./outfiles/*
	rm -f massif.*

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)