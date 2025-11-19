INC_DIR    = -I../include -I/usr/include/eigen3
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS    = $(SRC:.cpp=)

#-g for gdb (http://www.gnu.org/software/gdb/documentation/), -pg for gprof-gui (https://www.gnu.org/software/gprofng-gui/)
#correctness testing
# CC         = g++ -std=c++20 -pg -g3 #-fopenmp#  -g for gdb, -pg for gprof (will create gmon.out)
# CFLAGS     = -Wall -O0

#perfomance testing
CC         = g++ -DNDEBUG -std=c++20 #-fopenmp #-g -pg
CFLAGS     = -O3

LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


# DEFAULT: make all test programs
all: $(TARGETS)


# RULES TO MAKE EACH TEST INDIVIDUALLY. listing individually helps autocomplete. using .PHONY forces re-build when header files have changed but not source files.
.PHONY: assembly charmsQ1 octree util charms_assembly matrix mesh

assembly:
	$(CC) $(CFLAGS) $(INC_DIR) assembly.cpp assembly

charmsQ1:
	$(CC) $(CFLAGS) $(INC_DIR) charmsQ1.cpp -o charmsQ1

octree:
	$(CC) $(CFLAGS) $(INC_DIR) octree.cpp -o octree

util:
	$(CC) $(CFLAGS) $(INC_DIR) util.cpp -o util

charms_assembly:
	$(CC) $(CFLAGS) $(INC_DIR) charms_assembly.cpp -o charms_assembly

matrix:
	$(CC) $(CFLAGS) $(INC_DIR) matrix.cpp -o matrix

mesh:
	$(CC) $(CFLAGS) $(INC_DIR) mesh.cpp -o mesh


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