INC_DIR    = -I../include -I/usr/include/eigen3
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS    = $(SRC:.cpp=)

#-g for gdb (http://www.gnu.org/software/gdb/documentation/), -pg for gprof-gui (https://www.gnu.org/software/gprofng-gui/)
#correctness testing
CC         = g++ -fsanitize=address -std=c++20 -fopenmp -pg -g #  -g for gdb, -pg for gprof (will create gmon.out)
CFLAGS     = -Wall -O0

#perfomance testing
# CC         = g++ -DNDEBUG -std=c++20 -fopenmp
# CFLAGS     = -Wall -O3

LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


# DEFAULT: make all test programs
all: $(TARGETS)


# RULES TO MAKE EACH TEST INDIVIDUALLY. listing individually helps autocomplete. using .PHONY forces re-build when header files have changed but not source files.
.PHONY: assembly charmsQ1 octree util charms_assembly matrix mesh

assembly:
	$(CC) $(CFLAGS) $(INC_DIR) -c assembly.cpp assembly

charmsQ1:
	$(CC) $(CFLAGS) $(INC_DIR) -c charmsQ1.cpp -o charmsQ1

octree:
	$(CC) $(CFLAGS) $(INC_DIR) -c octree.cpp -o octree

util:
	$(CC) $(CFLAGS) $(INC_DIR) -c util.cpp -o util

charms_assembly:
	$(CC) $(CFLAGS) $(INC_DIR) -c charms_assembly.cpp -o charms_assembly

matrix:
	$(CC) $(CFLAGS) $(INC_DIR) -c matrix.cpp -o matrix

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