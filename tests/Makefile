INC_DIR    = -I../include -I/usr/include/eigen3
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS    = $(SRC:.cpp=)

#-g for gdb (http://www.gnu.org/software/gdb/documentation/), -pg for gprof-gui (https://www.gnu.org/software/gprofng-gui/)
CC         = g++ -g -pg -fopenmp
CFLAGS     = -Wall -O2
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


# DEFAULT: make all test programs
all: $(TARGETS)


# RULES TO MAKE EACH TEST INDIVIDUALLY. listing individually helps autocomplete. using .PHONY forces re-build when header files have changed but not source files.
.PHONY: assembly
assembly :
	$(CC) $(CFLAGS) $(INC_DIR) -c assembly.cpp -o assembly.o
	$(CC) assembly.o -o assembly

.PHONY: charmsQ1
charmsQ1 :
	$(CC) $(CFLAGS) $(INC_DIR) -c charmsQ1.cpp -o charmsQ1.o
	$(CC) charmsQ1.o -o charmsQ1

.PHONY: octree
octree :
	$(CC) $(CFLAGS) $(INC_DIR) -c octree.cpp -o octree.o
	$(CC) octree.o -o octree

.PHONY: util
util :
	$(CC) $(CFLAGS) $(INC_DIR) -c util.cpp -o util.o
	$(CC) util.o -o util



.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)

.PHONY: remove_tests
remove_tests:
	rm -f gmon.out
	rm -rf *.er
	rm -f *.vtk
	rm -f ./outfiles/*

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)