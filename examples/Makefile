INC_DIR    = -I../include -I/usr/include/eigen3
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS    = $(SRC:.cpp=)

CC         = g++ -fopenmp -g
CFLAGS     = -Wall -O2
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


# DEFAULT: make all test programs
all: $(TARGETS)


# RULES TO MAKE EACH TEST INDIVIDUALLY. listing individually helps autocomplete. using .PHONY forces re-build when header files have changed but not source files.
.PHONY: geometry_conforming_poisson
geometry_conforming_poisson :
	$(CC) $(CFLAGS) $(INC_DIR) -c geometry_conforming_poisson.cpp -o geometry_conforming_poisson.o
	$(CC) geometry_conforming_poisson.o -o geometry_conforming_poisson

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)