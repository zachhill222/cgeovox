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
.PHONY: poisson_standard
poisson_standard :
	$(CC) $(CFLAGS) $(INC_DIR) -c poisson_standard.cpp -o poisson_standard.o
	$(CC) poisson_standard.o -o poisson_standard

.PHONY: poisson_charms
poisson_charms :
	$(CC) $(CFLAGS) $(INC_DIR) -c poisson_charms.cpp -o poisson_charms.o
	$(CC) poisson_charms.o -o poisson_charms

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)