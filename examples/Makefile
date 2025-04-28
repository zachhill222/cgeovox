INC_DIR    = -I/usr/include/eigen3 -I../include
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS     = $(SRC:.cpp=)

CC         = g++ -fopenmp -g
CFLAGS     = -Wall -O2
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


.PHONY: all
all: $(TARGETS)


# LINK
$(TARGETS) : % : %.o $(OBJ)
	$(CC) $< -o $@ 

# COMPILE
%.o: %.cpp
	$(CC) $(CFLAGS) $(INC_DIR) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)

.PHONY: print
print:
	@echo INC_DIR:= $(INC_DIR)
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)