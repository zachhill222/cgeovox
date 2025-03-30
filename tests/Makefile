INC_DIR    = ../include
LIB_DIR    = #../lib

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS     = $(SRC:.cpp=)

CC         = g++ -fopenmp -g
CFLAGS     = -Wall -O2
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = #-lcgeovox


.PHONY: all
all: $(TARGETS)


# LINK
$(TARGETS) : % : %.o $(OBJ)
	$(CC) $< -o $@

# COMPILE
%.o: %.cpp
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)
	@echo TARGETS:= $(TARGETS)