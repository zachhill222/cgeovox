INC_DIR    = -I../include -I/usr/include/eigen3 #-I/usr/include/x86_64-linux-gnu/qt5
LIB_DIR    = 

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)
TARGETS     = $(SRC:.cpp=)

CC         = g++ -fopenmp -g
CFLAGS     = -Wall -O3 # -fPIC
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = 


.PHONY: all
all: $(TARGETS)


# LINK
$(TARGETS) : % : %.o $(OBJ)
	$(CC) $< -o $@ #-lQt5Gui -lQt5Core

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