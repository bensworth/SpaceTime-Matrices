# Compiler, compile options, and link options
LIB_DIR   = /Users/ben/Documents/SoftwareLibraries
MFEM_DIR  = $(LIB_DIR)/mfem
HYPRE_DIR = $(LIB_DIR)/hypre/src/hypre
METIS_DIR = $(LIB_DIR)/metis/build/Darwin-x86_64
BRAID_DIR = $(LIB_DIR)/xbraid
CXX      = mpic++
FLAGS    = -g
INC      = -I$(MFEM_DIR)/bens_build -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -I$(BRAID_DIR)
CXXFLAGS = $(FLAGS) $(INC)
LDFLAGS  = -L$(MFEM_DIR)/bens_build -L$(HYPRE_DIR)/lib -L$(METIS_DIR)/libmetis -L$(BRAID_DIR) -L/System/Library/Frameworks/Accelerate.framework
LDLIBS   = -framework Accelerate -lmfem -lHYPRE -lmetis -lcblas -lbraid


BRAID_FLAGS = -I$(BRAID_DIR)
BRAID_LIB_FILE = 


# Files to compile
SOURCE_FILES  = 
OBJECT_FILES  = $(SOURCE_FILES:.cpp=.o)
EXEC_FILES    = drive-adv-diff-DG

.PHONY: all clean clean-build clean-exec

# Compile source files if the hpp or cpp has changed.
%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Recompile executables if cpp files or object dependencies have changed
%: %.cpp $(OBJECT_FILES)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OBJECT_FILES) $(LDFLAGS) $(LDLIBS)

# Specify that 'make' compiles all examples.
all: $(OBJECT_FILES) $(EXEC_FILES)

# Remove all object files and executables
clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(OBJECT_FILES) $(EXEC_FILES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh displaced.mesh mesh.*

