EXEC_NAME=simulation
NVFLAGS= -arch sm_50

CUDA_DIR=/usr/local/cuda
CUDA_BIN=$(CUDA_DIR)/bin
LIBDIR=$(CUDA_DIR)/lib64
INCDIR=$(CUDA_DIR)/include
NVCC = $(CUDA_BIN)/nvcc
CC = g++-6

# Folders
SRC=src
BIN=bin

# Files
SRC_FILES=\
	simulation.cu

EXEC_FILES = $(EXEC_NAME:%=$(BIN)/%)
EXEC_FILES_D = $(EXEC_NAME:%=$(BIN)/%)
OBJECT_FILES = $(SOURCE_FILES:%.cpp=$(OBJ)/%.o)

build: $(EXEC_FILES)

build-debug: $(EXEC_FILES_D)

clean:
	rm -r -f $(BIN)/*

test:
	./bin/simulation
	read -p "Press enter to continue" nothing

debug:
	/usr/local/cuda/bin/cuda-gdb ./bin/$(EXEC_NAME)

.PHONY: build clean

$(EXEC_FILES): $(SRC)/$(SRC_FILES)
	export PATH=/usr/local/cuda/bin$${PATH:+:$${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}
	@$(NVCC) $(NVFLAGS) -o $@ $^ -ccbin $(CC)
	@echo "Build successful"

$(EXEC_FILES_D): $(SRC)/$(SRC_FILES)
	export PATH=/usr/local/cuda/bin$${PATH:+:$${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}
	@$(NVCC) $(NVFLAGS) -o $@ $^ -ccbin $(CC) -g -G
	@echo "Build successful"
