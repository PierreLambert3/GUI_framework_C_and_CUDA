CC=gcc
NVCC=nvcc

# path to the external libraries' headers
EXT_LIB_INCLUDE_PATH=-I./c_lib/ext_libs/include

# Include the external library headers in GCC_FLAGS
GCC_FLAGS=-I./c_lib/include -I./c_lib/ext_libs/include -I/usr/include/SDL2 -O3 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -ffast-math -fopenmp -march=native -funroll-loops -flto -finline-functions -fexpensive-optimizations -fipa-pta
NVCC_FLAGS=-I./c_lib/ext_libs/include -O3

GCC_LDFLAGS=-lSDL2 -lSDL2_ttf -lSDL2_gfx -lm -fopenmp -lGL
NVCC_LDFLAGS=-lcudart -lSDL2 -lSDL2_ttf -lm -Xcompiler -fopenmp,-ffast-math -lGL, -lSDL2_gfx

# Include the external library headers in DEPS
DEPS = $(wildcard c_lib/include/*.h) $(wildcard c_lib/ext_libs/include/*.h)

OBJ_DIR = obj

# Update object files paths for c_lib and c_lib/ext_libs to avoid naming conflicts
C_OBJ = $(patsubst c_lib/src/%.c,$(OBJ_DIR)/%.o,$(wildcard c_lib/src/*.c))
C_EXT_LIB_OBJ = $(patsubst c_lib/ext_libs/src/%.c,$(OBJ_DIR)/c_ext_libs/%.o,$(wildcard c_lib/ext_libs/src/*.c))

CU_SRC = $(wildcard cuda_lib/src/*.cu)
CU_OBJ = $(patsubst cuda_lib/src/%.cu,$(OBJ_DIR)/%.o,$(CU_SRC))

all: clean main run

# Rule for compiling c_lib/ext_libs/src files
$(OBJ_DIR)/c_ext_libs/%.o: c_lib/ext_libs/src/%.c $(DEPS)
	@mkdir -p $(@D)
	@$(CC) -c -o $@ $< $(GCC_FLAGS)

# Generic rule for compiling .c files to .o files
$(OBJ_DIR)/%.o: c_lib/src/%.c $(DEPS)
	@mkdir -p $(@D)
	@$(CC) -c -o $@ $< $(GCC_FLAGS)

# Rule for compiling CUDA source files
$(OBJ_DIR)/%.o: cuda_lib/src/%.cu $(DEPS)
	@mkdir -p $(@D)
	@$(NVCC) $(NVCC_FLAGS) -c -o $@ $< 

# Updated main target to include C_EXT_LIB_OBJ
main: $(C_OBJ) $(C_EXT_LIB_OBJ) $(CU_OBJ)
	@mkdir -p ../bin
	@printf "[compiling..."; $(NVCC) -o ../bin/main $^ $(NVCC_LDFLAGS) && printf "\e[38;5;82mOK\e0]\n"

run: main
	@../bin/main

clean:
	@rm -rf $(OBJ_DIR) ../bin/main
