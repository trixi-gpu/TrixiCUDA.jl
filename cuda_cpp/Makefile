# Specify the NVIDIA GPU architecture
ARCH=sm_70

# The compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-arch=$(ARCH) -std=c++17

# The target binary program
TARGET=test2 	# cuda_dg_1d, cuda_dg_2d, cuda_dg_3d
OBJECTS=test2.o	# cuda_dg_1d.o, cuda_dg_2d.o, cuda_dg_3d.o

all: $(TARGET)

$(OBJECTS): test2.cu # cuda_dg_1d.cu, cuda_dg_2d.cu, cuda_dg_3d.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
