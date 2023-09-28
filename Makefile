# Specify the NVIDIA GPU architecture
ARCH=sm_70

# The compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-arch=$(ARCH) -std=c++17

# The target binary program
TARGET=cuda_dg_1d # cuda_dg_1d, cuda_dg_2d, cuda_dg_3d

all: $(TARGET)

$(TARGET): cuda_dg_1d.cu header.hpp # cuda_dg_1d.cu, cuda_dg_2d.cu, cuda_dg_3d.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
