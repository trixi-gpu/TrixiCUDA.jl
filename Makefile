# Specify the NVIDIA GPU architecture
ARCH=sm_70

# The compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-arch=$(ARCH) -std=c++17

# The target binary program
TARGET=test

all: $(TARGET)

$(TARGET): test.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
