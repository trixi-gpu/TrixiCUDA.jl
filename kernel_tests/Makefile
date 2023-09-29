# Specify the NVIDIA GPU architecture
ARCH=sm_70

# The compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-arch=$(ARCH) -std=c++17

# The target binary program
TARGET=kernel_test3 	# kernel_test1, kernel_test2, kernel_test3
OBJECTS=kernel_test3.o	# kernel_test1.o, kernel_test2.o, kernel_test3.o

all: $(TARGET)

$(OBJECTS): kernel_test3.cu test.h # kernel_test1.cu, kernel_test2.cu, kernel_test3.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
