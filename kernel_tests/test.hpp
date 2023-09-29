#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

// CUDA kernel to add elements of two arrays
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements);

#endif
