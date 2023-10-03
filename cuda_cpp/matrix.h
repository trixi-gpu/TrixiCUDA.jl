/*
This is the definition file for the common matrix structures (different from special matrix
structures) and some matrix-realted functions. Here the linear memory is allocated through
`cudaMalloc`.
Note that all matrices are stored in the row-major order and all indices are 0-based.
*/

#ifndef MATRIX_H
#define MATRIX_H

// Define the 1D array structure and related functions
// 1D Array: A(i) = *(A.elements + i)
struct Array {
    int width;

    float *elements;

    __host__ void initOnHost(int w) { elements = new float[w]; }

    __host__ void initOnDevice(int w) { cudaMalloc(&elements, w * sizeof(float)); }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(elements); }
};

inline __device__ float getElement(Array A, int col) { return A.elements[col]; }

inline __device__ void setElement(Array A, int col, float value) { A.elements[col] = value; }

// Define the 2D matrix structure and related functions
// 2D Matrix: M(row, col) = *(M.elements + row * M.width + col)
struct Array2D {
    int width;
    int height;

    float *elements;
    size_t pitch;

    __host__ void initOnHost(int w, int h) { elements = new float[w * h]; }

    __host__ void initOnDevice(int w, int h) {
        cudaMallocPitch(&elements, &pitch, w * sizeof(float), h);
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(elements); }
};

inline __device__ float getElement2D(Array2D A, int row, int col) {
    return A.elements[row * A.width + col];
}

inline __device__ void setElement2D(Array2D A, int row, int col, float value) {
    A.elements[row * A.width + col] = value;
}

// Define the 3D matrix structure and related functions
// 3D Matrix: M(row, col, layer1) = *(M.elements + layer1 * M.width * M.height + row * M.width +
// col)
struct Array3D {
    int width;
    int height;
    int depth;

    float *elements;
    size_t pitch;
    cudaPitchedPtr pitchedPtr;

    __host__ void initOnHost(int w, int h, int d) {
        width = w;
        height = h;
        depth = d;
        elements = new float[width * height * depth];
    }

    __host__ void initOnDevice(int w, int h, int d) {
        cudaExtent extent = make_cudaExtent(w * sizeof(float), h, d);
        cudaMalloc3D(&pitchedPtr, extent);
        elements = (float *)pitchedPtr.ptr;
        pitch = pitchedPtr.pitch;
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(pitchedPtr.ptr); }
};

inline __device__ float getElement3D(Array3D A, int row, int col, int layer1) {
    return A.elements[layer1 * A.width * A.height + row * A.width + col];
}

inline __device__ void setElement3D(Array3D A, int row, int col, int layer1, float value) {
    A.elements[layer1 * A.width * A.height + row * A.width + col] = value;
}

// Define the 4D matrix structure and related functions
// 4D Matrix: M(row, col, layer1, layer2) = *(M.elements + layer2 * M.width * M.height * M.depth +
// layer1 * M.width * M.height + row * M.width + col)
struct Matrix4D {
    int width;
    int height;
    int depth1;
    int depth2;
    float *elements;

    __host__ void initOnHost(int w, int h, int d1, int d2) {
        width = w;
        height = h;
        depth1 = d1;
        depth2 = d2;
        elements = new float[width * height * depth1 * depth2];
    }

    __host__ void initOnDevice(int w, int h, int d1, int d2) {
        width = w;
        height = h;
        depth1 = d1;
        depth2 = d2;
        cudaMalloc(&elements, width * height * depth1 * depth2 * sizeof(float));
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(elements); }
};

inline __device__ float getElement4D(Matrix4D M, int row, int col, int layer1, int layer2) {
    return M.elements[layer2 * M.width * M.height * M.depth1 + layer1 * M.width * M.height +
                      row * M.width + col];
}

inline __device__ void setElement4D(Matrix4D M, int row, int col, int layer1, int layer2,
                                    float value) {
    M.elements[layer2 * M.width * M.height * M.depth1 + layer1 * M.width * M.height +
               row * M.width + col] = value;
}

// Define the 5D matrix structure and related functions
// 5D Matrix: M(row, col, layer1, layer2, layer3) = *(M.elements + layer3 * M.width * M.height *
// M.depth1 * M.depth2 + layer2 * M.width * M.height * M.depth1 + layer1 * M.width * M.height + row
// * M.width + col)
struct Matrix5D {
    int width;
    int height;
    int depth1;
    int depth2;
    int depth3;
    float *elements;

    __host__ void initOnHost(int w, int h, int d1, int d2, int d3) {
        width = w;
        height = h;
        depth1 = d1;
        depth2 = d2;
        depth3 = d3;
        elements = new float[width * height * depth1 * depth2 * depth3];
    }

    __host__ void initOnDevice(int w, int h, int d1, int d2, int d3) {
        width = w;
        height = h;
        depth1 = d1;
        depth2 = d2;
        depth3 = d3;
        cudaMalloc(&elements, width * height * depth1 * depth2 * depth3 * sizeof(float));
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(elements); }
};

inline __device__ float getElement5D(Matrix5D M, int row, int col, int layer1, int layer2,
                                     int layer3) {
    return M.elements[layer3 * M.width * M.height * M.depth1 * M.depth2 +
                      layer2 * M.width * M.height * M.depth1 + layer1 * M.width * M.height +
                      row * M.width + col];
}

inline __device__ void setElement5D(Matrix5D M, int row, int col, int layer1, int layer2,
                                    int layer3, float value) {
    M.elements[layer3 * M.width * M.height * M.depth1 * M.depth2 +
               layer2 * M.width * M.height * M.depth1 + layer1 * M.width * M.height +
               row * M.width + col] = value;
}

// Define more matrix structure and related functions if needed...

#endif // MATRIX_H