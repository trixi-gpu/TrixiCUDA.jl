/*
This is the definition file for the common array structures and some matrix-realted functions. Here
the linear memory is allocated through `cudaMalloc`. Note that all arrays are stored in the
row-major order and all indices are 0-based.
*/

#ifndef ARRAYS_H
#define ARRAYS_H

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

inline __host__ __device__ float getElement(Array A, int i) { return A.elements[i]; }

inline __host__ __device__ void setElement(Array A, int i, float value) { A.elements[i] = value; }

// Define the 2D array structure and related functions
// 2D Array: A(i, j) = *(A.elements + i * A.width + j)
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

inline __host__ __device__ float getElement2D(Array2D A, int i, int j) {
    return A.elements[i * A.width + j];
}

inline __host__ __device__ void setElement2D(Array2D A, int i, int j, float value) {
    A.elements[i * A.width + j] = value;
}

// Define the 3D array structure and related functions
// 3D Array: A(i, j, k) = *(A.elements + k * A.width * A.height + i * A.width + j)
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

inline __host__ __device__ float getElement3D(Array3D A, int i, int j, int k) {
    return A.elements[k * A.width * A.height + i * A.width + j];
}

inline __host__ __device__ void setElement3D(Array3D A, int i, int j, int k, float value) {
    A.elements[k * A.width * A.height + i * A.width + j] = value;
}

// Define the 4D array structure and related functions
// 4D Array: A(i, j1, j2, k) = *(A.elements + k * A.width * A.height1 * A.height2 + j2 * A.width *
// A.height1 + i * A.width + j1)
struct Array4D {
    int width;
    int height1;
    int height2;
    int depth;

    float *elements;
    size_t pitch;
    cudaPitchedPtr pitchedPtr;

    __host__ void initOnHost(int w, int h1, int h2, int d) {
        width = w;
        height1 = h1;
        height2 = h2;
        depth = d;
        elements = new float[width * height1 * height2 * depth];
    }

    __host__ void initOnDevice(int w, int h1, int h2, int d) {
        cudaExtent extent = make_cudaExtent(w * sizeof(float), h1 * h2, d);
        cudaMalloc3D(&pitchedPtr, extent);
        elements = (float *)pitchedPtr.ptr;
        pitch = pitchedPtr.pitch;
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(pitchedPtr.ptr); }
};

// Define the 5D array structure and related functions
// 5D Array: A(i, j1, j2, j3, k) = *(A.elements + k * A.width * A.height1 * A.height2 * A.height3 +
// j3 * A.width * A.height1 * A.height2 + j2 * A.width * A.height1 + i * A.width + j1)
struct Array5D {
    int width;
    int height1;
    int height2;
    int height3;
    int depth;

    float *elements;
    size_t pitch;
    cudaPitchedPtr pitchedPtr;

    __host__ void initOnHost(int w, int h1, int h2, int h3, int d) {
        width = w;
        height1 = h1;
        height2 = h2;
        height3 = h3;
        depth = d;
        elements = new float[width * height1 * height2 * height3 * depth];
    }

    __host__ void initOnDevice(int w, int h1, int h2, int h3, int d) {
        cudaExtent extent = make_cudaExtent(w * sizeof(float), h1 * h2 * h3, d);
        cudaMalloc3D(&pitchedPtr, extent);
        elements = (float *)pitchedPtr.ptr;
        pitch = pitchedPtr.pitch;
    }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(pitchedPtr.ptr); }
};

inline __host__ __device__ float getElement5D(Array5D A, int i, int j1, int j2, int j3, int k) {
    return A.elements[k * A.width * A.height1 * A.height2 * A.height3 +
                      j3 * A.width * A.height1 * A.height2 + j2 * A.width * A.height1 +
                      i * A.width + j1];
}

inline __host__ __device__ void setElement5D(Array5D A, int i, int j1, int j2, int j3, int k,
                                             float value) {
    A.elements[k * A.width * A.height1 * A.height2 * A.height3 +
               j3 * A.width * A.height1 * A.height2 + j2 * A.width * A.height1 + i * A.width + j1] =
        value;
}

// Define more matrix structure and related functions if needed...

#endif // ARRAYS_H
