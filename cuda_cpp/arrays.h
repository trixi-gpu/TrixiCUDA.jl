/*
This is the definition file for the common array structures and some matrix-realted functions. Note
that all arrays are stored in the row-major order and all indices are 0-based.
*/

#ifndef ARRAYS_H
#define ARRAYS_H

// Define the 1D array structure and related functions
/*
====================================================================================================
Note that the 1D array can be directly store in 1D structure on device, so there is no need to
provide any convertion formulas here.
====================================================================================================
*/
struct Array {
    int width;

    float *elements;

    __host__ void initOnHost(int w) { elements = new float[w]; }

    __host__ void initOnDevice(int w) { cudaMalloc(&elements, w * sizeof(float)); }

    __host__ void freeOnHost() { delete[] elements; }

    __host__ void freeOnDevice() { cudaFree(elements); }
};

__host__ void copyToDevice(Array hostArray, Array deviceArray) {
    size_t size = hostArray.width * sizeof(float);
    float *hostPtr = hostArray.elements;
    float *devicePtr = deviceArray.elements;

    cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
}

__host__ void copyToHost(Array deviceArray, Array hostArray) {
    size_t size = deviceArray.width * sizeof(float);
    float *devicePtr = deviceArray.elements;
    float *hostPtr = hostArray.elements;

    cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
}

// Define the 2D array structure and related functions
/*
====================================================================================================
Note that the 2D array can be directly store in 2D structure on device, so there is no need to
provide any convertion formulas here.
====================================================================================================
*/
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

__host__ void copyToDevice(Array2D hostArray, Array2D deviceArray) {
    float *hostPtr = hostArray.elements;
    float *devicePtr = deviceArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy2D(devicePtr, pitch, hostPtr, hostArray.width * sizeof(float),
                 hostArray.width * sizeof(float), hostArray.height, cudaMemcpyHostToDevice);
}

__host__ void copyToHost(Array2D deviceArray, Array2D hostArray) {
    float *devicePtr = deviceArray.elements;
    float *hostPtr = hostArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy2D(hostPtr, hostArray.width * sizeof(float), devicePtr, pitch,
                 deviceArray.width * sizeof(float), deviceArray.height, cudaMemcpyDeviceToHost);
}

// Define the 3D array structure and related functions
/*
====================================================================================================
Note that the 3D array can be directly store in 3D structure on device, so there is no need to
provide any convertion formulas here.
====================================================================================================
*/
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

__host__ void copyToDevice(Array3D hostArray, Array3D deviceArray) {
    float *hostPtr = hostArray.elements;
    float *devicePtr = deviceArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float),
                                            hostArray.width, hostArray.height);
    copyParams.dstPtr =
        make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width, deviceArray.height);
    copyParams.extent =
        make_cudaExtent(hostArray.width * sizeof(float), hostArray.height, hostArray.depth);
    copyParams.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);
}

__host__ void copyToHost(Array3D deviceArray, Array3D hostArray) {
    float *devicePtr = deviceArray.elements;
    float *hostPtr = hostArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width, deviceArray.height);
    copyParams.dstPtr = make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float),
                                            hostArray.width, hostArray.height);
    copyParams.extent =
        make_cudaExtent(hostArray.width * sizeof(float), hostArray.height, hostArray.depth);
    copyParams.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3D(&copyParams);
}

// Define the 4D array structure (stored in 3D structure on device) and related functions
/*
====================================================================================================
Suppose the 4D array is of dimension (a, b1, b2, c) and the corresponding 3D array is of
dimension (a, b1 * b2, c), then we have the following convertion formulas, which are useful
when we access the specific element of the arrays in the kernels.

1. Convertion formula (forward from 4D to 3D, when initialize data in the cpp file):
A[i][j1][j2][k] = A[i][j2 * b1 + j1][k]

2. Convertion formula (back from 3D to 4D, when acess data in the kernel):
A[i][j][k] = A[i][j1][j2][k] where
j2 = j1 / b1
j1 = j1 % b1

Note that the convertion method can be either row-based or column-based as long as it is consistent
in both forward and backward directions.
====================================================================================================
*/
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

__host__ void copyToDevice(Array4D hostArray, Array4D deviceArray) {
    float *hostPtr = hostArray.elements;
    float *devicePtr = deviceArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float),
                                            hostArray.width, hostArray.height1 * hostArray.height2);
    copyParams.dstPtr = make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width,
                                            deviceArray.height1 * deviceArray.height2);
    copyParams.extent = make_cudaExtent(hostArray.width * sizeof(float),
                                        hostArray.height1 * hostArray.height2, hostArray.depth);
    copyParams.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);
}

__host__ void copyToHost(Array4D deviceArray, Array4D hostArray) {
    float *devicePtr = deviceArray.elements;
    float *hostPtr = hostArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width,
                                            deviceArray.height1 * deviceArray.height2);
    copyParams.dstPtr = make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float),
                                            hostArray.width, hostArray.height1 * hostArray.height2);
    copyParams.extent = make_cudaExtent(hostArray.width * sizeof(float),
                                        hostArray.height1 * hostArray.height2, hostArray.depth);
    copyParams.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3D(&copyParams);
}

// Define the 5D array structure (stored in 3D structure on device) and related functions
/*
====================================================================================================
Suppose the 5D array is of dimension (a, b1, b2, b3, c) and the corresponding 3D array is of
dimension (a, b1 * b2 * b3, c), then we have the following convertion formulas, which are useful
when we access the specific element of the arrays in the kernels.

1. Convertion formula (forward from 4D to 3D, when initialize data in the cpp file):
A[i][j1][j2][j3][k] = A[i][j3 * b1 * b2 + j2 * b1 + j1][k]

2. Convertion formula (back from 3D to 4D, when acess data in the kernel):
A[i][j][k] = A[i][j1][j2][j3][k] where
j3 = j1 / (b1 * b2)
j2 = (j1 % (b1 * b2)) / b1
j1 = (j1 % (b1 * b2)) % b1

Note that the convertion method can be either row-based or column-based as long as it is consistent
in both forward and backward directions.
====================================================================================================
*/
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

__host__ void copyToDevice(Array5D hostArray, Array5D deviceArray) {
    float *hostPtr = hostArray.elements;
    float *devicePtr = deviceArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float), hostArray.width,
                            hostArray.height1 * hostArray.height2 * hostArray.height3);
    copyParams.dstPtr =
        make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width,
                            deviceArray.height1 * deviceArray.height2 * deviceArray.height3);
    copyParams.extent =
        make_cudaExtent(hostArray.width * sizeof(float),
                        hostArray.height1 * hostArray.height2 * hostArray.height3, hostArray.depth);
    copyParams.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);
}

__host__ void copyToHost(Array5D deviceArray, Array5D hostArray) {
    float *devicePtr = deviceArray.elements;
    float *hostPtr = hostArray.elements;
    size_t pitch = deviceArray.pitch;

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(devicePtr, pitch, deviceArray.width,
                            deviceArray.height1 * deviceArray.height2 * deviceArray.height3);
    copyParams.dstPtr =
        make_cudaPitchedPtr(hostPtr, hostArray.width * sizeof(float), hostArray.width,
                            hostArray.height1 * hostArray.height2 * hostArray.height3);
    copyParams.extent =
        make_cudaExtent(hostArray.width * sizeof(float),
                        hostArray.height1 * hostArray.height2 * hostArray.height3, hostArray.depth);
    copyParams.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3D(&copyParams);
}

// Define more matrix structure and related functions if needed...

#endif // ARRAYS_H