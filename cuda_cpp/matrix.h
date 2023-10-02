/*
This is the definition file for the matrix structure and some matrix-realted functions. Note that
all matrices are stored in the row-major order and all indices are 0-based.
*/

#ifndef MATRIX_H
#define MATRIX_H

// Define the 1D matrix(array) structure and related functions
// 1D Matrix: M(col) = *(M.elements + col)
struct Matrix1D {
    const int width;
    float *elements;

    Matrix1D(int w) : width(w) { elements = new float[width]; }

    ~Matrix1D() { delete[] elements; }
};

inline __device__ float getElement1D(Matrix1D M, int col) { return M.elements[col]; }

inline __device__ void setElement1D(Matrix1D M, int col, float value) { M.elements[col] = value; }

// Define the 2D matrix structure and related functions
// 2D Matrix: M(row, col) = *(M.elements + row * M.width + col)
struct Matrix2D {
    const int width;
    const int height;
    float *elements;

    Matrix2D(int w, int h) : width(w), height(h) { elements = new float[width * height]; }

    ~Matrix2D() { delete[] elements; }
};

inline __device__ float getElement2D(Matrix2D M, int row, int col) {
    return M.elements[row * M.width + col];
}

inline __device__ void setElement2D(Matrix2D M, int row, int col, float value) {
    M.elements[row * M.width + col] = value;
}

// Define the 3D matrix structure and related functions
// 3D Matrix: M(row, col, layer1) = *(M.elements + layer1 * M.width * M.height + row * M.width +
// col)
struct Matrix3D {
    const int width;
    const int height;
    const int depth1;
    float *elements;

    Matrix3D(int w, int h, int d1) : width(w), height(h), depth1(d1) {
        elements = new float[width * height * depth1];
    }

    ~Matrix3D() { delete[] elements; }
};

inline __device__ float getElement3D(Matrix3D M, int row, int col, int layer1) {
    return M.elements[layer1 * M.width * M.height + row * M.width + col];
}

inline __device__ void setElement3D(Matrix3D M, int row, int col, int layer1, float value) {
    M.elements[layer1 * M.width * M.height + row * M.width + col] = value;
}

// Define the 4D matrix structure and related functions
// 4D Matrix: M(row, col, layer1, layer2) = *(M.elements + layer2 * M.width * M.height * M.depth +
// layer1 * M.width * M.height + row * M.width + col)
struct Matrix4D {
    const int width;
    const int height;
    const int depth1;
    const int depth2;
    float *elements;

    Matrix4D(int w, int h, int d1, int d2) : width(w), height(h), depth1(d1), depth2(d2) {
        elements = new float[width * height * depth1 * depth2];
    }

    ~Matrix4D() { delete[] elements; }
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
    const int width;
    const int height;
    const int depth1;
    const int depth2;
    const int depth3;
    float *elements;

    __host__ __device__ Matrix5D(int w, int h, int d1, int d2, int d3)
        : width(w), height(h), depth1(d1), depth2(d2), depth3(d3) {
        elements = new float[width * height * depth1 * depth2 * depth3];
    }

    __host__ __device__ ~Matrix5D() { delete[] elements; }
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