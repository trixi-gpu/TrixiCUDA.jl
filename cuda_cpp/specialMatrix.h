/*
This file defines special matrix structures and operations. The concept of specail matrix mainly
derives from `du` and `u` in the `rhs!` functions, which has multiple equal dimensions. Here the
linear memory is allocated through `cudaMallocPitch` and `cudaMalloc3D`.

Note that all matrices are stored in the row-major order and all indices are 0-based.
*/

#ifndef SPECIAL_MATRIX_H
#define SPECIAL_MATRIX_H

#endif // SPECIAL_MATRIX_H