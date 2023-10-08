/*
This file defines the helper functions and flux functions for kernel calls in the 1D, 2D, and 3D
cases.
*/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// Forward declaration
struct Array;
struct Array3D;
struct AbstractEquations;

// Helper functions
__device__ Array getNodesVars(Array3D array, AbstractEquations euqation, int j, int k);

// Flux functions
__device__ Array flux(Array array, int direction, AbstractEquations equation);

#endif // FUNCTIONS_H