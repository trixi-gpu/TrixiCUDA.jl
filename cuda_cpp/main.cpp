/*
Main C++/CUDA file for testing all the kernel functions defined in `cuda_dg_1d.cu`, `cuda_dg_2d.cu`,
and `cuda_dg_3d.cu`. The testing are based on
*/

#include "cuda_dg_1d.cu"
#include "cuda_dg_2d.cu"
#include "cuda_dg_3d.cu"

#include <iostream>

int main() {
    std::cout << "Testing in 1D" << std::endl;

    std::cout << "Testing in 2D" << std::endl;

    std::cout << "Testing in 3D" << std::endl;

    return 0;
}