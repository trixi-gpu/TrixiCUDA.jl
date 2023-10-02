/*
Define kernel congifurators for parallel computing on 1D, 2D and 3D blocks and grids.
*/

#ifndef CONFIGURATOR_H
#define CONFIGURATOR_H

// CUDA kernel configurator for parallel computing on 1D blocks and grids
__host__ pair<dim3, dim3> configurator1D(void *kernelFun, int arrayLength) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       kernelFun); // Use CUDA occupancy calculator

    int threads = blockSize;
    int blocks = ceil(static_cast<float>(arrayLength) / threads);

    return {dim3(blocks), dim3(threads)};
}

// CUDA kernel configurator for parallel computing on 2D blocks and grids
__host__ pair<dim3, dim3> configurator2D(void *kernelFun, int arrayWidth, int arrayHeight) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFun);

    int threadsPerDimension = static_cast<int>(sqrt(blockSize));

    dim3 threads(threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x),
                ceil(static_cast<float>(arrayHeight) / threads.y));

    return {blocks, threads};
}

// CUDA kernel configurator for parallel computing on 3D blocks and grids
__host__ pair<dim3, dim3> configurator3D(void *kernelFun, int arrayWidth, int arrayHeight,
                                         int arrayDepth) {
    int blockSize;
    int minGridSize;

    // Get the potential block size for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       kernelFun); // Use CUDA occupancy calculator

    int threadsPerDimension = static_cast<int>(cbrt(blockSize));

    dim3 threads(threadsPerDimension, threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x),
                ceil(static_cast<float>(arrayHeight) / threads.y),
                ceil(static_cast<float>(arrayDepth) / threads.z));

    return {blocks, threads};
}

#endif // CONFIGURATOR_HS