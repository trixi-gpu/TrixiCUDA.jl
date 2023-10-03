/*
Define kernel congifurators for parallel computing on 1D, 2D and 3D blocks and grids. Here the CUDA
occupancy calculator is applied.
*/

#ifndef CONFIGURATOR_H
#define CONFIGURATOR_H

// CUDA kernel configurator for parallel computing on 1D blocks and grids
__host__ std::pair<dim3, dim3> configurator1D(void *kernelFun, int arrayLength) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFun);

    int threadsPerDimension = blockSize;
    int blocksPerDimension = ceil(arrayLength / threadsPerDimension);

    dim3 configBlocks(threadsPerDimension);
    dim3 configGrids(blocksPerDimension);

    return {configGrids, configBlocks};
}

// CUDA kernel configurator for parallel computing on 2D blocks and grids
__host__ std::pair<dim3, dim3> configurator2D(void *kernelFun, int arrayWidth, int arrayHeight) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFun);

    int threadsPerDimension = sqrt(blockSize);

    dim3 configBlocks(threadsPerDimension, threadsPerDimension);
    dim3 configGrids(ceil(arrayWidth / configBlocks.x), ceil(arrayHeight / configBlocks.y));

    return {configGrids, configBlocks};
}

// CUDA kernel configurator for parallel computing on 3D blocks and grids
__host__ std::pair<dim3, dim3> configurator3D(void *kernelFun, int arrayWidth, int arrayHeight,
                                              int arrayDepth) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFun);

    int threadsPerDimension = cbrt(blockSize);

    dim3 configBlocks(threadsPerDimension, threadsPerDimension, threadsPerDimension);
    dim3 configGrids(ceil(arrayWidth / configBlocks.x), ceil(arrayHeight / configBlocks.y),
                     ceil(arrayDepth / configBlocks.z));

    return {configGrids, configBlocks};
}

#endif // CONFIGURATOR_HS