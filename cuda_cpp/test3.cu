#include "arrays.h"
#include <iostream>

__global__ void DummyKernel(Array3D arr) {
    // This is just a placeholder kernel. You can add operations here.
    // For the sake of this example, it does nothing.
}

int main() {
    // 1. Initialize host array
    Array3D hostArray;
    hostArray.initOnHost(10, 10, 10); // For example, a 10x10x10 array

    // 2. Fill host array with some data
    for (int i = 0; i < hostArray.width * hostArray.height * hostArray.depth; ++i) {
        hostArray.elements[i] = static_cast<float>(i);
    }

    // 3. Initialize device array
    Array3D deviceArray;
    deviceArray.initOnDevice(10, 10, 10);

    // 4. Copy data from host to device
    copyToDevice(hostArray, deviceArray);

    // 5. Optionally modify the data on the device
    DummyKernel<<<1, 1>>>(deviceArray);
    cudaDeviceSynchronize();

    // 6. Copy data back from device to host
    copyToHost(deviceArray, hostArray);

    // 7. Verify the data (for this example, we'll just print some values)
    for (int i = 0; i < 100; ++i) { // Print first 10 values as a simple check
        std::cout << hostArray.elements[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    hostArray.freeOnHost();
    deviceArray.freeOnDevice();

    return 0;
}
