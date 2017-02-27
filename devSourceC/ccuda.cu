#include"globals.h"
void freeCudaPointer(float ** pointer) {
	cudaError(cudaFreeHost(*pointer));
}
void allocCudaPointer(float ** pointer, size_t size) {
	cudaError(cudaMallocHost((void**)pointer, size));
}
