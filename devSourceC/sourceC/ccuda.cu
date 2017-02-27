#include"globals.h"
void freeCudaPointer(real ** pointer) {
	cudaError(cudaFreeHost(*pointer));
}
void allocCudaPointer(real ** pointer, size_t size) {
	cudaError(cudaMallocHost((void**)pointer, size));
}
