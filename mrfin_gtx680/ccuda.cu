#include"globals.h"
#include"cudaGlobals.h"
#include"cudaError.h"
float * devPii[2];
float * devTau[2];
float * devAfloat[2];
float * devAImag[2];
float * devBfloat[2];
float * devBImag[2];
float * devII[2];
int * devNmax[2];
cudaStream_t stream[2];
float * devReferences[2];
float * devErr[2];
float * devPatterns[2];
float * devInvRSquare[2];
float * devPSquare[2];
cudaStream_t streamRef[2];
float * devMin[2];
float * devMax[2];
int * devMinIndex[2];
float * devMedian[2];
int * devOut;


void freeCudaPointer(void ** pointer) {
	CudaSafeCall(cudaFreeHost(*pointer));
}
void allocCudaPointer(void ** pointer, size_t size) {
	CudaSafeCall(cudaMallocHost((void**)pointer, size));
}
	
void mallocCudaReferences(int i, int const mPatterns, int const nPatterns, int const mReferences, int const nReferences ) {
			CudaSafeCall(cudaStreamCreate(&streamRef[i]));
			CudaSafeCall(cudaMalloc((void**)&devPatterns[i], mPatterns*nPatterns*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devReferences[i],mReferences*nReferences*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devInvRSquare[i], mReferences*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devPSquare[i], mPatterns*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devErr[i], mPatterns*mReferences*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devMin[i], mPatterns*sizeof(float)));
			CudaSafeCall(cudaMalloc((void**)&devMax[i], mPatterns*sizeof(float))); //TODO: czy rozmiar dobry? (04.04.13 by szmigacz)
			CudaSafeCall(cudaMalloc((void**)&devMinIndex[i], mPatterns*sizeof(int)));
			CudaSafeCall(cudaMalloc((void**)&devMedian[i], mPatterns*sizeof(float)));

}

void freeCudaMemory() {
	#ifdef CUDA
		for(int i=0;i<2;i++) {
			CudaSafeCall(cudaStreamSynchronize(stream[i]));
		}
	
		for(int i=0;i<2;i++) {
			CudaSafeCall(cudaFree(devPii[i]));
			CudaSafeCall(cudaFree(devTau[i]));
			CudaSafeCall(cudaFree(devAfloat[i]));
			CudaSafeCall(cudaFree(devAImag[i]));
			CudaSafeCall(cudaFree(devBfloat[i]));
			CudaSafeCall(cudaFree(devBImag[i]));
			CudaSafeCall(cudaFree(devII[i]));
			CudaSafeCall(cudaFree(devNmax[i]));
			CudaSafeCall(cudaStreamDestroy(stream[i]));
		}
	#endif //CUDA
}

void freeCudaRefMemory() {
	#ifdef CUDA
		for(int i=0;i<2;i++) {
			CudaSafeCall(cudaStreamSynchronize(streamRef[i]));
		}
		for(int i=0;i<2;i++) {
	
			CudaSafeCall(cudaFree(devInvRSquare[i]));
			CudaSafeCall(cudaFree(devPSquare[i]));
			CudaSafeCall(cudaFree(devErr[i]));
			CudaSafeCall(cudaFree(devPatterns[i]));
			CudaSafeCall(cudaFree(devReferences[i]));
			CudaSafeCall(cudaFree(devMin[i]));
			CudaSafeCall(cudaFree(devMax[i]));
			CudaSafeCall(cudaFree(devMinIndex[i]));
			CudaSafeCall(cudaFree(devMedian[i]));
			CudaSafeCall(cudaStreamDestroy(streamRef[i]));
		}
	#endif //CUDA
}

void cudaFinalize() {
	#ifdef CUDA
		CudaSafeCall(cudaDeviceSynchronize());
	#endif //CUDA
}
void cuda1stPolarizationSync() {
	#ifdef CUDA
		CudaSafeCall(cudaStreamSynchronize(streamRef[0]));
		CudaSafeCall(cudaStreamSynchronize(streamRef[1]));
	#endif //CUDA
}

void freeCudaMemoryMin() {
	#ifdef CUDA
			CudaSafeCall(cudaFree(devOut));
	#endif //CUDA
}
