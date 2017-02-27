#include<math.h>
#include"globals.h"
#include "cudaGlobals.h"
#include"cudaError.h"
#include<cstdio>

#define THREADS 128
//texture<float> texIn;
texture<float> texErrs;
texture<float> texErrp;

__global__ void cudaFindMinKernel(float const * const invMedS, float const * const invMedP, int * out, int rSize) {
	volatile __shared__ float inShared[THREADS];
	volatile __shared__ int idxShared[THREADS];
	
	float value = INFINITY;
	int idx=0;
	float invMedSval = invMedS[blockIdx.x];
	float invMedPval = invMedP[blockIdx.x];

	for (int index = 0; index < rSize; index+=THREADS) {
		if (index + threadIdx.x < rSize) {
			float valErrs = tex1Dfetch( texErrs, blockIdx.x*rSize+ index + threadIdx.x);
			float valErrp = tex1Dfetch( texErrp, blockIdx.x*rSize+ index + threadIdx.x);
			float val = valErrs*valErrp*invMedSval*invMedPval;
			if( val < value) {
				value = val;
				idx = index + threadIdx.x;
			}
		}
	}
	
	inShared[threadIdx.x] = value;
	idxShared[threadIdx.x] = idx;
	__syncthreads();

	if( threadIdx.x < 64 ) {
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 64] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 64];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+64];
		}
	}
	__syncthreads();
	if ( threadIdx.x < 32) {
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 32] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 32];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+32];
		}
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 16] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 16];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+16];
		}
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 8] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 8];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+8];
		}
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 4] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 4];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+4];
		}
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 2] ) {
			inShared[threadIdx.x] = inShared[threadIdx.x + 2];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+2];
		}
		if( inShared[threadIdx.x] > inShared[threadIdx.x + 1] ) {
			//inShared[threadIdx.x] = inShared[threadIdx.x + 1];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+1];
		}
	}

	if(threadIdx.x==0){
		out[blockIdx.x] = idxShared[0];
	}
}


void cudaFindMin(int * out, int pol, int rSize, int movieSize) {
	//CudaSafeCall(cudaMalloc((void**)&devIn[pol], rSize*movieSize*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devOut, movieSize*sizeof(int)));
	CudaSafeCall(cudaBindTexture( NULL, texErrp, devErr[0], rSize*movieSize*sizeof(float)));
	CudaSafeCall(cudaBindTexture( NULL, texErrs, devErr[1], rSize*movieSize*sizeof(float)));
	cudaFuncSetCacheConfig(cudaFindMinKernel, cudaFuncCachePreferShared);
	cudaFindMinKernel<<<movieSize, THREADS, 0, 0>>>(devMedian[1], devMedian[0], devOut, rSize);
	CudaSafeCall(cudaMemcpy(out, devOut, movieSize*sizeof(int), cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaUnbindTexture(texErrs));
	CudaSafeCall(cudaUnbindTexture(texErrp));
}





