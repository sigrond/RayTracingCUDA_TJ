#include<cstdio>
#include<omp.h>
#include"globals.h"
#define THREADSREF 128
__global__ void kernelCalcInvSquare(float *const references, float *const invSquare, int mReferences, int nReferences) {
	volatile __shared__ float values[128];
	int index = blockIdx.x*nReferences + threadIdx.x;
	register float partial;

	if( threadIdx.x < nReferences ) {
		float value = references[index];
		partial = value * value;
		for(int id = 128; id < nReferences; id+=128) {
			if( threadIdx.x + id < nReferences ) {
				index += 128;
				float value = references[index];
				partial += value * value;
			}
		}
	}
	if ( threadIdx.x < nReferences )
		values[threadIdx.x]=partial;
	else
		values[threadIdx.x]=0.0f;
	__syncthreads();

	if(threadIdx.x < 64 )
		values[threadIdx.x]+=values[threadIdx.x+64];
	__syncthreads();
	if(threadIdx.x < 32) {
		values[threadIdx.x]+=values[threadIdx.x+32];
		values[threadIdx.x]+=values[threadIdx.x+16];
		values[threadIdx.x]+=values[threadIdx.x+8];
		values[threadIdx.x]+=values[threadIdx.x+4];
		values[threadIdx.x]+=values[threadIdx.x+2];
		values[threadIdx.x]+=values[threadIdx.x+1];
		if(threadIdx.x==0) {
			invSquare[blockIdx.x] = 1.0f/(values[0]);
		}
	}
}
__global__ void kernelCalcSquare(float * const references, float * const square, int mReferences, int nReferences) {
	volatile __shared__ float values[128];
	int index = blockIdx.x*nReferences + threadIdx.x;
	register float partial;

	if( threadIdx.x < nReferences ) {
		float value = references[index];
		partial = value * value;
		for(int id = 128; id < nReferences; id+=128) {
			if( threadIdx.x + id < nReferences ) {
				index += 128;
				float value = references[index];
				partial += value * value;
			}
		}
	}
	if ( threadIdx.x < nReferences )
		values[threadIdx.x]=partial;
	else
		values[threadIdx.x]=0.0f;
	__syncthreads();

	if(threadIdx.x < 64 )
		values[threadIdx.x]+=values[threadIdx.x+64];
	__syncthreads();
	if(threadIdx.x < 32) {
		values[threadIdx.x]+=values[threadIdx.x+32];
		values[threadIdx.x]+=values[threadIdx.x+16];
		values[threadIdx.x]+=values[threadIdx.x+8];
		values[threadIdx.x]+=values[threadIdx.x+4];
		values[threadIdx.x]+=values[threadIdx.x+2];
		values[threadIdx.x]+=values[threadIdx.x+1];
		if(threadIdx.x==0)
			square[blockIdx.x] = values[0];
	}
}
//sprawdzic inne konfiguracje liczby watkow i pamieci shaaared
__global__ void __launch_bounds__(THREADSREF, 8) kernelReferenceDistance(float const * const references, float const * const patterns, float const * const square, float const * const pSquare, float * const err, int const nPatterns, int const nReferences,int const mReferences) {
	volatile __shared__ float pr[THREADSREF];
	volatile __shared__ float pr2[THREADSREF];
	volatile __shared__ float pr3[THREADSREF];
	volatile __shared__ float pr4[THREADSREF];
	register int indexPatterns = blockIdx.x * nPatterns  + threadIdx.x;
	register int indexReferences = (4*blockIdx.y) * nReferences  + threadIdx.x;
	register int indexReferences2 = indexReferences + nReferences ;//(4*blockIdx.y+1) * nReferences  + threadIdx.x;
	register int indexReferences3 = indexReferences2 + nReferences;//(4*blockIdx.y+2) * nReferences  + threadIdx.x;
	register int indexReferences4 = indexReferences3 + nReferences ;//(4*blockIdx.y+3) * nReferences  + threadIdx.x;
	register float prPartial;
	register float prPartial2;
	register float prPartial3;
	register float prPartial4;

	if( threadIdx.x < nPatterns ) {
		float p = patterns[indexPatterns];
		float r = references[indexReferences];
		float r2 = references[indexReferences2];
		float r3 = references[indexReferences3];
		float r4 = references[indexReferences4];
		prPartial = p*r;
		prPartial2 = p*r2;
		prPartial3 = p*r3;
		prPartial4 = p*r4;
		for(int index=THREADSREF;index<nPatterns;index+=THREADSREF) {
			if( index+threadIdx.x < nPatterns )
			{
				indexPatterns +=THREADSREF; 
				indexReferences +=THREADSREF; 
				indexReferences2 +=THREADSREF; 
				indexReferences3 +=THREADSREF; 
				indexReferences4 +=THREADSREF; 
				p = patterns[indexPatterns];
				r = references[indexReferences];
				r2 = references[indexReferences2];
				r3 = references[indexReferences3];
				r4 = references[indexReferences4];
				prPartial += p*r;
				prPartial2 += p*r2;
				prPartial3 += p*r3;
				prPartial4 += p*r4;
			}
		}
	}
	if(threadIdx.x < nPatterns){
		pr[threadIdx.x]=prPartial;
		pr2[threadIdx.x]=prPartial2;
		pr3[threadIdx.x]=prPartial3;
		pr4[threadIdx.x]=prPartial4;
	}
	else {
		pr[threadIdx.x]=0.0f;
		pr2[threadIdx.x]=0.0f;
		pr3[threadIdx.x]=0.0f;
		pr4[threadIdx.x]=0.0f;
	}
	__syncthreads();
	if(threadIdx.x < 64 ){
		pr[threadIdx.x]+=pr[threadIdx.x+64];
		pr2[threadIdx.x]+=pr2[threadIdx.x+64];
		pr3[threadIdx.x]+=pr3[threadIdx.x+64];
		pr4[threadIdx.x]+=pr4[threadIdx.x+64];
	}
	__syncthreads();
	if(threadIdx.x < 32) {
		pr[threadIdx.x] += pr[threadIdx.x+32];
		pr[threadIdx.x] += pr[threadIdx.x+16];
		pr[threadIdx.x] += pr[threadIdx.x+8];
		pr[threadIdx.x] += pr[threadIdx.x+4];
		pr[threadIdx.x] += pr[threadIdx.x+2];
		pr[threadIdx.x] += pr[threadIdx.x+1];

		pr2[threadIdx.x] += pr2[threadIdx.x+32];
		pr2[threadIdx.x] += pr2[threadIdx.x+16];
		pr2[threadIdx.x] += pr2[threadIdx.x+8];
		pr2[threadIdx.x] += pr2[threadIdx.x+4];
		pr2[threadIdx.x] += pr2[threadIdx.x+2];
		pr2[threadIdx.x] += pr2[threadIdx.x+1];

		pr3[threadIdx.x] += pr3[threadIdx.x+32];
		pr3[threadIdx.x] += pr3[threadIdx.x+16];
		pr3[threadIdx.x] += pr3[threadIdx.x+8];
		pr3[threadIdx.x] += pr3[threadIdx.x+4];
		pr3[threadIdx.x] += pr3[threadIdx.x+2];
		pr3[threadIdx.x] += pr3[threadIdx.x+1];

		pr4[threadIdx.x] += pr4[threadIdx.x+32];
		pr4[threadIdx.x] += pr4[threadIdx.x+16];
		pr4[threadIdx.x] += pr4[threadIdx.x+8];
		pr4[threadIdx.x] += pr4[threadIdx.x+4];
		pr4[threadIdx.x] += pr4[threadIdx.x+2];
		pr4[threadIdx.x] += pr4[threadIdx.x+1];
		if ( threadIdx.x==0 ){
			int index = blockIdx.x*mReferences + 4u*blockIdx.y;
			int index2 = 4u*blockIdx.y;
			err[index]   = pSquare[blockIdx.x] - pr[0]*pr[0]*square[index2];
			err[index+1] = pSquare[blockIdx.x] - pr2[0]*pr2[0]*square[index2+1];
			err[index+2] = pSquare[blockIdx.x] - pr3[0]*pr3[0]*square[index2+2];
			err[index+3] = pSquare[blockIdx.x] - pr4[0]*pr4[0]*square[index2+3];
			}
	}
}

void cudaReferenceDistance(float ** err, float const * const patterns, float const * const references, int const mPatterns, int const nPatterns, int const mReferences, int const nReferences) {
	float * devReferences;
	float * devErr;
	float * devPatterns;
	float * devInvRSquare;
	float * devPSquare;
	cudaError(cudaMalloc((void**)&devPatterns, mPatterns*nPatterns*sizeof(float)));
	cudaError(cudaMalloc((void**)&devReferences,mReferences*nReferences*sizeof(float)));
	cudaError(cudaMalloc((void**)&devInvRSquare, mReferences*sizeof(float)));
	cudaError(cudaMalloc((void**)&devPSquare, mPatterns*sizeof(float)));
	//printf("%d %d %d\n", mPatterns, mReferences, nPatterns);

	cudaError(cudaMemcpy(devReferences, references,mReferences*nReferences*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devPatterns, patterns, mPatterns*nPatterns*sizeof(float), cudaMemcpyHostToDevice));

	cudaFuncSetCacheConfig(kernelCalcInvSquare, cudaFuncCachePreferL1);
	kernelCalcInvSquare<<< mReferences, 128>>>(devReferences, devInvRSquare, mReferences, nReferences);
	cudaFuncSetCacheConfig(kernelCalcSquare, cudaFuncCachePreferL1);
	kernelCalcSquare<<< mPatterns, 128>>>(devPatterns, devPSquare, mPatterns, nPatterns);
	cudaError(cudaMalloc((void**)&devErr, mPatterns*mReferences*sizeof(float)));

	cudaFuncSetCacheConfig(kernelReferenceDistance, cudaFuncCachePreferL1);
	//printf("%d %d %d %d\n", mPatterns, nPatterns, mReferences, nReferences);
	kernelReferenceDistance<<<dim3(mPatterns, mReferences/4, 1),THREADSREF>>>(devReferences, devPatterns, devInvRSquare, devPSquare, devErr, nPatterns, nReferences, mReferences);

	cudaError(cudaMallocHost((void**)err, mPatterns*mReferences*sizeof(float)));
	cudaError(cudaMemcpy(*err, devErr, mPatterns*mReferences*sizeof(float), cudaMemcpyDeviceToHost));
	cudaError(cudaFree(devInvRSquare));
	cudaError(cudaFree(devPSquare));
	cudaError(cudaFree(devErr));
	cudaError(cudaFree(devPatterns));
	cudaError(cudaFree(devReferences));
	cudaError(cudaDeviceSynchronize());
}
