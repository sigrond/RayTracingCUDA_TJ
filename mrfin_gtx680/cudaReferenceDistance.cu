#include "math.h"
#include<cstdio>
#include<omp.h>
#include"globals.h"
#include"cudaGlobals.h"
#include"cudaError.h"
#define THREADSREF 128
#define THREADS 128




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
//sprawdzic inne konfiguracje liczby watkow i pamieci shared

texture<float> texPatterns;
texture<float> texReferences;

#if __CUDA_ARCH__ >= 300
__global__ void kernelReferenceDistance(float const * const references, float const * const patterns, float const * const square, float const * const pSquare, float * const err, int const nPatterns, int const nReferences,int const mReferences) {
	volatile __shared__ float pr1[THREADSREF/32];
	volatile __shared__ float pr2[THREADSREF/32];
	volatile __shared__ float pr3[THREADSREF/32];
	volatile __shared__ float pr4[THREADSREF/32];
	register int indexPatterns = blockIdx.x * nPatterns  + threadIdx.x;
	register int indexReferences = (4*blockIdx.y) * nReferences  + threadIdx.x;
	register int indexReferences2 = indexReferences + nReferences ;//(4*blockIdx.y+1) * nReferences  + threadIdx.x;
	register int indexReferences3 = indexReferences2 + nReferences;//(4*blockIdx.y+2) * nReferences  + threadIdx.x;
	register int indexReferences4 = indexReferences3 + nReferences ;//(4*blockIdx.y+3) * nReferences  + threadIdx.x;
	register float prPartial1=0.0f;
	register float prPartial2=0.0f;
	register float prPartial3=0.0f;
	register float prPartial4=0.0f;

	if( threadIdx.x < nPatterns ) {
		float p  = tex1Dfetch(texPatterns,    indexPatterns);
		float r1 = tex1Dfetch(texReferences, indexReferences);
		float r2 = tex1Dfetch(texReferences, indexReferences2);
		float r3 = tex1Dfetch(texReferences, indexReferences3);
		float r4 = tex1Dfetch(texReferences, indexReferences4);
		prPartial1 = p*r1;
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
				p  = tex1Dfetch(texPatterns,    indexPatterns);
				r1 = tex1Dfetch(texReferences, indexReferences);
				r2 = tex1Dfetch(texReferences, indexReferences2);
				r3 = tex1Dfetch(texReferences, indexReferences3);
				r4 = tex1Dfetch(texReferences, indexReferences4);
				prPartial1 += p*r1;
				prPartial2 += p*r2;
				prPartial3 += p*r3;
				prPartial4 += p*r4;
			}
		}
	}
	//butterfly reduction
	for (int i=16; i>=1; i/=2) {
		prPartial1 += __shfl_xor(prPartial1, i, 32);
		prPartial2 += __shfl_xor(prPartial2, i, 32);
		prPartial3 += __shfl_xor(prPartial3, i, 32);
		prPartial4 += __shfl_xor(prPartial4, i, 32);
	}
	if(threadIdx.x % 32 == 0) {
		pr1[threadIdx.x>>5] = prPartial1;
		pr2[threadIdx.x>>5] = prPartial2;
		pr3[threadIdx.x>>5] = prPartial3;
		pr4[threadIdx.x>>5] = prPartial4;
	}
	__syncthreads();
	if(threadIdx.x <2) {
		pr1[threadIdx.x] += pr1[threadIdx.x+2];
		pr2[threadIdx.x] += pr2[threadIdx.x+2];
		pr3[threadIdx.x] += pr3[threadIdx.x+2];
		pr4[threadIdx.x] += pr4[threadIdx.x+2];
		pr1[threadIdx.x] += pr1[threadIdx.x+1];
		pr2[threadIdx.x] += pr2[threadIdx.x+1];
		pr3[threadIdx.x] += pr3[threadIdx.x+1];
		pr4[threadIdx.x] += pr4[threadIdx.x+1];
	}



	if ( threadIdx.x==0 ){
		int index = blockIdx.x*mReferences + 4u*blockIdx.y;
		int index2 = 4u*blockIdx.y;
		err[index]   = pSquare[blockIdx.x] - pr1[0]*pr1[0]*square[index2];
		err[index+1] = pSquare[blockIdx.x] - pr2[0]*pr2[0]*square[index2+1];
		err[index+2] = pSquare[blockIdx.x] - pr3[0]*pr3[0]*square[index2+2];
		err[index+3] = pSquare[blockIdx.x] - pr4[0]*pr4[0]*square[index2+3];
	}
}
#endif
#if __CUDA_ARCH__ < 300
__global__ void kernelReferenceDistance(float const * const references, float const * const patterns, float const * const square, float const * const pSquare, float * const err, int const nPatterns, int const nReferences,int const mReferences) {
	volatile __shared__ float pr1[THREADSREF];
	volatile __shared__ float pr2[THREADSREF];
	volatile __shared__ float pr3[THREADSREF];
	volatile __shared__ float pr4[THREADSREF];
	register int indexPatterns = blockIdx.x * nPatterns  + threadIdx.x;
	register int indexReferences = (4*blockIdx.y) * nReferences  + threadIdx.x;
	register int indexReferences2 = indexReferences + nReferences ;//(4*blockIdx.y+1) * nReferences  + threadIdx.x;
	register int indexReferences3 = indexReferences2 + nReferences;//(4*blockIdx.y+2) * nReferences  + threadIdx.x;
	register int indexReferences4 = indexReferences3 + nReferences ;//(4*blockIdx.y+3) * nReferences  + threadIdx.x;
	register float prPartial1=0.0f;
	register float prPartial2=0.0f;
	register float prPartial3=0.0f;
	register float prPartial4=0.0f;

	if( threadIdx.x < nPatterns ) {
		float p = patterns[indexPatterns];
		float r1 = references[indexReferences];
		float r2 = references[indexReferences2];
		float r3 = references[indexReferences3];
		float r4 = references[indexReferences4];
		prPartial1 = p*r1;
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
				r1 = references[indexReferences];
				r2 = references[indexReferences2];
				r3 = references[indexReferences3];
				r4 = references[indexReferences4];
				prPartial1 += p*r1;
				prPartial2 += p*r2;
				prPartial3 += p*r3;
				prPartial4 += p*r4;
			}
		}
	}
	if(threadIdx.x < nPatterns){
		pr1[threadIdx.x]=prPartial1;
		pr2[threadIdx.x]=prPartial2;
		pr3[threadIdx.x]=prPartial3;
		pr4[threadIdx.x]=prPartial4;
	}
	else {
		pr1[threadIdx.x]=0.0f;
		pr2[threadIdx.x]=0.0f;
		pr3[threadIdx.x]=0.0f;
		pr4[threadIdx.x]=0.0f;
	}
	__syncthreads();
	if(threadIdx.x < 64 ){
		pr1[threadIdx.x]+=pr1[threadIdx.x+64];
		pr2[threadIdx.x]+=pr2[threadIdx.x+64];
		pr3[threadIdx.x]+=pr3[threadIdx.x+64];
		pr4[threadIdx.x]+=pr4[threadIdx.x+64];
	}
	__syncthreads();
	if(threadIdx.x < 32) {
		pr1[threadIdx.x] += pr1[threadIdx.x+32];
		pr1[threadIdx.x] += pr1[threadIdx.x+16];
		pr1[threadIdx.x] += pr1[threadIdx.x+8];
		pr1[threadIdx.x] += pr1[threadIdx.x+4];
		pr1[threadIdx.x] += pr1[threadIdx.x+2];
		pr1[threadIdx.x] += pr1[threadIdx.x+1];

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
			err[index]   = pSquare[blockIdx.x] - pr1[0]*pr1[0]*square[index2];
			err[index+1] = pSquare[blockIdx.x] - pr2[0]*pr2[0]*square[index2+1];
			err[index+2] = pSquare[blockIdx.x] - pr3[0]*pr3[0]*square[index2+2];
			err[index+3] = pSquare[blockIdx.x] - pr4[0]*pr4[0]*square[index2+3];
		}
	}
}
#endif

texture<float> texErr;
__global__ void cudaFindMinMaxKernel(float * in, int * out, float * max, float *min, int rSize) {
	volatile __shared__ float minShared[THREADS];
	volatile __shared__ float maxShared[THREADS];
	volatile __shared__ int idxShared[THREADS];
	float minValue = INFINITY;
	float maxValue = -INFINITY;
	int idx=0;
	for (int index = 0; index < rSize; index+=THREADS) {
		if (index + threadIdx.x < rSize) {
			float valtemp = tex1Dfetch( texErr, blockIdx.x*rSize+ index + threadIdx.x);
			if( valtemp < minValue) {
				minValue = valtemp;
				idx = index + threadIdx.x;
			}
			if ( valtemp > maxValue) {
				maxValue = valtemp;
			}
		}
	}

	maxShared[threadIdx.x] = maxValue;
	minShared[threadIdx.x] = minValue;
	idxShared[threadIdx.x] = idx;
	__syncthreads();

	if( threadIdx.x < 64 ) {
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 64] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 64];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+64];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 64] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 64];

	}
	__syncthreads();
	if ( threadIdx.x < 32) {
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 32] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 32];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+32];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 32] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 32];
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 16] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 16];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+16];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 16] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 16];
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 8] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 8];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+8];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 8] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 8];
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 4] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 4];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+4];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 4] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 4];
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 2] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 2];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+2];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 2] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 2];
		if( minShared[threadIdx.x] > minShared[threadIdx.x + 1] ) {
			minShared[threadIdx.x] = minShared[threadIdx.x + 1];
			idxShared[threadIdx.x] = idxShared[threadIdx.x+1];
		}
		if( maxShared[threadIdx.x] < maxShared[threadIdx.x + 1] )
			maxShared[threadIdx.x] = maxShared[threadIdx.x + 1];
	}

	if(threadIdx.x==0){
		out[blockIdx.x] = idxShared[0];
		min[blockIdx.x] = minShared[0];
		max[blockIdx.x] = maxShared[0];
	}
}

__global__ void kernelHistogram(float const * const in, float const * const min, float const * const max, float *const median, int nBins, int rSize) {
	extern __shared__ int bins[];
	for (int i = threadIdx.x; i < nBins; i+=blockDim.x) {
		bins[i] = 0;
	}
	__syncthreads();
	float minval = min[blockIdx.x];
	float maxval = max[blockIdx.x];
	float invBinWidth = ((float)nBins)/(maxval-minval);
	for (int index = 0; index < rSize; index+=THREADS) {
		if (index + threadIdx.x < rSize) {
			float valtemp = tex1Dfetch( texErr, blockIdx.x*rSize+ index + threadIdx.x);
			atomicAdd(bins+(int)((valtemp-minval)*invBinWidth) ,1);      
		}
	}
	__syncthreads();
	int sum=0;
	int idx=0;
	if (threadIdx.x==0) {
		while(sum<rSize/2) {
			sum+=bins[idx];
			idx++;
		}
		median[blockIdx.x] = 1.0f/(minval + ((float)idx)/invBinWidth);
	}


}



void cudaReferenceDistance(float const * const patterns, float const * const references, int * const minIndex,  int const mPatterns, int const nPatterns, int const mReferences, int const nReferences, int polarization) {

	//printf("%d %d %d\n", mPatterns, mReferences, nPatterns);

	CudaSafeCall(cudaMemcpyAsync(devReferences[polarization], references,mReferences*nReferences*sizeof(float), cudaMemcpyHostToDevice, streamRef[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devPatterns[polarization], patterns, mPatterns*nPatterns*sizeof(float), cudaMemcpyHostToDevice, streamRef[polarization]));
	CudaSafeCall( cudaBindTexture( NULL, texPatterns, devPatterns[polarization], mPatterns*nPatterns*sizeof(float)));
	CudaSafeCall( cudaBindTexture( NULL, texReferences, devReferences[polarization], mReferences*nReferences*sizeof(float)));
	CudaSafeCall(cudaBindTexture( NULL, texErr, devErr[polarization], mPatterns*mReferences*sizeof(float)));

	cudaFuncSetCacheConfig(kernelCalcInvSquare, cudaFuncCachePreferShared);
	kernelCalcInvSquare<<< mReferences, 128,0, streamRef[polarization]>>>(devReferences[polarization], devInvRSquare[polarization], mReferences, nReferences);
	cudaFuncSetCacheConfig(kernelCalcSquare, cudaFuncCachePreferShared);
	kernelCalcSquare<<< mPatterns, 128,0, streamRef[polarization]>>>(devPatterns[polarization], devPSquare[polarization], mPatterns, nPatterns);

#ifdef GF580
	cudaFuncSetCacheConfig(kernelReferenceDistance, cudaFuncCachePreferL1);
#endif //GF580
#ifdef GF680
	cudaFuncSetCacheConfig(kernelReferenceDistance, cudaFuncCachePreferShared);
#endif //GF680
	kernelReferenceDistance<<<dim3(mPatterns, mReferences/4, 1),THREADSREF,0, streamRef[polarization]>>>(devReferences[polarization], devPatterns[polarization], devInvRSquare[polarization], 
			devPSquare[polarization], devErr[polarization], nPatterns, nReferences, mReferences);

	cudaFuncSetCacheConfig(cudaFindMinMaxKernel, cudaFuncCachePreferShared);
	cudaFindMinMaxKernel<<<mPatterns, THREADS, 0, streamRef[polarization]>>>(devErr[polarization], devMinIndex[polarization], devMax[polarization], devMin[polarization], mReferences);
	CudaSafeCall(cudaMemcpyAsync(minIndex, devMinIndex[polarization], mPatterns*sizeof(int), cudaMemcpyDeviceToHost, streamRef[polarization]));
	cudaFuncSetCacheConfig(kernelHistogram, cudaFuncCachePreferShared);
	kernelHistogram<<<mPatterns, THREADS, 1536*sizeof(int), streamRef[polarization]>>>(devErr[polarization], devMin[polarization], devMax[polarization], devMedian[polarization], 1535, mReferences);


	CudaSafeCall(cudaUnbindTexture( texPatterns));
	CudaSafeCall(cudaUnbindTexture( texReferences));
	CudaSafeCall(cudaUnbindTexture( texErr ));
}
