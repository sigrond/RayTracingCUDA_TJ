#include<cstdio>
#include<omp.h>
#include"globals.h"
#include"cudaGlobals.h"
#include"cudaError.h"
#define sq(x) ((x)*(x))
//tutaj tez zrobic te optymalizacje zeby 2x nie czytac najlepiej wzgledem wymiaru Xowego
#define THREADS 128


texture<float> texPii;
texture<float> texTau;
texture<float> texafloat;
texture<float> texbfloat;
texture<float> texaImag;
texture<float> texbImag;

#if __CUDA_ARCH__ >= 300
__global__ void kernelGenerate(int const rSize, int const nPiiTau, float const * const pii, float const * const tau, 
		float const * const afloat, float const * const aImag, float const * const bfloat, float const * const bImag,
		int const * const NmaxTable, float * const II) {
	volatile __shared__ float floatis[4];
	volatile __shared__ float imaginalis[4];
	const int Nmax = NmaxTable[blockIdx.y];
	int index  = blockIdx.x * nPiiTau + threadIdx.x;
	int indexY = blockIdx.y * nPiiTau + threadIdx.x;
	float r=0.0f;
	float i=0.0f;
	if(threadIdx.x < Nmax) {
		//const float pi = pii[index];
		//const float ta = tau[index];
		const float pi = tex1Dfetch(texPii, index);
		const float ta = tex1Dfetch(texTau, index);
		//r = afloat[indexY] * pi + bfloat[indexY] * ta;
		//i = aImag[indexY] * pi + bImag[indexY] * ta;
		r = tex1Dfetch(texafloat,indexY) * pi + tex1Dfetch(texbfloat,indexY) * ta;
		i = tex1Dfetch(texaImag,indexY) * pi + tex1Dfetch(texbImag,indexY) * ta;
		for(int id = THREADS ; id < Nmax ; id+=THREADS) {
			if(threadIdx.x + id < Nmax) {
				index += THREADS; 
				indexY += THREADS;
				//const float pi = pii[index];
				//const float ta = tau[index];
				const float pi = tex1Dfetch(texPii, index);
				const float ta = tex1Dfetch(texTau, index);
				//r += afloat[indexY] * pi + bfloat[indexY] * ta;
				//i += aImag[indexY] * pi + bImag[indexY] * ta;
				r += tex1Dfetch(texafloat,indexY) * pi + tex1Dfetch(texbfloat,indexY) * ta;
				i += tex1Dfetch(texaImag,indexY) * pi + tex1Dfetch(texbImag,indexY) * ta;
			}
		}
	}

	//butterfly reduction across warp
	for (int j=16; j>=1; j/=2) {
		r += __shfl_xor(r , j, 32);
		i += __shfl_xor(i , j, 32);
	}
	//further reduction across block
	if(threadIdx.x % 32 == 0) {
		floatis[threadIdx.x>>5] = r;
		imaginalis[threadIdx.x>>5] = i;
	}
	__syncthreads();
	if(threadIdx.x <2) {
		floatis[threadIdx.x] += floatis[threadIdx.x+2];
		imaginalis[threadIdx.x] += imaginalis[threadIdx.x+2];
		floatis[threadIdx.x] += floatis[threadIdx.x+1];
		imaginalis[threadIdx.x] += imaginalis[threadIdx.x+1];
	}

	if(threadIdx.x==0)
		II[blockIdx.x + blockIdx.y*gridDim.x] = sq(floatis[0]) + sq(imaginalis[0]);
}
#endif
#if __CUDA_ARCH__ < 300
__global__ void kernelGenerate(int const rSize, int const nPiiTau, float const * const pii, float const * const tau, float const * const afloat, float const * const aImag, float const * const bfloat, float const * const bImag, int const * const NmaxTable, float * const II) {
	volatile __shared__ float floatis[THREADS];
	volatile __shared__ float imaginalis[THREADS];
	const int Nmax = NmaxTable[blockIdx.y];
	int index = blockIdx.x * nPiiTau + threadIdx.x;
	int indexY = blockIdx.y * nPiiTau + threadIdx.x; 
	float r=0.0f;
	float i=0.0f;
	if(threadIdx.x < Nmax) {
		const float pi = pii[index];
		const float ta = tau[index];
		//const float pi = tex1Dfetch(texPii, index);
		//const float ta = tex1Dfetch(texTau, index);
		r = afloat[indexY] * pi + bfloat[indexY] * ta;
		i = aImag[indexY] * pi + bImag[indexY] * ta;
		//r = tex1Dfetch(texafloat,indexY) * pi + tex1Dfetch(texbfloat,indexY) * ta;
		//i = tex1Dfetch(texaImag,indexY) * pi + tex1Dfetch(texbImag,indexY) * ta;
		for(int id = THREADS ; id < Nmax ; id+=THREADS) {
			if(threadIdx.x + id < Nmax) {
				index += THREADS; 
				indexY += THREADS;
				const float pi = pii[index];
				const float ta = tau[index];
				//const float pi = tex1Dfetch(texPii, index);
				//const float ta = tex1Dfetch(texTau, index);
				r += afloat[indexY] * pi + bfloat[indexY] * ta;
				i += aImag[indexY] * pi + bImag[indexY] * ta;
				//r += tex1Dfetch(texafloat,indexY) * pi + tex1Dfetch(texbfloat,indexY) * ta;
				//i += tex1Dfetch(texaImag,indexY) * pi + tex1Dfetch(texbImag,indexY) * ta;
			}
		}
	}
	if ( threadIdx.x < Nmax ) {
		floatis[threadIdx.x] = r;
		imaginalis[threadIdx.x] = i;
	}
	else {
		floatis[threadIdx.x]=0.0f;
		imaginalis[threadIdx.x]=0.0f;
	}
	__syncthreads();
	if(threadIdx.x < 64 ) {
		floatis[threadIdx.x]+=floatis[threadIdx.x+64];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+64];
	}
	__syncthreads();
	if(threadIdx.x < 32) {
		floatis[threadIdx.x]+=floatis[threadIdx.x+32];
		floatis[threadIdx.x]+=floatis[threadIdx.x+16];
		floatis[threadIdx.x]+=floatis[threadIdx.x+8];
		floatis[threadIdx.x]+=floatis[threadIdx.x+4];
		floatis[threadIdx.x]+=floatis[threadIdx.x+2];
		floatis[threadIdx.x]+=floatis[threadIdx.x+1];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+32];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+16];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+8];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+4];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+2];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+1];
		if(threadIdx.x==0)
			II[blockIdx.x + blockIdx.y*gridDim.x] = sq(floatis[0]) + sq(imaginalis[0]);
	}
}


#endif

void cudaGenerate(int rSize, int pattern_length, int * Nmax, float * pii, int nPiiTau,  float * tau, float * afloat, float * aImag, float * bfloat, float * bImag, float * II, int polarization ) {
	CudaSafeCall(cudaStreamCreate(&stream[polarization]));
	CudaSafeCall(cudaMalloc((void**)&devPii[polarization]  , nPiiTau*pattern_length*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devTau[polarization]  , nPiiTau*pattern_length*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devAfloat[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devBfloat[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devAImag[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devBImag[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devII[polarization]   , rSize*pattern_length*sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&devNmax[polarization] , rSize*sizeof(int)));

	CudaSafeCall(cudaMemcpyAsync(devPii[polarization]  , pii  , nPiiTau*pattern_length*sizeof(float) , cudaMemcpyHostToDevice , stream[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devTau[polarization]  , tau  , nPiiTau*pattern_length*sizeof(float) , cudaMemcpyHostToDevice , stream[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devNmax[polarization] , Nmax , rSize*sizeof(int)                   , cudaMemcpyHostToDevice , stream[polarization]));
	CudaSafeCall( cudaBindTexture( NULL, texPii, devPii[polarization], nPiiTau*pattern_length*sizeof(float)));
	CudaSafeCall( cudaBindTexture( NULL, texTau, devTau[polarization], nPiiTau*pattern_length*sizeof(float)));

	CudaSafeCall(cudaMemcpyAsync(devAfloat[polarization], afloat, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice, stream[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devBfloat[polarization], bfloat, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice, stream[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devAImag[polarization], aImag, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice, stream[polarization]));
	CudaSafeCall(cudaMemcpyAsync(devBImag[polarization], bImag, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice, stream[polarization]));
	CudaSafeCall( cudaBindTexture( NULL, texafloat, devAfloat[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall( cudaBindTexture( NULL, texbfloat, devBfloat[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall( cudaBindTexture( NULL, texaImag, devAImag[polarization], rSize*nPiiTau*sizeof(float)));
	CudaSafeCall( cudaBindTexture( NULL, texbImag, devBImag[polarization], rSize*nPiiTau*sizeof(float)));

#ifdef GF580
	cudaFuncSetCacheConfig(kernelGenerate, cudaFuncCachePreferL1);
#endif //GF580
#ifdef GF680
	cudaFuncSetCacheConfig(kernelGenerate, cudaFuncCachePreferShared);
#endif //GF680
	kernelGenerate<<<dim3(pattern_length,rSize,1), THREADS, 0, stream[polarization]>>>(rSize, nPiiTau, devPii[polarization], devTau[polarization],
			devAfloat[polarization], devAImag[polarization], devBfloat[polarization], devBImag[polarization],
			devNmax[polarization], devII[polarization]);

	CudaSafeCall(cudaMemcpyAsync(II, devII[polarization], rSize*pattern_length*sizeof(float), cudaMemcpyDeviceToHost, stream[polarization]));
	CudaSafeCall(cudaUnbindTexture( texPii));
	CudaSafeCall(cudaUnbindTexture( texTau));
	CudaSafeCall(cudaUnbindTexture( texafloat));
	CudaSafeCall(cudaUnbindTexture( texbfloat));
	CudaSafeCall(cudaUnbindTexture( texbImag));
	CudaSafeCall(cudaUnbindTexture( texaImag));
}
#undef sq



