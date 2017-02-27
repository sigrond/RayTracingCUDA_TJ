#include<cstdio>
#include<omp.h>
#include"globals.h"
#define sq(x) ((x)*(x))
//tutaj tez zrobic te optymalizacje zeby 2x nie czytac najlepiej wzgledem wymiaru Xowego
__global__ void __launch_bounds__(128) kernelGenerate(int const rSize, int const nPiiTau, float const * const pii, float const * const tau, float const * const afloat, float const * const aImag, float const * const bfloat, float const * const bImag, int const * const NmaxTable, float * const II) {
	volatile __shared__ float floatis[128];
	volatile __shared__ float imaginalis[128];
	const int Nmax = NmaxTable[blockIdx.y];
	int index = blockIdx.x * nPiiTau + threadIdx.x;
	int indexY = blockIdx.y * nPiiTau + threadIdx.x; 
	float r;
	float i;
	if(threadIdx.x < Nmax) {
		const float pi = pii[index];
		const float ta = tau[index];
		r = afloat[indexY] * pi + bfloat[indexY] * ta;
		i = aImag[indexY] * pi + bImag[indexY] * ta;
		for(int id = 128 ; id < Nmax ; id+=128) {
			if(threadIdx.x + id < Nmax) {
				index += 128; 
				indexY += 128;
				const float pi = pii[index];
				const float ta = tau[index];
				r += afloat[indexY] * pi + bfloat[indexY] * ta;
				i += aImag[indexY] * pi + bImag[indexY] * ta;
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

void cudaGenerate(int rSize, int pattern_length, int * Nmax, float * pii, int nPiiTau,  float * tau, float * afloat, float * aImag, float * bfloat, float * bImag, float * II ) {
	//double init = omp_get_wtime();
	float * devPii;
	float * devTau;
	float * devAfloat;
	float * devAImag;
	float * devBfloat;
	float * devBImag;
	float * devII;
	int * devNmax;

	cudaError(cudaSetDevice(0));
	cudaError(cudaMalloc((void**)&devPii, nPiiTau*pattern_length*sizeof(float)));
	cudaError(cudaMalloc((void**)&devTau, nPiiTau*pattern_length*sizeof(float)));
	cudaError(cudaMalloc((void**)&devAfloat, rSize*nPiiTau*sizeof(float)));
	cudaError(cudaMalloc((void**)&devBfloat, rSize*nPiiTau*sizeof(float)));
	cudaError(cudaMalloc((void**)&devAImag, rSize*nPiiTau*sizeof(float)));
	cudaError(cudaMalloc((void**)&devBImag, rSize*nPiiTau*sizeof(float)));
	cudaError(cudaMalloc((void**)&devII, rSize*pattern_length*sizeof(float)));
	cudaError(cudaMalloc((void**)&devNmax, rSize*sizeof(int)));

	cudaError(cudaMemcpy(devPii, pii, nPiiTau*pattern_length*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devTau, tau, nPiiTau*pattern_length*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devNmax, Nmax, rSize*sizeof(int), cudaMemcpyHostToDevice));

	cudaError(cudaMemcpy(devAfloat, afloat, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devBfloat, bfloat, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devAImag, aImag, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devBImag, bImag, rSize*nPiiTau*sizeof(float), cudaMemcpyHostToDevice));

	cudaFuncSetCacheConfig(kernelGenerate, cudaFuncCachePreferL1);	//TODO lepiej L1, nie wiem czemu
	//printf("%d %d\n", pattern_length, rSize);
	kernelGenerate<<<dim3(pattern_length,rSize,1), 128>>>(rSize, nPiiTau, devPii, devTau, devAfloat, devAImag, devBfloat, devBImag, devNmax, devII);

	cudaError(cudaMemcpy(II, devII, rSize*pattern_length*sizeof(float), cudaMemcpyDeviceToHost));
	cudaError(cudaFree(devPii));
	cudaError(cudaFree(devTau));
	cudaError(cudaFree(devNmax));

	cudaError(cudaFree(devAfloat));
	cudaError(cudaFree(devAImag));
	cudaError(cudaFree(devBfloat));
	cudaError(cudaFree(devBImag));
	cudaError(cudaFree(devII));
	//printf("%f C Z A S CUDA\n", omp_get_wtime()-init);
}
#undef sq



