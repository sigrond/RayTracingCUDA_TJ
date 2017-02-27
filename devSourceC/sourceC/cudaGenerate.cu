#include<cstdio>
#include<omp.h>
#include"globals.h"
#define sq(x) ((x)*(x))
//tutaj tez zrobic te optymalizacje zeby 2x nie czytac najlepiej wzgledem wymiaru Xowego
__global__ void __launch_bounds__(128) kernelGenerate(int const rSize, int const nPiiTau, real const * const pii, real const * const tau, real const * const aReal, real const * const aImag, real const * const bReal, real const * const bImag, int const * const NmaxTable, real * const II) {
	volatile __shared__ real realis[128];
	volatile __shared__ real imaginalis[128];
	const int Nmax = NmaxTable[blockIdx.y];
	int index = blockIdx.x * nPiiTau + threadIdx.x;
	int indexY = blockIdx.y * nPiiTau + threadIdx.x; 
	real r;
	real i;
	if(threadIdx.x < Nmax) {
		const real pi = pii[index];
		const real ta = tau[index];
		r = aReal[indexY] * pi + bReal[indexY] * ta;
		i = aImag[indexY] * pi + bImag[indexY] * ta;
		for(int id = 128 ; id < Nmax ; id+=128) {
			if(threadIdx.x + id < Nmax) {
				index += 128; 
				indexY += 128;
				const real pi = pii[index];
				const real ta = tau[index];
				r += aReal[indexY] * pi + bReal[indexY] * ta;
				i += aImag[indexY] * pi + bImag[indexY] * ta;
			}
		}
	}
	
	if ( threadIdx.x < Nmax ) {
		realis[threadIdx.x] = r;
		imaginalis[threadIdx.x] = i;
	}
	else {
		realis[threadIdx.x]=0.0f;
		imaginalis[threadIdx.x]=0.0f;
	}
	__syncthreads();
	if(threadIdx.x < 64 ) {
		realis[threadIdx.x]+=realis[threadIdx.x+64];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+64];
	}
	__syncthreads();
	if(threadIdx.x < 32) {
		realis[threadIdx.x]+=realis[threadIdx.x+32];
		realis[threadIdx.x]+=realis[threadIdx.x+16];
		realis[threadIdx.x]+=realis[threadIdx.x+8];
		realis[threadIdx.x]+=realis[threadIdx.x+4];
		realis[threadIdx.x]+=realis[threadIdx.x+2];
		realis[threadIdx.x]+=realis[threadIdx.x+1];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+32];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+16];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+8];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+4];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+2];
		imaginalis[threadIdx.x]+=imaginalis[threadIdx.x+1];
		if(threadIdx.x==0)
			II[blockIdx.x + blockIdx.y*gridDim.x] = sq(realis[0]) + sq(imaginalis[0]);
	}
}

void cudaGenerate(int rSize, int pattern_length, int * Nmax, real * pii, int nPiiTau,  real * tau, real * aReal, real * aImag, real * bReal, real * bImag, real * II ) {
	//double init = omp_get_wtime();
	real * devPii;
	real * devTau;
	real * devAReal;
	real * devAImag;
	real * devBReal;
	real * devBImag;
	real * devII;
	int * devNmax;

	cudaError(cudaSetDevice(0));
	cudaError(cudaMalloc((void**)&devPii, nPiiTau*pattern_length*sizeof(real)));
	cudaError(cudaMalloc((void**)&devTau, nPiiTau*pattern_length*sizeof(real)));
	cudaError(cudaMalloc((void**)&devAReal, rSize*nPiiTau*sizeof(real)));
	cudaError(cudaMalloc((void**)&devBReal, rSize*nPiiTau*sizeof(real)));
	cudaError(cudaMalloc((void**)&devAImag, rSize*nPiiTau*sizeof(real)));
	cudaError(cudaMalloc((void**)&devBImag, rSize*nPiiTau*sizeof(real)));
	cudaError(cudaMalloc((void**)&devII, rSize*pattern_length*sizeof(real)));
	cudaError(cudaMalloc((void**)&devNmax, rSize*sizeof(int)));

	cudaError(cudaMemcpy(devPii, pii, nPiiTau*pattern_length*sizeof(real), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devTau, tau, nPiiTau*pattern_length*sizeof(real), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devNmax, Nmax, rSize*sizeof(int), cudaMemcpyHostToDevice));

	cudaError(cudaMemcpy(devAReal, aReal, rSize*nPiiTau*sizeof(real), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devBReal, bReal, rSize*nPiiTau*sizeof(real), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devAImag, aImag, rSize*nPiiTau*sizeof(real), cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(devBImag, bImag, rSize*nPiiTau*sizeof(real), cudaMemcpyHostToDevice));

	cudaFuncSetCacheConfig(kernelGenerate, cudaFuncCachePreferL1);	//TODO lepiej L1, nie wiem czemu
	//printf("%d %d\n", pattern_length, rSize);
	kernelGenerate<<<dim3(pattern_length,rSize,1), 128>>>(rSize, nPiiTau, devPii, devTau, devAReal, devAImag, devBReal, devBImag, devNmax, devII);

	cudaError(cudaMemcpy(II, devII, rSize*pattern_length*sizeof(real), cudaMemcpyDeviceToHost));
	cudaError(cudaFree(devPii));
	cudaError(cudaFree(devTau));
	cudaError(cudaFree(devNmax));

	cudaError(cudaFree(devAReal));
	cudaError(cudaFree(devAImag));
	cudaError(cudaFree(devBReal));
	cudaError(cudaFree(devBImag));
	cudaError(cudaFree(devII));
	//printf("%f C Z A S CUDA\n", omp_get_wtime()-init);
}
#undef sq



