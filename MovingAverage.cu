/** \file MovingAverage.cu
 * \author Tomasz Jakubczyk
 * \brief plik g³ówny funkcji licz¹cej œredni¹ krocz¹c¹
 *
 *
 *
 */

#define WIN32
#include "mex.h"
#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include "MovingAverage_CUDA_kernel.cuh"

__host__
//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__host__
/** \brief compute grid and thread block size for a given number of elements
 *
 * \param n uint
 * \param blockSize uint
 * \param numBlocks uint&
 * \param numThreads uint&
 * \return void
 *
 */
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

/** \brief wyg³adza Theta i I
 * function [sI] = MovingAverage(I, I_S, step)
 * \param nlhs int
 * \param plhs[] mxArray*
 * \param nrhs int
 * \param prhs[] const mxArray*
 * \return void
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float* I;/**< skorygowana klatka */
    unsigned int I_size;
    float* I_S;/**< indeksy posortowanej klatki */
    unsigned int I_S_size;
    float* step;

    /**< sprawdzanie argumentów */
    if(nlhs!=1)
    {
    	printf("function returns [I] \n");
    	return;
    }
    if(nrhs!=3)
    {
        printf("function arguments are (I, I_S, step) \n");
        return;
    }
    if(!mxIsSingle(prhs[0]))
    {
        printf("1st argument needs to be single precision vector\n");
        return;
    }

    if(!mxIsSingle(prhs[1]))
    {
        printf("2nd argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[2]))
    {
        printf("3rd argument needs to be single precision number\n");
        return;
    }

    /**< pobranie argumentów z matlaba */
    I=(float*)mxGetPr(prhs[0]);
    I_size=mxGetN(prhs[0])*mxGetM(prhs[0]);
    I_S=(float*)mxGetPr(prhs[1]);
    I_S_size=mxGetN(prhs[1])*mxGetM(prhs[1]);
    step=(float*)mxGetPr(prhs[2]);
    if(mxGetN(prhs[2])*mxGetM(prhs[2])!=1)
    {
        printf("3rd argument (step) must be a number\n");
        return;
    }

    float* dev_I=NULL;
    float* dev_I_S=NULL;
    float* dev_sI=NULL;

    cudaError_t err;
    checkCudaErrors(cudaMalloc((void**)&dev_I, sizeof(float)*I_size));
    checkCudaErrors(cudaMemcpy((void*)dev_I, I, sizeof(float)*I_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dev_I_S, sizeof(float)*I_S_size));
    checkCudaErrors(cudaMemcpy((void*)dev_I_S, I_S, sizeof(float)*I_S_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dev_sI, sizeof(float)*I_size));
    checkCudaErrors(cudaMemset(dev_sI,0,sizeof(float)*I_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc,Memcpy): %s\n", cudaGetErrorString(err));
    }

    uint numThreads, numBlocks;
    computeGridSize(I_size, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);

    MovingAverageD<<< dimGrid, numThreads >>>(dev_I,I_size,dev_I_S,dev_sI,*step);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(MovingAverageD): %s\n", cudaGetErrorString(err));
    }

    DivD<<< dimGrid, numThreads >>>(I_size,dev_sI,*step);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(DivD): %s\n", cudaGetErrorString(err));
    }

    int dimssI[1]={(int)I_size};
    plhs[0]=mxCreateNumericArray(1,dimssI,mxSINGLE_CLASS,mxREAL);
    float* sI=(float*)mxGetPr(plhs[0]);

    checkCudaErrors(cudaMemcpy((void*)sI,dev_sI,sizeof(float)*I_size,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaFree(dev_sI));
    checkCudaErrors(cudaFree(dev_I_S));
    checkCudaErrors(cudaFree(dev_I));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
}
