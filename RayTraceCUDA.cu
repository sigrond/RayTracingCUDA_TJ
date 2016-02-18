/** \file RayTraceCUDA.cu
 * \author Tomasz Jakubczyk
 * \brief Implementation of RayTrace function
 * which calls RayTraceD CUDA kernels
 */
#define WIN32
#include "HandlesStructures.cuh"
#include "RayTraceCUDA_kernel.cuh"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include <stdlib.h>
#include<stdio.h>
#include "mex.h"

extern "C"
{
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

    __host__
    /** \brief calculate ray tracing.
     * call RayTraceD CUDA kernels
     * \param Br float*
     * \param Vb int*
     * \param VH float*
     * \param Vb_length int
     * \param VH_length int
     * \param S HandlesStructures
     * \param IM float3* image. must be zeroed before ray tracing.
     * to skip image calculation set pointer to NULL
     * \param P float3*
     * \return void
     *
     */
    void RayTrace(float* Br, int Br_size, float* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float* IC, int IC_size, float* PX)
    {
        cudaError_t err;
        float* dev_Br=0;
        float* dev_Vb=0;
        float* dev_VH=0;
        float* dev_IC=0;
        float* dev_PX=0;
        checkCudaErrors(cudaMalloc((void**)&dev_Br, sizeof(float)*Br_size));
        err = cudaGetLastError();
        if (err != cudaSuccess)
		{
			printf("cudaError(cudaMalloc((void**)&dev_Br, sizeof(float)*Br_size)): %s\n", cudaGetErrorString(err));
		}
        checkCudaErrors(cudaMemcpy((void*)dev_Br, Br, sizeof(float)*Br_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&dev_Vb, sizeof(float)*Vb_length));
        checkCudaErrors(cudaMemcpy((void*)dev_Vb, Vb, sizeof(float)*Vb_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&dev_VH, sizeof(float)*VH_length));
        checkCudaErrors(cudaMemcpy((void*)dev_VH, VH, sizeof(float)*VH_length, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&dev_IC, sizeof(float)*IC_size));
		checkCudaErrors(cudaMemset(dev_IC,0,sizeof(float)*IC_size));
        checkCudaErrors(cudaMalloc((void**)&dev_PX, sizeof(float)*4*480*640));
        checkCudaErrors(cudaMemset(dev_PX,0,sizeof(float)*4*480*640));

        uint numThreads, numBlocks;
        computeGridSize(VH_length*Vb_length, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
        dim3 dimGrid(dimGridX,dimGridY);

        err = cudaGetLastError();
        if (err != cudaSuccess)
		{
			printf("1cudaError(while GPU memory allocation): %s\n", cudaGetErrorString(err));
		}

        //system("pause");
        printf("dev_IC:%d\n",dev_IC);

        printf("numBlocks: %d\n",numBlocks);
        printf("numThreads: %d\n",numThreads);
        printf("dimGrid.x: %d\n",dimGrid.x);
        printf("dimGrid.y: %d\n",dimGrid.y);

        RayTraceD<<< dimGrid, numThreads >>>(dev_Br,dev_Vb,dev_VH,Vb_length,VH_length,S,dev_IC,dev_PX);
        //RayTraceD<<< 2, 25 >>>(dev_Br,dev_Vb,dev_VH,Vb_length,VH_length,S,dev_IM,dev_P);

        err = cudaGetLastError();
        if (err != cudaSuccess)
		{
			printf("2cudaError(while CUDA kernel execution): %s\n", cudaGetErrorString(err));
		}

        checkCudaErrors(cudaMemcpy((void*)PX,dev_PX,sizeof(float)*4*480*640,cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void*)IC,dev_IC,sizeof(float)*IC_size,cudaMemcpyDeviceToHost));

		err = cudaGetLastError();
        if (err != cudaSuccess)
		{
			printf("3cudaError(while cudaMemcpy): %s\n", cudaGetErrorString(err));
		}

		checkCudaErrors(cudaFree(dev_IC));

        checkCudaErrors(cudaFree(dev_PX));
        checkCudaErrors(cudaFree(dev_VH));
        checkCudaErrors(cudaFree(dev_Vb));
        checkCudaErrors(cudaFree(dev_Br));
    }
}
