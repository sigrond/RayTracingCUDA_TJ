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
    void RayTrace(float* Br, int Br_size, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, int IM_size, float3* P)
    {
        float3* dev_Br=0;
        int* dev_Vb=0;
        float* dev_VH=0;
        float* dev_IM=0;
        float3* dev_P=0;
        checkCudaErrors(cudaMalloc((void**)&dev_Br, sizeof(float)*Br_size));
        checkCudaErrors(cudaMemcpy((void*)dev_Br, Br, sizeof(float)*Br_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&dev_Vb, sizeof(int)*Vb_length));
        checkCudaErrors(cudaMemcpy((void*)dev_Vb, Vb, sizeof(int)*Vb_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&dev_VH, sizeof(float)*VH_length));
        checkCudaErrors(cudaMemcpy((void*)dev_VH, VH, sizeof(float)*VH_length, cudaMemcpyHostToDevice));
        if(IM!=NULL && IM_size>0)
        {
            checkCudaErrors(cudaMalloc((void**)&dev_IM, sizeof(float)*IM_size));
            checkCudaErrors(cudaMemset(dev_IM,0,sizeof(float)*IM_size));
        }
        checkCudaErrors(cudaMalloc((void**)&dev_P, sizeof(float)*VH_length*Vb_length*3*7));

        uint numThreads, numBlocks;
        computeGridSize(VH_length*Vb_length, 512, numBlocks, numThreads);
        //system("pause");
        printf("dev_IM:%d\n",dev_IM);
        RayTraceD<<< numBlocks, numThreads >>>(dev_Br,dev_Vb,dev_VH,Vb_length,VH_length,S,dev_IM,dev_P);

        checkCudaErrors(cudaMemcpy((void*)P,dev_P,sizeof(float)*VH_length*Vb_length*3*7,cudaMemcpyDeviceToHost));
        if(IM!=NULL && IM_size>0)
        {
            checkCudaErrors(cudaMemcpy((void*)IM,dev_IM,sizeof(float)*IM_size,cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(dev_IM));
        }

        checkCudaErrors(cudaFree(dev_P));
        checkCudaErrors(cudaFree(dev_VH));
        checkCudaErrors(cudaFree(dev_Vb));
        checkCudaErrors(cudaFree(dev_Br));
    }
}
