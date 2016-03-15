/** \file IntensCalc_CUDA.cu
 * \author Tomasz Jakubczyk
 * \brief plik z implementacjami funkcji wywołujących CUDA'ę
 *
 *
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include "IntensCalc_CUDA_kernel.cuh"

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

cudaError_t err;
char* dev_buff=NULL;
unsigned short* dev_frame=NULL;

extern "C"
{

void setupCUDA_IC()
{
    /**< przygotowanie CUDA'y */

    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_buff, sizeof(char)*640*480*2));
    checkCudaErrors(cudaMalloc((void**)&dev_frame, sizeof(unsigned short)*640*480));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_buff,0,sizeof(char)*640*480*2));
    checkCudaErrors(cudaMemset(dev_frame,0,sizeof(unsigned short)*640*480));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
}

void copyBuff(char* buff)
{
    /**< kopiujemy na kartę */
    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemcpy((void*)dev_buff, buff, sizeof(char)*640*480*2, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
    }
}

void doIC(float* I_Red, float* I_Green, float* I_Blue)
{
    uint numThreads, numBlocks;
    computeGridSize(640*480, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);

    /**< Jeśli tutaj będzie działało za wolno, to można wykozystać dodatkowy wątek CPU i CUDA streams */
    aviGetValueD<<< dimGrid, numThreads >>>(dev_buff,dev_frame,640*480);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(aviGetValueD): %s\n", cudaGetErrorString(err));
    }
    unsigned short int klatka[307200];
    checkCudaErrors(cudaMemcpy((void*)klatka,dev_frame,sizeof(unsigned short)*640*480,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
    /*for(int i=0;i<480;i++)
    {
        for(int j=0;j<640;j++)
        {
            printf("%d ",klatka[i*640+j]);
        }
        printf("\n");
    }*/
}

void freeCUDA_IC()
{
    checkCudaErrors(cudaFree(dev_buff));
    checkCudaErrors(cudaFree(dev_frame));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
}

}
