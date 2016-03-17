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
#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "IntensCalc_CUDA_kernel.cuh"
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

cudaError_t err;
char* dev_buff=NULL;
unsigned short* dev_frame=NULL;
short* dev_outArray=NULL;

int* dev_ipR=NULL;
int ipR_Size=0;
int* dev_ipG=NULL;
int ipG_Size=0;
int* dev_ipB=NULL;
int ipB_Size=0;
float* dev_ICR_N=NULL;
float* dev_ICG_N=NULL;
float* dev_ICB_N=NULL;
int* dev_I_S_R=NULL;
int* dev_I_S_G=NULL;
int* dev_I_S_B=NULL;

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
    checkCudaErrors(cudaMalloc((void**)&dev_outArray, sizeof(short)*640*480*3));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_buff,0,sizeof(char)*640*480*2));
    checkCudaErrors(cudaMemset(dev_frame,0,sizeof(unsigned short)*640*480));
    checkCudaErrors(cudaMemset(dev_outArray,0,sizeof(short)*640*480*3));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
}

void setMasksAndImagesAndSortedIndexes(
    int* ipR,int ipR_size,int* ipG,int ipG_size,int* ipB, int ipB_size,
    float* ICR_N, float* ICG_N, float* ICB_N,
    int* I_S_R, int* I_S_G, int* I_S_B)
{
    ipR_Size=ipR_size;
    ipG_Size=ipG_size;
    ipB_Size=ipB_size;
    checkCudaErrors(cudaMalloc((void**)&dev_ipR, sizeof(int)*ipR_size));
    checkCudaErrors(cudaMalloc((void**)&dev_ipG, sizeof(int)*ipG_size));
    checkCudaErrors(cudaMalloc((void**)&dev_ipB, sizeof(int)*ipB_size));
    checkCudaErrors(cudaMalloc((void**)&dev_ICR_N, sizeof(float)*ipR_size));
    checkCudaErrors(cudaMalloc((void**)&dev_ICG_N, sizeof(float)*ipG_size));
    checkCudaErrors(cudaMalloc((void**)&dev_ICB_N, sizeof(float)*ipB_size));
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

    demosaicD<<< dimGrid, numThreads >>>(dev_frame,640*480,dev_outArray);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(demosaicD): %s\n", cudaGetErrorString(err));
    }

    unsigned short int klatka[307200];
    checkCudaErrors(cudaMemcpy((void*)klatka,dev_frame,sizeof(unsigned short)*640*480,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
}

void freeCUDA_IC()
{
    checkCudaErrors(cudaFree(dev_buff));
    checkCudaErrors(cudaFree(dev_frame));
    checkCudaErrors(cudaFree(dev_outArray));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
    cudaProfilerStop();
}

}