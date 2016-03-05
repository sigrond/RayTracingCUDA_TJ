/** \file ReducedMean.cu
 * \author Tomasz Jakubczyk
 * \brief liczenie œrednich dla klatki przy zredukowanej liczbie punktów
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
#include "ReducedMean_CUDA_kernel.cuh"

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

/** \brief
 * function [nTheta, I_] = ReducedMean(Theta_S, deltaT, I, I_S)
 * \param nlhs int
 * \param plhs[] mxArray*
 * \param nrhs int
 * \param prhs[] const mxArray*
 * \return void
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float* Theta_S;/**< posortowany wektor theta */
    unsigned int Theta_S_size;
    float* deltaT;/**< sta³a delta */
    float* I;/**< skorygowana klatka */
    unsigned int I_size;
    float* I_S;/**< indeksy posortowanej klatki */
    unsigned int I_S_size;

    /**< sprawdzanie argumentów */
    if(nlhs!=2)
    {
    	printf("function returns [nTheta, I_] \n");
    	return;
    }
    if(nrhs!=4)
    {
        printf("function arguments are (Theta_S, deltaT, I, I_S) \n");
        return;
    }
    if(!mxIsSingle(prhs[0]))
    {
        printf("1st argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[1]))
    {
        printf("2nd argument needs to be single precision number\n");
        return;
    }
    if(!mxIsSingle(prhs[2]))
    {
        printf("3rd argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[3]))
    {
        printf("4th argument needs to be single precision vector\n");
        return;
    }

    /**< pobranie argumentów z matlaba */
    Theta_S=(float*)mxGetPr(prhs[0]);
    Theta_S_size=mxGetN(prhs[0])*mxGetM(prhs[0]);
    deltaT=(float*)mxGetPr(prhs[1]);
    if(mxGetN(prhs[1])*mxGetM(prhs[1])!=1)
    {
        printf("2nd argument (deltaT) must be a number\n");
        return;
    }
    I=(float*)mxGetPr(prhs[2]);
    I_size=mxGetN(prhs[2])*mxGetM(prhs[2]);
    I_S=(float*)mxGetPr(prhs[3]);
    I_S_size=mxGetN(prhs[3])*mxGetM(prhs[3]);

    /**< dla każdego punktu z theta_S odpalamy osobny kernel */
    /**< w każdym kerneluętla nom 1:nom_max */
    /**< w pętli sprawdzamy czy w tym punkcie spełniony jest warunek na ind */
    /**< jeśli jest spełniony to dodajemy odpowiednie wartości theta i I, oraz zwiększamy licznik */
    /**< po zakończeniu wyliczamy średnią */
    unsigned int max_nom=floor((Theta_S[Theta_S_size-1]-Theta_S[0])/(float)*deltaT);
    float* dev_Theta_S=NULL;
    float* dev_I=NULL;
    float* dev_I_S=NULL;
    float* dev_nTheta=NULL;
    float* dev_nI=NULL;
    float* dev_counter=NULL;
    cudaError_t err;
    checkCudaErrors(cudaMalloc((void**)&dev_Theta_S, sizeof(float)*Theta_S_size));
    checkCudaErrors(cudaMemcpy((void*)dev_Theta_S, Theta_S, sizeof(float)*Theta_S_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dev_I, sizeof(float)*I_size));
    checkCudaErrors(cudaMemcpy((void*)dev_I, I, sizeof(float)*I_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dev_I_S, sizeof(float)*I_S_size));
    checkCudaErrors(cudaMemcpy((void*)dev_I_S, I_S, sizeof(float)*I_S_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dev_nTheta, sizeof(float)*max_nom));
    checkCudaErrors(cudaMemset(dev_nTheta,0,sizeof(float)*max_nom));
    checkCudaErrors(cudaMalloc((void**)&dev_nI, sizeof(float)*max_nom));
    checkCudaErrors(cudaMemset(dev_nI,0,sizeof(float)*max_nom));
    checkCudaErrors(cudaMalloc((void**)&dev_counter, sizeof(float)*max_nom));
    checkCudaErrors(cudaMemset(dev_counter,0,sizeof(float)*max_nom));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc,Memcpy): %s\n", cudaGetErrorString(err));
    }
    uint numThreads, numBlocks;
    computeGridSize(Theta_S_size, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);

    ReducedMeanD<<< dimGrid, numThreads >>>(dev_Theta_S,Theta_S_size,(float)*deltaT,max_nom,dev_I,dev_I_S,dev_nTheta,dev_nI,dev_counter);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(ReducedMeanD): %s\n", cudaGetErrorString(err));
    }

    int dimsnTheta[1]={(int)max_nom};
    plhs[0]=mxCreateNumericArray(1,dimsnTheta,mxSINGLE_CLASS,mxREAL);
    float* nTheta=(float*)mxGetPr(plhs[0]);
    int dimsnI[1]={(int)max_nom};
    plhs[1]=mxCreateNumericArray(1,dimsnI,mxSINGLE_CLASS,mxREAL);
    float* nI=(float*)mxGetPr(plhs[1]);
    float* counter=(float*)malloc(sizeof(float)*max_nom);

    checkCudaErrors(cudaMemcpy((void*)nTheta,dev_nTheta,sizeof(float)*max_nom,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)nI,dev_nI,sizeof(float)*max_nom,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)counter,dev_counter,sizeof(float)*max_nom,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }

    /*printf("\nnTheta: ");
    for(int i=0,j=0;i<max_nom && j<100;i++)
    {
        if(i>0 && nTheta[i]==0.0f)
        {
            if(nTheta[i-1]==0.0f)
            continue;
        }
        printf("%f ",nTheta[i]);
        j++;
    }
    printf("\n");

    printf("\nnI: ");
    for(int i=0,j=0;i<max_nom && j<100;i++)
    {
        if(i>0 && nI[i]==0.0f)
        {
            if(nI[i-1]==0.0f)
            continue;
        }
        printf("%f ",nI[i]);
        j++;
    }
    printf("\n");

    printf("\ncounter: ");
    for(int i=0,j=0;i<max_nom && j<100;i++)
    {
        if(i>0 && counter[i]==0.0f)
        {
            if(counter[i-1]==0.0f)
            continue;
        }
        printf("%f ",counter[i]);
        j++;
    }
    printf("\n");*/

    for(int i=0;i<max_nom;i++)
    {
        nTheta[i]/=counter[i];
        nI[i]/=counter[i];
    }

    free(counter);
    checkCudaErrors(cudaFree(dev_counter));
    checkCudaErrors(cudaFree(dev_nI));
    checkCudaErrors(cudaFree(dev_nTheta));
    checkCudaErrors(cudaFree(dev_I_S));
    checkCudaErrors(cudaFree(dev_I));
    checkCudaErrors(cudaFree(dev_Theta_S));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
}
