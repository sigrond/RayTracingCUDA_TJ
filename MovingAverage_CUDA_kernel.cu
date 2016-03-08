/** \file MovingAverage_CUDA_kernel.cu
 * \author Tomasz Jakubczyk
 * \brief plik z kernelem CUDA wyg³adzaj¹cym za pomoc¹ œredniej krocz¹cej
 *
 *
 *
 */

#define WIN32
#include<stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include "math_constants.h"

#define STEP 64

extern "C"
{

__global__
void MovingAverageD(float* Theta_S,unsigned int Theta_S_size, float* I, float* I_S, float* sTheta, float* sI)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=Theta_S_size)
        return;
    float value;
    float* val0;

    #pragma unroll
    for(unsigned int i=0;i<STEP && index+i<Theta_S_size;i++)
    {
        val0=sTheta+index+i;
        value=Theta_S[index];
        atomicAdd(val0, value);
        val0=sI+index+i;
        value=I[(unsigned int)round(I_S[index]-1.0f)];
        atomicAdd(val0, value);
    }
}

__global__
void DivD(unsigned int Theta_S_size, float* sTheta, float* sI)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=Theta_S_size)
        return;
    if(index>=STEP-1)
    {
        sTheta[index]/=(float)STEP;
        sI[index]/=(float)STEP;
    }
    else
    {
        sTheta[index]/=(float)(index+1);
        sI[index]/=(float)(index+1);
    }
}

}
