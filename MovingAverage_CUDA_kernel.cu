/** \file MovingAverage_CUDA_kernel.cu
 * \author Tomasz Jakubczyk
 * \brief file with CUDA kernel smoothing with MovingAverage
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
void MovingAverageD(float* I, unsigned int I_size, int* I_S, float* sI, float step)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=I_size)
        return;
    float value;
    float* val0;
    int hStep=step/2;

    #pragma unroll
    for(int i=-hStep;i<hStep && index+i<I_size;i++)
    {
        if(index+i>=0)
        {
            val0=sI+index+i;
            value=I[(unsigned int)round(I_S[index]-1.0f)];
            atomicAdd(val0, value);
        }
    }
}

__global__
void DivD(unsigned int I_size, float* sI, float step)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=I_size)
        return;
    int hStep=step/2;
    if(index>hStep && index+hStep<I_size)
    {
        sI[index]/=(float)step;
    }
    else if(index<=hStep)
    {
        sI[index]/=(float)(index+hStep);
    }
    else
    {
        sI[index]/=(float)((I_size-index)+hStep);
    }
}

}
