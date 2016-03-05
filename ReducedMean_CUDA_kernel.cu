/** \file ReducedMean_CUDA_kernel.cu
 * \author Tomasz Jakubczyk
 * \brief kernel function
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

extern "C"
{

__global__
void ReducedMeanD(float* Theta_S,unsigned int Theta_S_size, float deltaT, unsigned int max_nom, float* I, float* I_S, float* nTheta, float* nI, float* counter)
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

    unsigned int nom=floor((Theta_S[index]-Theta_S[0])/deltaT);
    if(nom>max_nom || nom<=0)
        return;
    //if(((Theta_S[0]+deltaT*nom)<=Theta_S[index]) && ((Theta_S[0]+deltaT*(nom+1))>=Theta_S[index]))
    //{
        val0=nTheta+nom-1;
        value=Theta_S[index];
        atomicAdd(val0, value);
        val0=nI+nom-1;
        value=I[(unsigned int)round(I_S[index])];
        atomicAdd(val0, value);
        val0=counter+nom-1;
        value=1.0f;
        atomicAdd(val0, value);
    //}
}

}

