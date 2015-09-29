/** \file RayTraceCUDA.cu
 * \author Tomasz Jakubczyk
 * \brief Implementation of RayTrace function
 * which calls RayTraceD CUDA kernels
 */
#define WIN32
#include "HandlesStructures.cuh"
//#include "RayTraceCUDA_kernel.cuh"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include <stdlib.h>

extern "C"
{
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

    }
}
