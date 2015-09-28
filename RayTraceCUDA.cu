/** \file RayTraceCUDA.cu
 * \author Tomasz Jakubczyk
 * \brief Implementation of RayTrace function
 * which calls RayTraceD CUDA kernels
 */

#include "HandlesStructures.cuh"
#include "RayTraceCUDA_kernel.cuh"

extern "C"
{
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
     * \param Br float3*
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
    void RayTrace(float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, float3* P)
    {
        uint numThreads, numBlocks;
        computeGridSize(VH_length*Vb_length, 256, numBlocks, numThreads);
        RayTraceD<<< numBlocks, numThreads >>>(Br,Vb,VH,Vb_length,VH_length,S,IM,P);
    }
}
