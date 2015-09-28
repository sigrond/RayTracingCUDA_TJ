/** \file RayTraceCUDA_kernel.cuh
 * \author Tomasz Jakubczyk
 * \brief RayTrace CUDA kernel header
 */

extern "C"
{
    __global__
    void RayTraceD(float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float* IM, float3* P);
}
