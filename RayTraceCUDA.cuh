/** \file RayTraceCUDA.cuh
 * \author Tomasz Jakubczyk
 * \brief RayTrace header
 * implementation in RayTraceCUDA.cu
 */

extern "C"
{
    void RayTrace(float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float* IM, float3* P);
}
