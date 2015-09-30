#pragma once
#include <vector_types.h>
#include "HandlesStructures.cuh"
/** \file RayTraceCUDA.cuh
 * \author Tomasz Jakubczyk
 * \brief RayTrace header
 * implementation in RayTraceCUDA.cu
 */
extern "C"
{
void RayTrace(float* Br, int Br_size, float* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, int IM_size, float3* P);
}
//extern "C"
//{
//    void RayTrace(float* Br, int Br_size, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float* IM, int IM_size, float3* P);
//}
