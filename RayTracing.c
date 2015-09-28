#include "mex.h"
#include<stdio.h>
#include<stdlib.h>
#include "RayTraceCUDA.cuh"
#include "HandlesStructures.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /**< \todo get data */
    /** call cuda kernels */
//float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, float3* P
    RayTrace(Br, Vb, VH, Vb_length, VH_length, S, IM, P);
    /**< \todo return data */
}
