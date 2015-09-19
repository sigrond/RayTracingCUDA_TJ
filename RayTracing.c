#include "mex.h"
#include<stdio.h>
#include<stdlib.h>
#include "RayTracingCUDA.cuh"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /**< \todo get data */
    /** call cuda kernels */
    RayTrace(P2,VH_length,Vb_length,S,IM);
    /**< \todo return data */
}
