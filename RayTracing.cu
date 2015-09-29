#define WIN32
#include "mex.h"
#include<stdio.h>
#include<stdlib.h>
#include "RayTraceCUDA.cuh"
#include "HandlesStructures.cuh"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"

/** \brief RayTracing mex function
 * function [P,IM] = RayTracing(Br,Vb,VH,handles)
 * function P = RayTracing(Br,Vb,VH,handles)
 * \param nlhs int
 * \param plhs[] mxArray*
 * \param nrhs int
 * \param prhs[] const mxArray*
 * \return void
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float* Br;
    int* Vb;
    float* VH;
    int Br_size;
    int Vb_length;
    int VH_length;
    HandlesStructures S;
    float3* IM;
    int IM_size;
    float3* P;

    /**< get data */
    if(nlhs<=1)//no place for IM output
    {
        IM=NULL;//don't calculate IM
        IM_size=0;
    }
    else
    {
        if(!mxIsSingle(plhs[1]))
        {
            printf("IM is not a float array");
            return;
        }
        IM=(float3*)mxGetPr(plhs[1]);
        IM_size=mxGetN(plhs[1])*mxGetM(plhs[1]);
    }
    if(nrhs<4)
    {
        printf("not enough arguments\n");
        return;
    }
    if(nrhs>4)
    {
        printf("to much arguments\n");
        return;
    }
    if(!mxIsSingle(prhs[0]))
    {
        printf("1st argument needs to be single precision floating point vector of 3d points");
        return;
    }
    Br=(float*)mxGetPr(prhs[0]);
    Br_size=mxGetN(prhs[0])*mxGetM(prhs[0]);
    Vb=(int*)mxGetPr(prhs[1]);
    Vb_length=mxGetN(prhs[1])*mxGetM(prhs[1]);
    VH=(float*)mxGetPr(prhs[2]);
    VH_length=mxGetN(prhs[2])*mxGetM(prhs[2]);
    mxArray* tmp;
    mxArray* tmp2;
    tmp=mxGetField(prhs[3],0,"shX");//handles.shX
    S.shX=(float)mxGetScalar(tmp);
    tmp=mxGetField(prhs[3],0,"shY");
    S.shY=(float)mxGetScalar(tmp);
    tmp=mxGetField(prhs[3],0,"S");//handles.S
    tmp2=mxGetField(tmp,0,"D");//handles.S.D
    S.D=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"efD");
    S.efD=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"R");
    S.R1=(float)((double*)mxGetPr(tmp2))[0];
    S.R2=(float)((double*)mxGetPr(tmp2))[1];
    tmp2=mxGetField(tmp,0,"g");
    S.g=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"l1");
    S.l1=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"lCCD");
    S.lCCD=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"CCDPH");
    S.CCDPH=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"CCDPW");
    S.CCDPW=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"PixSize");
    S.PixSize=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"CCDH");
    S.CCDH=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"CCDW");
    S.CCDW=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"Pk");
    S.Pk.x=(float)((double*)mxGetPr(tmp2))[0];
    S.Pk.y=(float)((double*)mxGetPr(tmp2))[1];
    S.Pk.z=(float)((double*)mxGetPr(tmp2))[2];
    tmp2=mxGetField(tmp,0,"m2");
    S.m2=(float)mxGetScalar(tmp2);

    int dims[4]={VH_length,Vb_length,7,3};
    plhs[0]=mxCreateNumericArray(4,dims,mxSINGLE_CLASS,mxREAL);
    P=(float3*)mxGetPr(plhs[0]);

    /** call cuda kernels */
//float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, float3* P
    RayTrace(Br, Br_size, Vb, VH, Vb_length, VH_length, S, IM, IM_size, P);

}