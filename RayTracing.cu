/** \file RayTracing.cu
 * \brief this file contains calls to matlab and main function
 * compilation line:
 * 32bit:
 * nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda -output RayTracingCUDA
 * 64bit:
 * nvmex -f nvmexopts64.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda -output RayTracingCUDA
 */
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
 * function [IC, PX] = RayTracing(Br,Vb,VH,handles)
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
    float* Vb;
    float* VH;
    int Br_size;
    int Vb_length=0;
    int VH_length=0;
    HandlesStructures S;
    float* IC;
    int IC_size;
    //float3* P;
    float* PX;

    /**< get data */
    printf("nlhs:%d\n",nlhs);
    if(nlhs<=1)
    {
    	printf("not enough outputs, function returns [IC, PX]\n");
    	return;
    }
    else
    {
        /*if(!mxIsSingle(plhs[1]))
        {
            printf("IM is not a float array");
            return;
        }*/
        int dimsIC[2]={480,640};/**< \todo wczytywać wartości z handles */
        plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
        IC=(float*)mxGetPr(plhs[0]);
        IC_size=mxGetN(plhs[0])*mxGetM(plhs[0]);
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
        printf("1st argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[1]))
    {
        printf("2nd argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[2]))
    {
        printf("3rd argument needs to be single precision vector\n");
        return;
    }
    Br=(float*)mxGetPr(prhs[0]);
    Br_size=mxGetN(prhs[0])*mxGetM(prhs[0]);
    Vb=(float*)mxGetPr(prhs[1]);
    Vb_length=mxGetN(prhs[1])*mxGetM(prhs[1]);
    VH=(float*)mxGetPr(prhs[2]);
    VH_length=mxGetN(prhs[2])*mxGetM(prhs[2]);
    printf("Br_size:%d,Vb_length:%d,VH_length:%d\n",Br_size,Vb_length,VH_length);

    mxArray* tmp;
    mxArray* tmp2;
    int tmpb=mxIsStruct(prhs[3]);
    printf("mxIsStruct:%d\n",tmpb);
    if(!tmpb)
    {
        printf("4th argument must be a struct\n");
        return;
    }
    tmp=mxGetField(prhs[3],0,"shX");//handles.shX
    S.shX=(float)mxGetScalar(tmp);
    printf("S.shX:%f\n",S.shX);
    tmp=mxGetField(prhs[3],0,"shY");
    S.shY=(float)mxGetScalar(tmp);
    printf("S.shY:%f\n",S.shY);
    tmp=mxGetField(prhs[3],0,"S");//handles.S
    tmp2=mxGetField(tmp,0,"D");//handles.S.D
    S.D=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"efD");
    S.efD=(float)mxGetScalar(tmp2);
    printf("S.D.efD:%f\n",S.efD);
    tmp2=mxGetField(tmp,0,"R");
    S.R1=(float)((double*)mxGetPr(tmp2))[0];
    S.R2=(float)((double*)mxGetPr(tmp2))[1];
    printf("S.R1:%f,S.R2:%f\n",S.R1,S.R2);
    tmp2=mxGetField(tmp,0,"g");
    S.g=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"l1");
    S.l1=(float)mxGetScalar(tmp2);
    printf("S.l1: %f\n",S.l1);//eL jeden !!!
    tmp2=mxGetField(tmp,0,"ll");
    S.ll=(float)mxGetScalar(tmp2);
    printf("S.ll: %f\n",S.ll);//eL eL !!!
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
    printf("S.Pk:(%f, %f, %f)\n",S.Pk.x,S.Pk.y,S.Pk.z);
    tmp2=mxGetField(tmp,0,"m2");
    S.m2=(float)mxGetScalar(tmp2);
    printf("S.m2:%f\n",S.m2);

    int dims[3]={4,480,640};
    plhs[1]=mxCreateNumericArray(3,dims,mxSINGLE_CLASS,mxREAL);
    PX=(float*)mxGetPr(plhs[1]);
    //system("pause");
    printf("IC:%d\n",IC);
    printf("IC_size:%d\n",IC_size);

    printf("Br: ");
    for(int i=0;i<Br_size && i<100;i++)
    printf("%f ",Br[i]);
    printf("\nVb: ");
    for(int i=0;i<Vb_length && i<100;i++)
    printf("%f ",Vb[i]);
    printf("\nVH: ");
    for(int i=0;i<VH_length && i<100;i++)
    printf("%f ",VH[i]);
    printf("\n new version 3 \n");

    /** call cuda kernels */
//float3* Br, int* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float3* IM, float3* P
    RayTrace(Br, Br_size, Vb, VH, Vb_length, VH_length, S, IC, IC_size, PX);

    printf("\nIC: ");
    for(int i=0,j=0;i<IC_size && j<100;i++)
    {
        if(i>0 && IC[i]==0.0f)
        {
            if(IC[i-1]==0.0f)
            continue;
        }
        printf("%.14f ",IC[i]);
        j++;
    }
    printf("\n");

    printf("\nPX: ");
    for(int i=0,j=0;i<640*480*4 && j<100;i++)
    {
        if(i>0 && PX[i]==0.0f)
        {
            if(PX[i-1]==0.0f)
            continue;
        }
        printf("%f ",PX[i]);
        j++;
    }
    printf("\n");
}
