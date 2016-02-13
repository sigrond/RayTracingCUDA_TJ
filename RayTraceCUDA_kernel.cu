/** \file RayTraceCUDA_kernel.cu
 * \author Tomasz Jakubczyk
 * \brief RayTrace CUDA kernel function & helpers
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
#include "rcstruct.cuh"
#include "HandlesStructures.cuh"

extern "C"
{
__device__
float3 findAlpha( float3 n, float3 v, float p, float m2 )
{
    float al1=acos(dot(n,v));
    float al2;
    if(p==1)
    {
        al2=asin(sin(al1)/m2);
    }
    else
    {
        al2=asin(m2*sin(al1));
    }
    float bet=al1-al2;
    float3 S=cross(v,n);
    float3 V2;
    if(length(S)==0.0f)
    {
        V2=v;
    }
    else
    {
        float W=S.x*S.x+S.y*S.y+S.z*S.z;
        float2 B=make_float2(cos(bet),cos(al2));
        float Wx=(B.x*n.y-B.y*v.y)*S.z+(B.y*v.z-B.x*n.z)*S.y;
        float Wy=(B.y*v.x-B.x*n.x)*S.z+(B.x*n.z-B.y*v.z)*S.x;
        float Wz=(B.y*v.y-B.x*n.y)*S.x+(B.x*n.x-B.y*v.x)*S.y;
        V2=make_float3(Wx/W,Wy/W,Wz/W);
    }
    return V2;
}

__device__
rcstruct SphereCross( float3 r, float3 V, float R )
{
    float A=V.x*V.x+V.y*V.y+V.z*V.z;
    float B=2.0f*dot(r,V);
    float C=r.x*r.x+r.y*r.y+r.z*r.z-R*R;
    float D=B*B-4.0f*A*C;
    rcstruct rc;
    if(D<0.0f)
    {
        rc.a=make_float3(CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F);
        rc.b=make_float3(CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F);
    }
    else
    {
        float t1=(-B+sqrt(D))/2.0f/A;
        float t2=(-B-sqrt(D))/2.0f/A;
        rc.a=r+V*t1;
        rc.b=r+V*t2;
    }
    return rc;
}

__global__
/** \brief RayTrace CUDA kernel function.
 *
 * \param Br float*
 * \param Vb float*
 * \param VH float*
 * \param Vb_length int
 * \param VH_length int
 * \param S HandlesStructures structure contains the parameters of the lens
 * \param IC float* correction matrix
 * \param PK float4* pixel position matrix
 * \return void
 *
 */
void RayTraceD(float* Br, float* Vb, float* VH, int Vb_length, int VH_length, HandlesStructures S, float* IC, float4* PK)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    //float3 P[11];
    /*if(index==0)
    {
        P[0]=make_float3(-1,-1,-1);//error1
    }*/
    uint indexi = index/Vb_length;
    if (indexi >= VH_length)
    {
        //P[index*7]=make_float3(-100,-100,-100);//error1
        return;//empty kernel
    }
    uint indexj = index%Vb_length;
    if (indexj >= Vb_length)
    {
        //P[index*7]=make_float3(-200,-200,-200);//error1
        return;//critical error
    }

    float3 P2=make_float3(Br[indexj],Vb[indexj],VH[indexi]);/**< point on the surface of the first diaphragm */

    //uint p=0;
    float3 nan3=make_float3(CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F);

    //Calculation of the position of the sphere's center
    S.Cs1=S.l1-S.R1+S.g;
    S.Cs2=S.Cs1+S.ll+2.0f*S.R2;

    float3 P1 = S.Pk;//droplet coordinates

    float3 v = normalize(P2 - P1);//direction vector of the line
    //looking for the point of intersection of the line and lenses
    float t = (S.l1 - P2.x)/v.x;
    float3 P3 = P2 + t*v;//Point in the plane parallel to the flat surface of the lens

    if (length(make_float2(P3.y,P3.z)) > (S.efD/2))//verification whether  the point inside the aperture of the lens or not
    {
        //recalculate coordinates
        float Kp = length(make_float2(P3.y,P3.z))/(S.efD/2);
        P3.y/=Kp;
        P3.z/=Kp;
        v = normalize(P3 - P1);//direction vector of the line
    }

    //normal vector to the surface
    float3 n=make_float3(1.0f,0.0f,0.0f);

    float3 v3 = findAlpha( n, v,1,S.m2 );

    //For intensity calculation
    float P8 = acos(dot(n,v));

    rcstruct rc = SphereCross( make_float3( P3.x - S.Cs1, P3.y, P3.z ), v3,S.R1 );

    if(isnan(rc.a.x))
    {
        /*p=0;
        P[index*7+p++]=P1;
        P[index*7+p++]=P2;
        P[index*7+p++]=P3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;*/
        //P[index*7]=make_float3(100,100,100);//error1
        return;
    }

    float3 ns = normalize(rc.a);
    float3 v4 = findAlpha( ns, v3,2,S.m2 );

    //For intensity calculation
    float P9 = acos(dot(ns, v3));

    float3 P4 = make_float3( rc.a.x + S.Cs1, rc.a.y, rc.a.z );

    if(length(make_float2(rc.a.y,rc.a.z)) > S.D/2)
    {
        /*p=0;
        P[index*7+p++]=P1;
        P[index*7+p++]=P2;
        P[index*7+p++]=P3;
        P[index*7+p++]=P4;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;*/
        //P[index*7]=make_float3(200,200,200);//error2
        return;
    }

    rcstruct rc1 = SphereCross( make_float3(P4.x-S.Cs2,P4.y,P4.z), v4,S.R2 );
    if(isnan( rc1.a.x ))
    {
        /*p=0;
        P[index*7+p++]=P1;
        P[index*7+p++]=P2;
        P[index*7+p++]=P3;
        P[index*7+p++]=P4;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;*/
        //P[index*7]=make_float3(300,300,300);//error3
        return;
    }
    float3 P5 = rc1.b;
    P5.x = P5.x + S.Cs2;


    if(length(make_float2(rc1.b.y,rc1.b.z)) > S.D/2)
    {
        /*p=0;
        P[index*7+p++]=P1;
        P[index*7+p++]=P2;
        P[index*7+p++]=P3;
        P[index*7+p++]=P5;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;
        P[index*7+p++]=nan3;*/
        //P[index*7]=make_float3(400,400,400);//error4
        return;
    }

    ns = normalize(rc1.b);

    float3 v5 = findAlpha( -ns, v4,1,S.m2 );

    //For intensity calculation
    float P10 = acos(dot(-ns, v4));

    float X = S.l1 + 2*S.g + S.ll;
    t = ( X - P5.x ) / v5.x;

    float3 P6 = P5 + v5*t;

    float3 v6 = findAlpha( n, v5,2,S.m2 );

    //For intensity calculation
    float P11 = acos(dot(n, v5));

    t = (S.lCCD - P6.x ) / v6.x;

    float3 P7 = P6 + v6*t;

    /*p=0;
    P[index*7+p++]=P1;
    P[index*7+p++]=P2;
    P[index*7+p++]=P3;
    P[index*7+p++]=P4;
    P[index*7+p++]=P5;
    P[index*7+p++]=P6;
    P[index*7+p++]=P7;*/

    /*if(IM==NULL || IM==0)
    {
        //P[index*7]=make_float3(500,500,500);//error5
        return;//no need to calculate image
    }*/

    /*float dist=length(P1-P2)+length(P2-P3)+
                length(P3-P4)+length(P4-P5)+
                length(P5-P6)+length(P6-P7);
    float3 vR = normalize(P7-P6);
    float alp = acos(dot(make_float3(1,0,0),vR));*/
    float W  = S.shX + ( S.CCDW/2.0f +P7.y)/S.PixSize;
    float Hi = S.shY + ( S.CCDH/2.0f +P7.z)/S.PixSize;
    //float value=cos(alp)/(dist*dist);

    //Recording position of rays and a number of rays that walk into the cell
    float value=1.0f;
    float* val0;
    val0=PX+(unsigned int)round(Hi)*4+(unsigned int)round(W)*480;
    atomicAdd(val0, P2.x);
    val0=PX+1+(unsigned int)round(Hi)*4+(unsigned int)round(W)*480;
    atomicAdd(val0, P2.y);
    val0=PX+2+(unsigned int)round(Hi)*4+(unsigned int)round(W)*480;
    atomicAdd(val0, P2.z);
    val0=PX+3+(unsigned int)round(Hi)*4+(unsigned int)round(W)*480;
    atomicAdd(val0, value);//+1

    //The calculation of energy loss,  caused by reflection on lens surfaces and rising distance
    value=0.01f*(length(P1-P2) + length(P2-P3));
    value*=value;//fast square
    float Ka1 = cos(P8)/value;
    value=0.01f*length(P3-P4);
    value*=value;
    float Ka2 = Ka1*cos(P9)/value;
    value=0.01f*length(P4-P5);
    value*=value;
    float Ka3 = Ka2*cos(P10)/value;
    value=cos(P11);//in calculation intensive code calculating same cosine twice isn't wise
    value*=value;
    float Ka4 = Ka3*value;
    value=0.01f*(length(P5-P6) + length(P6-P7));
    Ka4/=value;
    value=1.0f/Ka4;
    val0=IC+(unsigned int)round(Hi)+(unsigned int)round(W)*480;
    atomicAdd(val0, value);

    //float* val0=IM+(unsigned int)round(Hi)+(unsigned int)round(W)*480;
    //atomicAdd(val0, value);
}
}//extern "C"
