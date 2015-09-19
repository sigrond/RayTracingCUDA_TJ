#include <math.h>
#include <float.h>
#include "helper_math.h"


__global__
void RayTraceD(float3* P2, int VH_length, int Vb_length, HandlesStructures S, float3* IM)
{
    uint indexi = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x)/Vb_length;
    if (indexi >= VH_length) return;//empty kernel
    uint indexj = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x)%Vb_length;
    if (indexj >= Vb_length) return;//critical error

    //Calculation of the position of the sphere's center
    S.Cs1=S.l1-S.R1+S.g;
    S.Cs2=S.Cs1+S.l1+2.0f*S.R2;

    float3 P1 = S.Pk;//droplet coordinates

    float3 v = (P2 - P1)/normalize(P2 - P1);//direction vector of the line
    //looking for the point of intersection of the line and lenses
    float t = (S.l1 - P2[j].x))/v.x;
    float3 P3 = P2 + t*v;//Point in the plane parallel to the flat surface of the lens

    if (normalize(make_float2(P3.y,P3.z)) > (S.efD/2))//verification whether  the point inside the aperture of the lens or not
    {
        //compare coordinates
        float Kp = normalize(make_float2(P3.y,P3.z))/(S.efD/2);
        P3.y/=Kp;
        P3.z/=Kp;
        v = (P3 - P1)/normalize(P3 - P1);//direction vector of the line
    }

    //normal vector to the surface
    float3 n=make_float3(1.0f,0.0f,0.0f);

    float3 v3 = findAlpha( n, v,1,S.m2 );

    /**< \todo ' - conjugate transpose */
    //float 3rc = SphereCross( [ P3(1) - S.Cs1, P3(2), P3(3) ], v3',S.R(1) );

}
