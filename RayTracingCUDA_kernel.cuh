#include <math.h>
#include <float.h>
#include "helper_math.h"
#include "rcstruct.cuh"

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
        float Wz=(B.y*v.y-B.x*n.y)*S.x+(B.x*n.x-B.y*v.x)*S.y
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
    if(D<0.0f)
    {
        rc.a=make_float3(NAN,NAN,NAN);
        rc.b=make_float3(NAN,NAN,NAN);
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
void RayTraceD(float3* P2, int VH_length, int Vb_length, HandlesStructures S, float* IM)
{
    uint indexi = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x)/Vb_length;
    if (indexi >= VH_length) return;//empty kernel
    uint indexj = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x)%Vb_length;
    if (indexj >= Vb_length) return;//critical error

    //Calculation of the position of the sphere's center
    S.Cs1=S.l1-S.R1+S.g;
    S.Cs2=S.Cs1+S.l1+2.0f*S.R2;

    float3 P1 = S.Pk;//droplet coordinates

    float3 v = normalize(P2 - P1);//direction vector of the line
    //looking for the point of intersection of the line and lenses
    float t = (S.l1 - P2[j].x))/v.x;
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

    rcstruct rc = SphereCross( make_float3( P3.x - S.Cs1, P3.y, P3.z ), v3,S.R1 );

    if(isnan(rc.a.x))
    {
        return;
    }

    float3 ns = normalize(rc.a);
    float3 v4 = findAlpha( ns, v3,2,S.m2 );

    float3 P4 = make_float3( rc.a.x + S.Cs1, rc.a.y, rc.a.z );

    if(length(make_float2(rc.a.y,rc.a.z)) > S.D/2)
    {
        return;
    }

    rcstruct rc1 = SphereCross( make_float3(P4.x-S.Cs2,P4.y,P4.z), v4,S.R2 );
    if(isnan( rc1.a.x ))
    {
        return;
    }

    float3 P5 = rc1.b;
    P5.x = P5.x + S.Cs2;

    if(length(make_float2(rc1.b.y,rc1.b.z)) > S.D/2)
    {
        return;
    }

    ns = normalize(rc1.b);

    float3 v5 = findAlpha( -ns, v4,1,S.m2 );

    float X = S.l1 + 2*S.g + S.ll;
    float t = ( X - P5.x ) / v5.x;

    float3 P6 = P5 + v5*t;

    float3 v6 = findAlpha( n, v5,2,S.m2 );

    t = (S.lCCD - P6.x ) / v6.x;

    float3 P7 = P6 + v6*t;

    float dist=length(P1-P2)+length(P2-P3)+
                length(P3-P4)+length(P4-P5)+
                length(P5-P6)+length(P6-P7);
    float3 vR = normalize(P7-P6);
    float alp = acos(dot(make_float3(1,0,0),vR));
    float W =  handles.shX + ( handles.S.CCDW/2 +P7.y)/handles.S.PixSize;
    float Hi =  handles.shY + (handles.S.CCDH/2 +P7.z)/handles.S.PixSize;
    atomicAdd(&(IM[round(Hi)][round(W)]), cos(alp)/(dist*dist));
}
