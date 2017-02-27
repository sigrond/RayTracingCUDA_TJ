#include "mex.h"
#include "float3.hpp"
#include "HandlesStructures.hpp"
#include <cstdio>
#include <cmath>

using namespace std;

float dot(const float3 &a, float3 &b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
}

float3 findAlpha( float3 n, float3 v, float p, float m2 )
{
    float al1=acos(dot(n,v));
	//printf("%f\n",al1);
    float al2;
    if(p==1)
    {
        al2=asin(sin(al1)/m2);
    }
    else
    {
        al2=asin(m2*sin(al1));
    }
	//printf("%f\n",al2);
    float bet=al1-al2;
    float3 S=cross(v,n);
	//printf("%f %f %f\n",S.x,S.y,S.z);
    float3 V2;
    if(norm(S)==0.0f)
    {
        V2=v;
    }
    else
    {
        float W=S.x*S.x+S.y*S.y+S.z*S.z;
        float3 B={cos(bet),cos(al2),0};
        float Wx=(B.x*n.y-B.y*v.y)*S.z+(B.y*v.z-B.x*n.z)*S.y;
        float Wy=(B.y*v.x-B.x*n.x)*S.z+(B.x*n.z-B.y*v.z)*S.x;
        float Wz=(B.y*v.y-B.x*n.y)*S.x+(B.x*n.x-B.y*v.x)*S.y;
        V2={Wx/W,Wy/W,Wz/W};
    }
	//printf("%f %f %f\n",V2.x,V2.y,V2.z);
    return V2;
}

float3* SphereCross( float3 r, float3 V, float R )
{
	//printf("r: %f %f %f\n",r.x,r.y,r.z);
	//printf("V: %f %f %f\n",V.x,V.y,V.z);
	//printf("R: %f\n",R);
    float A=V.x*V.x+V.y*V.y+V.z*V.z;
    float B=2.0f*dot(r,V);
    float C=r.x*r.x+r.y*r.y+r.z*r.z-R*R;
    float D=B*B-4.0f*A*C;
	//printf("A: %f\n",A);
	//printf("B: %f\n",B);
	//printf("C: %f\n",C);
	//printf("D: %f\n",D);
    float3* rc=new float3[2];
    if(D<0.0f)
    {
        rc[0]={NAN,NAN,NAN};
        rc[1]={NAN,NAN,NAN};
    }
    else
    {
        float t1=(-B+sqrt(D))/2.0f/A;
        float t2=(-B-sqrt(D))/2.0f/A;
        rc[0]=r+V*t1;
        rc[1]=r+V*t2;
    }
    return rc;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double* P2p;
	float3* P;
	
	if(nrhs<2)
    {
        printf("not enough arguments\n");
        return;
    }
	
	if(nlhs<1)
    {
    	printf("not enough outputs, function returns P\n");
    	return;
    }
	
	
	
	P2p=(double*)mxGetPr(prhs[0]);
	int P2_size=mxGetN(prhs[0])*mxGetM(prhs[0]);
	#ifdef DEBUG
	printf("P2_size:%d\n",P2_size);
	#endif
	
	HandlesStructures S;
	
	mxArray* tmp;
    mxArray* tmp2;
    int tmpb=mxIsStruct(prhs[1]);
	#ifdef DEBUG
    printf("mxIsStruct:%d\n",tmpb);
	#endif
    if(!tmpb)
    {
        printf("2nd argument must be a struct\n");
        return;
    }
    tmp=mxGetField(prhs[1],0,"shX");//handles.shX
    S.shX=(float)mxGetScalar(tmp);
	#ifdef DEBUG
    printf("S.shX:%f\n",S.shX);
	#endif
    tmp=mxGetField(prhs[1],0,"shY");
    S.shY=(float)mxGetScalar(tmp);
	#ifdef DEBUG
    printf("S.shY:%f\n",S.shY);
	#endif
    tmp=mxGetField(prhs[1],0,"S");//handles.S
    tmp2=mxGetField(tmp,0,"D");//handles.S.D
    S.D=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"efD");
    S.efD=(float)mxGetScalar(tmp2);
	#ifdef DEBUG
    printf("S.D.efD:%f\n",S.efD);
	#endif
    tmp2=mxGetField(tmp,0,"R");
    S.R1=(float)((double*)mxGetPr(tmp2))[0];
    S.R2=(float)((double*)mxGetPr(tmp2))[1];
	#ifdef DEBUG
    printf("S.R1:%f,S.R2:%f\n",S.R1,S.R2);
	#endif
    tmp2=mxGetField(tmp,0,"g");
    S.g=(float)mxGetScalar(tmp2);
    tmp2=mxGetField(tmp,0,"l1");
    S.l1=(float)mxGetScalar(tmp2);
	#ifdef DEBUG
    printf("S.l1: %f\n",S.l1);//eL jeden !!!
	#endif
    tmp2=mxGetField(tmp,0,"ll");
    S.ll=(float)mxGetScalar(tmp2);
	#ifdef DEBUG
    printf("S.ll: %f\n",S.ll);//eL eL !!!
	#endif
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
	#ifdef DEBUG
    printf("S.Pk:(%f, %f, %f)\n",S.Pk.x,S.Pk.y,S.Pk.z);
	#endif
    tmp2=mxGetField(tmp,0,"m2");
    S.m2=(float)mxGetScalar(tmp2);
	#ifdef DEBUG
    printf("S.m2:%f\n",S.m2);
	#endif
	
	
	float3 P2;
	P2.x=P2p[0];
	P2.y=P2p[1];
	P2.z=P2p[2];
	
	
	S.Cs1=S.l1-S.R1+S.g;
    S.Cs2=S.Cs1+S.ll+2.0f*S.R2;
	//printf("S.Cs2: %f\n",S.Cs2);
	
	float3 P1 = S.Pk;
	
	float3 v = (P2 - P1)/norm(P2 - P1);
	
	float t = (S.l1 - P2.x)/v.x;
	float3 P3 = P2 + v*t;
	float3 tmpP3;
	tmpP3.x=0;
	tmpP3.y=P3.y;
	tmpP3.z=P3.z;
	float Kp;
	
	if( norm(tmpP3) > (S.efD/2) )
	{
		Kp = norm(tmpP3)/(S.efD/2);
		P3.y = P3.y/Kp;
		P3.z = P3.z/Kp;
		v = (P3 - P1)/norm(P3 - P1);
	}
	
	float3 n ={ 1, 0, 0 };
	
	float3 v3 = findAlpha( n, v,1,S.m2 );
	
	float P8 = acos(dot(n,v));
	
	float3* rc = SphereCross( { P3.x - S.Cs1, P3.y, P3.z }, v3,S.R1 );
	
	if(isnan(rc[0].x))
	{
		int dimsIC[2]={3,4};
		plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
		P=(float3*)mxGetPr(plhs[0]);
		P[0]=P1;
		P[1]=P2;
		P[2]=P3;
		P[3]={NAN,NAN,NAN};
		return;
	}
	
	float3 ns = rc[0] / norm( rc[0] );
	
	//printf("ns: %f %f %f\n",ns.x,ns.y,ns.z);
	//printf("v3: %f %f %f\n",v3.x,v3.y,v3.z);
	float3 v4 = findAlpha( ns, v3,2,S.m2 );
	//printf("v4: %f %f %f\n",v4.x,v4.y,v4.z);
	
	float P9 = acos(dot(ns, v3));
	
	float3 P4 = { rc[0].x + S.Cs1, rc[0].y, rc[0].z };
	
	//printf("%f %f %f %f\n",rc[0].y,rc[0].z,norm({0,rc[0].y,rc[0].z}),S.D/2);
	
	if(norm({0,rc[0].y,rc[0].z}) > S.D/2)
	{
		int dimsIC[2]={3,5};
		plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
		P=(float3*)mxGetPr(plhs[0]);
		P[0]=P1;
		P[1]=P2;
		P[2]=P3;
		P[3]=P4;
		P[4]={NAN,NAN,NAN};
		return;
	}
	
	float3* rc1 = SphereCross( {P4.x-S.Cs2,P4.y,P4.z}, v4,S.R2 );
	//printf("%f %f %f %f %f %f\n",rc1[0].x,rc1[0].y,rc1[0].z,rc1[1].x,rc1[1].y,rc1[1].z);
    if(isnan( rc1[0].x ))
    {
		int dimsIC[2]={3,5};
		plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
		P=(float3*)mxGetPr(plhs[0]);
		P[0]=P1;
		P[1]=P2;
		P[2]=P3;
		P[3]=P4;
		P[4]={NAN,NAN,NAN};
        return;
    }
	
	float3 P5 = rc1[1];
    P5.x = P5.x + S.Cs2;
	
	if(norm({0,rc1[1].y,rc1[1].z}) > S.D/2)
    {
		int dimsIC[2]={3,5};
		plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
		P=(float3*)mxGetPr(plhs[0]);
		P[0]=P1;
		P[1]=P2;
		P[2]=P3;
		P[3]=P5;
		P[4]={NAN,NAN,NAN};
        return;
    }
	
	ns = rc1[1] / norm( rc1[1] );
	
	float3 v5 = findAlpha( -ns, v4,1,S.m2 );
	
	float P10 = acos(dot(-ns, v4));
	
	float X = S.l1 + 2*S.g + S.ll;
    t = ( X - P5.x ) / v5.x;
	
	float3 P6 = P5 + v5*t;
	
	float3 v6 = findAlpha( n, v5,2,S.m2 );
	
	float P11 = acos(dot(n, v5));
	
	t = (S.lCCD - P6.x ) / v6.x;
	
	float3 P7 = P6 + v6*t;
	
	int dimsIC[2]={3,11};
	plhs[0]=mxCreateNumericArray(2,dimsIC,mxSINGLE_CLASS,mxREAL);
	P=(float3*)mxGetPr(plhs[0]);
	P[0]=P1;
	P[1]=P2;
	P[2]=P3;
	P[3]=P4;
	P[4]=P5;
	P[5]=P6;
	P[6]=P7;
	P[7]={P8*57.29577951308f,P8*57.29577951308f,P8*57.29577951308f};
	P[8]={P9*57.29577951308f,P9*57.29577951308f,P9*57.29577951308f};
	P[9]={P10*57.29577951308f,P10*57.29577951308f,P10*57.29577951308f};
	P[10]={P11*57.29577951308f,P11*57.29577951308f,P11*57.29577951308f};
}