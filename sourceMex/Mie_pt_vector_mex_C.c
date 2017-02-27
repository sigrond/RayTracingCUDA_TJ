#include "mex.h"
#include "math.h"
/*
This mex /C/ program calculates Mie p and t coefficients.
author: Szymon Migacz, mailto: szmigacz@gmail.com
Last updated: 13 Oct 2011

Program returns two complex matrices: P and T, using two parameters: 
u - double in range [-1,1] (may be a vector)
nmax - integer

Calling example:
[P T]=Mie_pt_vector_mex_C(u, nmax)

The program was written in pure C using standard library math.h

The code was succesfully compiled under Linux Mint 11 and Matlab2010b 64bit

*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/*Initial declarations */
	int *nmax;
	double *u;
	int kolumn;
	double *p;
	double *t;
	int i,j;
	
	
	/* Praparing input data */
	u=mxGetPr(prhs[0]);
	nmax=(int*)mxGetPr(prhs[1]);
	kolumn=mxGetN(prhs[0]);
	if(mxIsDouble(prhs[0])!=1) {
		mexErrMsgTxt("Error: U is not a double");
	}
	if(mxIsInt32(prhs[1])!=1) {
		mexErrMsgTxt("Error: nmax is not an integer");
	}
	
	/*Preparing output data */
	
	plhs[0]=mxCreateDoubleMatrix(*nmax, kolumn, mxREAL);
	plhs[1]=mxCreateDoubleMatrix(*nmax, kolumn, mxREAL);
	p=mxGetPr(plhs[0]);
	t=mxGetPr(plhs[1]);
	
	/*Evaluating loop */
	
	for (j=0;j<kolumn;j++) {
		p[j*(*nmax)]=1;
		t[j*(*nmax)]=u[j];
		p[j*(*nmax)+1]=3*u[j];
		t[j*(*nmax)+1]=3*cos(2*acos(u[j]));
		for (i=3;i<=*nmax;i++) {
			p[j*(*nmax)+i-1]=(double)(2*i-1)/(i-1)*p[j*(*nmax)+i-2]*u[j]-((double)(i)/(i-1)*p[j*(*nmax)+i-3]);
			t[j*(*nmax)+i-1]=i*u[j]*p[j*(*nmax)+i-1]-((i+1)*p[j*(*nmax)+i-2]);
		}
	}
}
