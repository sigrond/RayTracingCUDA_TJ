#include "mex.h"
#include <complex.h>
#include <math.h>
/*
This mex /C/ program calculates Mie a and b coefficients.
author: Szymon Migacz, mailto: szmigacz@gmail.com
Last updated: 13 Oct 2011

Program returns two complex matrices: a and b, using two complex parameters m and x
Calling example:
[A B]=Mie_ab_mex_omp(m,x)

The program was written in pure C using standard libraries: complex.h and math.h

The code was succesfully compiled under Linux Mint 11 and Matlab2010b 64bit
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/*Initial declarations  */
	double *ar, *ai, *br, *bi, *xr, *xi, *mr, *mi;
	double j, ii;
	int Nadn;
	double complex r;
	double complex x, m;
	
	/*Preparing input data*/
	if( !mxIsComplex(prhs[0]) || !mxIsComplex(prhs[1]) )
		mexErrMsgTxt("Inputs must be complex.\n");
	mr=mxGetPr(prhs[0]);
	mi=mxGetPi(prhs[0]);
	xr=mxGetPr(prhs[1]);
	xi=mxGetPi(prhs[1]);
	x=*xr+I*(*xi);
	m=*mr+I*(*mi);
	
	
	int Nmax=ceil(*xr+4*pow(*xr, 0.33333333)+2);
	ii=ceil(*xr+4*pow(*xr, 0.33333333)+2);
	r=x*m;
	j=cabs(r);
	if (ceil(j)>ii) Nadn=(int)ceil(j)+15;
	else Nadn=(int)ii+15;
	
	/* Preparing output data */
	plhs[0]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	plhs[1]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	
	ar=mxGetPr(plhs[0]);
	ai=mxGetPi(plhs[0]);
	br=mxGetPr(plhs[1]);
	bi=mxGetPi(plhs[1]);
	
	/* declarations */
	double complex D[Nadn+1];
	double complex Theta[Nmax];
	double complex Eta[Nmax];
	double complex Psi[Nmax];
	int i;
	double complex temp1;
	double complex temp2;
	
		/* Calculating D */
	D[Nadn]=0.0;
	for (i=Nadn;i>=1;i--)
   {
      D[i - 1] = (double)i / r - 1. / (D[i] + (double)i / r);
   }
	/*initial values */	
	Theta[0]=csin(x);
 	Theta[1]=Theta[0]/(x)-ccos(x);

	Eta[0]=ccos(x);
	Eta[1]=Eta[0]/(x)+csin(x);

	Psi[0]=Theta[0]-I*Eta[0];
	Psi[1]=Theta[1]-I*Eta[1];
	
	
	temp1 = ((D[1] / m + (double)1 / x) * Theta[1] - Theta[0])
		   / ((D[1] / m + (double)1 / x) * Psi[1] - Psi[0]);
	temp2 = ((D[1] * m + (double)1 / x) * Theta[1] - Theta[0])
		   / ((D[1] * m + (double)1 / x) * Psi[1] - Psi[0]);
	ar[0]=creal(temp1);
	ai[0]=cimag(temp1);
	br[0]=creal(temp2);
	bi[0]=cimag(temp2);
	
/* Evaluating loop 
remark: this loop can be parallelized using OMP library adding appropriate pragma */
	
	for (i=2;i<=Nmax;i++) {
		Theta[i] = (2. * i - 1.) / (x) * Theta[i - 1] - Theta[i - 2];
		Eta[i] = (2. * i - 1.) / (x) * Eta[i - 1] - Eta[i - 2];
		Psi[i] = Theta[i] - I*Eta[i];
		temp1=((D[i] / m + (double)i / x) * Theta[i] - Theta[i-1])
		   / ((D[i] / m + (double)i / x) * Psi[i] - Psi[i-1]);
		temp2=((D[i] * m + (double)i / x) * Theta[i] - Theta[i-1])
		   / ((D[i] * m + (double)i / x) * Psi[i] - Psi[i-1]);
		ar[i-1]=creal(temp1);
		ai[i-1]=cimag(temp1);
		br[i-1]=creal(temp2);
		bi[i-1]=cimag(temp2);
	}
	
}
