#include "mex.h"
#include <complex.h>
#include <math.h>
#include <omp.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	double *ar, *ai, *br, *bi, *xr, *xi, *mr, *mi;
	double j, ii;
	int Nadn;
	double complex r;
	double complex x, m;
	
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
	
	
	plhs[0]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	plhs[1]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	
	ar=mxGetPr(plhs[0]);
	ai=mxGetPi(plhs[0]);
	br=mxGetPr(plhs[1]);
	bi=mxGetPi(plhs[1]);
	
	
	
	/* bessel neuman hankel */
	
	double complex Theta[Nmax];
	double complex Eta[Nmax];
	double complex Psi[Nmax];
	
	Theta[0]=csin(x);
 	Theta[1]=Theta[0]/(x)-ccos(x);

	Eta[0]=ccos(x);
	Eta[1]=Eta[0]/(x)+csin(x);

	Psi[0]=Theta[0]-I*Eta[0];
	Psi[1]=Theta[1]-I*Eta[1];
	int i;
	for (i=2;i<=Nmax;i++) {
		Theta[i] = (2. * i - 1.) / (x) * Theta[i - 1] - Theta[i - 2];
		Eta[i] = (2. * i - 1.) / (x) * Eta[i - 1] - Eta[i - 2];
		Psi[i] = Theta[i] - I*Eta[i];
	}

	/* D */

	double complex D[Nadn+1];
	D[Nadn]=0.0;
	for (i=Nadn;i>=1;i--)
   {
      D[i - 1] = (double)i / r - 1. / (D[i] + (double)i / r);
   }

	/* a and b coefiicients */
	double complex a[Nmax];
	double complex b[Nmax];
	for (i=1;i<=Nmax;i++) {
		a[i - 1] = ((D[i] / m + (double)i / x) * Theta[i] - Theta[i-1])
		   / ((D[i] / m + (double)i / x) * Psi[i] - Psi[i-1]);
		b[i - 1] = ((D[i] * m + (double)i / x) * Theta[i] - Theta[i-1])
		   / ((D[i] * m + (double)i / x) * Psi[i] - Psi[i-1]);
	}

	for (i=0;i<Nmax;i++) {
		ar[i]=creal(a[i]);
		br[i]=creal(b[i]);
		ai[i]=cimag(a[i]);
		bi[i]=cimag(b[i]);

	}
	
	
	
}