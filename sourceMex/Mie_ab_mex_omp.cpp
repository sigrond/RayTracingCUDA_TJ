#include "mex.h"
#include <complex>
#include <cmath>
#include <exception>
#include <cstdio>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	try{
	double *ar=NULL, *ai=NULL, *br=NULL, *bi=NULL, *xr=NULL, *xi=NULL, *mr=NULL, *mi=NULL;
	double j, ii;
	int Nadn;
	complex<double> r;
	complex<double> x, m;
	
	if( !mxIsComplex(prhs[0]) || !mxIsComplex(prhs[1]) )
		mexErrMsgTxt("Inputs must be complex.\n");
	
	mr=mxGetPr(prhs[0]);
	mi=mxGetPi(prhs[0]);
	xr=mxGetPr(prhs[1]);
	xi=mxGetPi(prhs[1]);
	
	x=complex<double>(*xr, *xi);
	m=complex<double>(*mr, *mi);
	
	int Nmax=(int)ceil(*xr+4*pow(*xr, 0.33333333)+2);
	if (Nmax <= 1)
	{
		printf("Nmax: %d\n", Nmax);
		return;
	}
	ii=ceil(*xr+4*pow(*xr, 0.33333333)+2);
	r=x*m;
	j=abs(r);
	if (ceil(j)>ii)
		Nadn=(int)ceil(j)+15;
	else
		Nadn=(int)ii+15;

	if (Nadn <= 0)
	{
		printf("Nadn: %d\n", Nadn);
		return;
	}
	
	plhs[0]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	plhs[1]=mxCreateDoubleMatrix(1, Nmax, mxCOMPLEX);
	
	ar=mxGetPr(plhs[0]);
	ai=mxGetPi(plhs[0]);
	br=mxGetPr(plhs[1]);
	bi=mxGetPi(plhs[1]);
	
	
	
	/* bessel neuman hankel */
	
	complex<double> *Theta=new complex<double>[Nmax];
	complex<double> *Eta=new complex<double>[Nmax];
	complex<double> *Psi=new complex<double>[Nmax];
	
	Theta[0]=sin(x);
 	Theta[1]=Theta[0]/(x)-cos(x);

	Eta[0]=cos(x);
	Eta[1]=Eta[0]/(x)+sin(x);

	Psi[0]=Theta[0]-complex<double>(0,1)*Eta[0];
	Psi[1]=Theta[1]-complex<double>(0,1)*Eta[1];

	for (int i=2;i<=Nmax;i++) {
		Theta[i] = (2. * i - 1.) / (x) * Theta[i - 1] - Theta[i - 2];
		Eta[i] = (2. * i - 1.) / (x) * Eta[i - 1] - Eta[i - 2];
		Psi[i] = Theta[i] + complex<double>(0, -1) * Eta[i];
	}

	/* D */

	complex<double> *D=new complex<double>[Nadn+1];
	D[Nadn]=0;
	for (int i=Nadn;i>=1;i--)
   {
      D[i - 1] = (double)i / r - 1. / (D[i] + (double)i / r);
   }

	/* a and b coefiicients */
	complex<double> *a=new complex<double>[Nmax];
	complex<double> *b=new complex<double>[Nmax];
	for (int i=1;i<=Nmax;i++) {
		a[i - 1] = ((D[i] / m + (double)i / x) * Theta[i] - Theta[i-1]) / ((D[i] / m + (double)i / x) * Psi[i] - Psi[i-1]);
		b[i - 1] = ((D[i] * m + (double)i / x) * Theta[i] - Theta[i-1]) / ((D[i] * m + (double)i / x) * Psi[i] - Psi[i-1]);
	}

	for (int i=0;i<Nmax;i++) {
		ar[i]=real(a[i]);
		br[i]=real(b[i]);
		ai[i]=imag(a[i]);
		bi[i]=imag(b[i]);

	}
	
	delete[] Theta;
	delete[] Eta;
	delete[] Psi;
	delete[] D;
	delete[] a;
	delete[] b;
	}
	catch(exception &e)
	{
		printf("exception: %s\n",e.what());
	}
	catch(...)
	{
		printf("unknown exception\n");
	}
}