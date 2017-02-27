#include"globals.h"
#include<cmath> 
#include"auxillary.h"
#ifdef _OPENMP
#include<omp.h>
#endif

void calculateMiePT(int col, int nmax, real * u, real * p, real * t) {
	real aux[nmax+1];
	real aux2[nmax+1];
	for(int i=3; i<=nmax; i++){
		aux2[i] = (real)(i)/(i-1);
		aux[i]=(real)(2*i-1)/(i-1);
	}
	for (int j = 0;j < col;j++ ) {
		int index1=j*nmax;
		p[index1] = 1;
		t[index1] = u[j];
		p[index1+1] = 3*u[j];
		t[index1+1] = 3*cos(2*acos(u[j]));
		for (int i = 3;i <= nmax;i++) {
			int index = index1+i-1;
			p[index] = aux[i] * p[index-1] * u[j] -  aux2[i] * p[index-2];
			t[index] = i*u[j]*p[index] - (i+1)*p[index-1];
		}
	}
}
