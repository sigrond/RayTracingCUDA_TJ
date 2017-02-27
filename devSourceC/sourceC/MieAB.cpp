#include <complex.h>
#include <math.h>
#include <cstdio>
#include"globals.h"


void calculateMieAB(int * nMaxTable, int nPiiTau, int sizeR, real complex m, real * x, real * aReal, real * aImag, real * bReal, real * bImag) {
	real complex a;
	real complex b;
	real complex Theta[nPiiTau];
	real complex Eta[nPiiTau];
	real complex Psi[nPiiTau];
#pragma omp parallel for private(a, b, Theta, Eta, Psi) default(none) shared(sizeR, nMaxTable, x, m, aReal, nPiiTau, aImag, bReal, bImag)
	for(int k=0;k<sizeR;k++) {
		real const complex invM = 1.0f/m;
		real const complex invX = 1.0f/x[k];
		int Nadn;
		real const complex r = x[k]*m;
		int const Nmax = nMaxTable[k];
		real j=cabs(r);
		if (((int)j)>Nmax) Nadn=(int)ceil(j)+15;
		else Nadn=Nmax+15;

		real complex D[Nadn+1];

		/* Calculating D */
		D[Nadn]=0.0f;
		for (int i=Nadn;i>=1;--i) {
			real const complex aux = (real complex) i/r;
			D[i - 1] =  aux - 1.0f / (D[i] + aux); 
		}
		/*initial values */	
		Theta[0]=csin(x[k]);
		Theta[1]=Theta[0]/(x[k])-ccos(x[k]);

		Eta[0]=ccos(x[k]);
		Eta[1]=Eta[0]/(x[k])+csin(x[k]);

		Psi[0]=Theta[0]-I*Eta[0];
		Psi[1]=Theta[1]-I*Eta[1];

		a = ((D[1] * invM + (real)1 * invX) * Theta[1] - Theta[0])
			/ ((D[1] * invM + (real)1 * invX) * Psi[1] - Psi[0]);
		b = ((D[1] * m + (real)1 * invX) * Theta[1] - Theta[0])
			/ ((D[1] * m + (real)1 * invX) * Psi[1] - Psi[0]);
		aReal[nPiiTau*k] = creal(a);
		aImag[nPiiTau*k] = cimag(a);
		bReal[nPiiTau*k] = creal(b);
		bImag[nPiiTau*k] = cimag(b);
		for (int i=2;i<=Nmax;++i) {
			real const complex aux = (2.0f*i - 1.0f)*invX;
			real const complex aux2 = (complex real)i*invX;
			real const complex aux3 = D[i] * invM + aux2;
			real const complex aux4 = D[i] * m + aux2;

			Theta[i] =  aux * Theta[i - 1] - Theta[i - 2];
			Eta[i]   =  aux * Eta[i - 1] - Eta[i - 2];
			Psi[i]   = Theta[i] - I*Eta[i];
			a=(aux3 * Theta[i] - Theta[i-1])
				/ (aux3 * Psi[i] - Psi[i-1]);
			b=(aux4 * Theta[i] - Theta[i-1])
				/ (aux4 * Psi[i] - Psi[i-1]);
			int index = i-1 + nPiiTau * k;
			aReal[index] = creal(a);
			aImag[index] = cimag(a);
			bReal[index] = creal(b);
			bImag[index] = cimag(b);
		}
	}
}
