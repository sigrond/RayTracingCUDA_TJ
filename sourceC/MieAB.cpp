#include <complex>
#include <cmath>
#include <cstdio>
#include <vector>
#include"globals.h"

using namespace std;

const complex<float> I(0.0f,1.0f);

void calculateMieAB(int * nMaxTable, int nPiiTau, int sizeR, complex<float> m, float * x, float * afloat, float * aImag, float * bfloat, float * bImag) {
	complex<float> a;
	complex<float> b;
	vector<complex<float> > Theta(nPiiTau+1,complex<float>(0.0f, 0.0f));
	vector<complex<float> > Eta(nPiiTau+1, complex<float>(0.0f, 0.0f));
	vector<complex<float> > Psi(nPiiTau+1, complex<float>(0.0f, 0.0f));
#pragma omp parallel for firstprivate(a, b, Theta, Eta, Psi) default(none) shared(sizeR, nMaxTable, x, m, afloat, nPiiTau, aImag, bfloat, bImag)
	for(int k=0;k<sizeR;k++) {
		const complex<float> invM = 1.0f/m;
		const complex<float> invX = 1.0f/x[k];
		int Nadn;
		const complex<float> r = x[k]*m;
		int const Nmax = nMaxTable[k];
		float j=abs(r);
		if (((int)j)>Nmax) Nadn=(int)ceil(j)+15;
		else Nadn=Nmax+15;

		//complex<float> *D=new complex<float> [Nadn+1];
		vector<complex<float> > D(Nadn + 1, complex<float>(0.0f, 0.0f));

		/* Calculating D */
		D[Nadn]=0.0f;
		for (int i=Nadn;i>=1;--i) {
			const complex<float> aux = (complex<float>) i/r;
			D[i - 1] =  aux - 1.0f / (D[i] + aux); 
		}
		/*initial values */	
		Theta[0]= complex<float>(sin(x[k]),0);
		Theta[1]=Theta[0]/(x[k])-cos(x[k]);

		Eta[0]= complex<float>(cos(x[k]),0);
		Eta[1]=Eta[0]/(x[k])+sin(x[k]);

		Psi[0]=Theta[0]-I*Eta[0];
		Psi[1]=Theta[1]-I*Eta[1];

		a = ((D[1] * invM + (float)1 * invX) * Theta[1] - Theta[0])
			/ ((D[1] * invM + (float)1 * invX) * Psi[1] - Psi[0]);
		b = ((D[1] * m + (float)1 * invX) * Theta[1] - Theta[0])
			/ ((D[1] * m + (float)1 * invX) * Psi[1] - Psi[0]);
		afloat[nPiiTau*k] = a.real();
		aImag[nPiiTau*k] = imag(a);
		bfloat[nPiiTau*k] = b.real();
		bImag[nPiiTau*k] = imag(b);
		if (Nmax >= nPiiTau)
		{
			printf("Nmax: %d, nPiiTau: %d\n", Nmax, nPiiTau);
		}
		for (int i=2;i<=Nmax;++i) {
			const complex<float> aux = (2.0f*i - 1.0f)*invX;
			const complex<float> aux2 = (complex<float>)i*invX;
			const complex<float> aux3 = D[i] * invM + aux2;
			const complex<float> aux4 = D[i] * m + aux2;

			Theta[i] =  aux * Theta[i - 1] - Theta[i - 2];
			Eta[i]   =  aux * Eta[i - 1] - Eta[i - 2];
			Psi[i]   = Theta[i] - I*Eta[i];
			a=(aux3 * Theta[i] - Theta[i-1])
				/ (aux3 * Psi[i] - Psi[i-1]);
			b=(aux4 * Theta[i] - Theta[i-1])
				/ (aux4 * Psi[i] - Psi[i-1]);
			int index = i-1 + nPiiTau * k;
			afloat[index] = real(a);
			aImag[index] = imag(a);
			bfloat[index] = real(b);
			bImag[index] = imag(b);
		}
	}
}
