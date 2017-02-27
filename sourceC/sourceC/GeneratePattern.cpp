#include<complex.h>
#include<cstdio>
#include<cmath>
#include<omp.h>
#include"MieAB.h"
#include"MiePT.h"
#include"auxillary.h"
#include"cudaGenerate.h"
#include"globals.h"

inline void GeneratePatternPT(int sizeTheta, real *  II, real *  r, int sizeR, real complex m, real *  pii, real *  tau,
		const int nPiiTau , real wavelength, int polarization, int pattern_length) {
	//srand(time(NULL));
	int offset=0;
	const real invWavelength=1.0/wavelength;
	//	int offset = (rand()/(real)RAND_MAX)*(real)(nPiiTau - pattern_length+1);

	int realImagSize = nPiiTau*sizeR;
	real *  aReal = new real[realImagSize];
	real *  bReal = new real[realImagSize];
	real *  aImag = new real[realImagSize];
	real *  bImag = new real[realImagSize];
	real *  k     = new real[nPiiTau];
	real * correctedR = new real[sizeR];
	int * nMax = new int[sizeR];

#pragma omp parallel shared(sizeR, correctedR, r, k, nMax) default(none) 
	{
#pragma omp for nowait
	for(int j=0;j<nPiiTau;j++)
		k[j] = (2*j+3)/(real)( (j+1)*(j+2));
#pragma omp for
	for(int j=0;j<sizeR;j++)
		correctedR[j]=r[j]*2.0*M_PI*invWavelength;
#pragma omp for nowait
	for(int j=0;j<sizeR;j++)
		nMax[j] = (int)ceil(correctedR[j]+4.0*pow(correctedR[j], (real)1/3)+2.0);
	}

	calculateMieAB(nMax, nPiiTau, sizeR, m, correctedR, aReal, aImag, bReal, bImag);
#pragma omp parallel for
	for(int i=0;i<sizeR;i++) { 
		for(int j=0;j<nMax[i];j++) {
			int index = i*nPiiTau+j;
			aReal[index]*=k[j];
			bReal[index]*=k[j];
			aImag[index]*=k[j]; 
			bImag[index]*=k[j];
		}
	}
#ifdef CUDA
	if(polarization==0) cudaGenerate(sizeR, pattern_length, nMax, pii, nPiiTau, tau, aReal, aImag, bReal, bImag, II );
	else                cudaGenerate(sizeR, pattern_length, nMax, tau, nPiiTau, pii, aReal, aImag, bReal, bImag, II );
#endif

#ifndef CUDA
	if(polarization==0){
#pragma omp parallel for default(none) shared(sizeR, pattern_length, aReal, aImag, bReal, bImag, pii, tau, nMax, II)
		for(int i=0;i<sizeR;i++)
			for(int j=0;j<pattern_length;j++) {
				real realProduct = (real)0.0;
				real imagProduct = (real)0.0;
				for(int kk=0;kk<nMax[i];++kk) {
					const int index = j*nPiiTau+kk;
					realProduct += aReal[i*nPiiTau+kk]*pii[index] + bReal[i*nPiiTau+kk]*tau[index];
					imagProduct += aImag[i*nPiiTau+kk]*pii[index] + bImag[i*nPiiTau+kk]*tau[index];
				}
				II[i*pattern_length+j]= realProduct*realProduct + imagProduct*imagProduct;
			}
	}
	if(polarization==1){
#pragma omp parallel for default(none) shared(sizeR, pattern_length, aReal, aImag, bReal, bImag, pii, tau, nMax, II)
		for(int i=0;i<sizeR;i++)
			for(int j=0;j<pattern_length;j++) {
				real realProduct = 0.0;
				real imagProduct = 0.0;
				for(int kk=0;kk<nMax[i];++kk) {
					const int index = j*nPiiTau+kk;
					const int index2 = i*nPiiTau+kk;
					realProduct += aReal[index2]*tau[index] + bReal[index2]*pii[index];
					imagProduct += aImag[index2]*tau[index] + bImag[index2]*pii[index];
				}
				II[i*pattern_length+j]= realProduct*realProduct + imagProduct*imagProduct;
			}
	}
#endif
	delete [] nMax;
	delete [] aReal;
	delete [] bReal;
	delete [] aImag;
	delete [] bImag;
	delete [] k;
	delete [] correctedR;
}
void GeneratePattern(real *  II, real *  r, const int sizeR, real complex m,
		real *  theta,const int sizeTheta ,real redWavelength, real greenWavelength, int polarization ,int pattern_length) {
	real maxR=0.0;
	for(int i=0; i< sizeR; i++) 
		if( r[i] > maxR ) maxR=r[i];
	real maxDiffractionParam;
	if ( polarization == 0)
		maxDiffractionParam=maxR*2.0*M_PI/redWavelength;
	if (polarization == 1)
		maxDiffractionParam=maxR*2.0*M_PI/greenWavelength;
	const int nMax=(int)ceil(maxDiffractionParam+4.0*pow(maxDiffractionParam, (real)1/3)+2.0);
	real *  u= new real[sizeTheta];
	for(int i=0;i<sizeTheta;++i)
		u[i]=cos(theta[i]);
	real *  p =new real[sizeTheta*nMax];
	real *  t =new real[sizeTheta*nMax];
	calculateMiePT(sizeTheta, nMax, u, p, t);
	if( polarization==0 ) 
		GeneratePatternPT(sizeTheta, II, r, sizeR, m, p, t, nMax, redWavelength, 0, pattern_length); 
	if( polarization==1 )
		GeneratePatternPT(sizeTheta, II, r, sizeR, m, p, t, nMax, greenWavelength, 1, pattern_length);
	delete [] u;
	delete [] p;
	delete [] t;
}
