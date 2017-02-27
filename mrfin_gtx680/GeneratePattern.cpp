#include<complex>
#include<cstdio>
#include<cmath>
#include<omp.h>
#include"MieAB.h"
#include"MiePT.h"
#include"auxillary.h"
#include"cudaGenerate.h"
#include"globals.h"

using namespace std;

inline void GeneratePatternPT(int sizeTheta, float *  II, float *  r, int sizeR, complex<float> const * const m, float *  pii, float *  tau,
		const int nPiiTau , float wavelength, int polarization, int pattern_length) {
	//srand(time(NULL));
	int offset=0;
	const float invWavelength=1.0/wavelength;
	//	int offset = (rand()/(float)RAND_MAX)*(float)(nPiiTau - pattern_length+1);

	int floatImagSize = nPiiTau*sizeR;
	float *  afloat = new float[floatImagSize];
	float *  bfloat = new float[floatImagSize];
	float *  aImag = new float[floatImagSize];
	float *  bImag = new float[floatImagSize];
	float *  k     = new float[nPiiTau];
	float * correctedR = new float[sizeR];
	int * nMax = new int[sizeR];

#pragma omp parallel shared(sizeR, correctedR, r, k, nMax) default(none) 
	{
#pragma omp for nowait
	for(int j=0;j<nPiiTau;j++)
		k[j] = (2*j+3)/(float)( (j+1)*(j+2));
#pragma omp for
	for(int j=0;j<sizeR;j++)
		correctedR[j]=r[j]*2.0*M_PI*invWavelength;
#pragma omp for nowait
	for(int j=0;j<sizeR;j++)
		nMax[j] = (int)ceil(correctedR[j]+4.0*pow(correctedR[j], (float)1/3)+2.0);
		//nMax[j] = (int)ceil(correctedR[((j+3)/4)*4]+4.0*pow(correctedR[((j+3)/4)*4], (float)1/3)+2.0);
	}

	calculateMieAB(nMax, nPiiTau, sizeR, m, correctedR, afloat, aImag, bfloat, bImag);
#pragma omp parallel for
	for(int i=0;i<sizeR;i++) { 
		for(int j=0;j<nMax[i];j++) {
			int index = i*nPiiTau+j;
			afloat[index]*=k[j];
			bfloat[index]*=k[j];
			aImag[index]*=k[j]; 
			bImag[index]*=k[j];
		}
	}
	//for(int i=0;i<sizeR;i++)
		//printf("%d\n", nMax[i]);
#ifdef CUDA
	if(polarization==0) cudaGenerate(sizeR, pattern_length, nMax, pii, nPiiTau, tau, afloat, aImag, bfloat, bImag, II, polarization );
	else                cudaGenerate(sizeR, pattern_length, nMax, tau, nPiiTau, pii, afloat, aImag, bfloat, bImag, II, polarization );
#endif

#ifndef CUDA
	if(polarization==0){
#pragma omp parallel for default(none) shared(sizeR, pattern_length, afloat, aImag, bfloat, bImag, pii, tau, nMax, II)
		for(int i=0;i<sizeR;i++)
			for(int j=0;j<pattern_length;j++) {
				float floatProduct = (float)0.0;
				float imagProduct = (float)0.0;
				for(int kk=0;kk<nMax[i];++kk) {
					const int index = j*nPiiTau+kk;
					floatProduct += afloat[i*nPiiTau+kk]*pii[index] + bfloat[i*nPiiTau+kk]*tau[index];
					imagProduct += aImag[i*nPiiTau+kk]*pii[index] + bImag[i*nPiiTau+kk]*tau[index];
				}
				II[i*pattern_length+j]= floatProduct*floatProduct + imagProduct*imagProduct;
			}
	}
	if(polarization==1){
#pragma omp parallel for default(none) shared(sizeR, pattern_length, afloat, aImag, bfloat, bImag, pii, tau, nMax, II)
		for(int i=0;i<sizeR;i++)
			for(int j=0;j<pattern_length;j++) {
				float floatProduct = 0.0;
				float imagProduct = 0.0;
				for(int kk=0;kk<nMax[i];++kk) {
					const int index = j*nPiiTau+kk;
					const int index2 = i*nPiiTau+kk;
					floatProduct += afloat[index2]*tau[index] + bfloat[index2]*pii[index];
					imagProduct += aImag[index2]*tau[index] + bImag[index2]*pii[index];
				}
				II[i*pattern_length+j]= floatProduct*floatProduct + imagProduct*imagProduct;
			}
	}
#endif
	//printVector(II,10);
	//printVector(II+pattern_length, 10);
//#ifdef DEBUG
//for(int i=0;i<sizeR;i++)
	//for(int j = 0; j<pattern_length; j++)
		//fprintf(stderr, "II %d %d %.2e\n", i, j, II[i*pattern_length+j]);


//#endif


	delete [] nMax;
	delete [] afloat;
	delete [] bfloat;
	delete [] aImag;
	delete [] bImag;
	delete [] k;
	delete [] correctedR;
}
void GeneratePattern(float *  II, float *  r, const int sizeR, complex<float> const * const m,
		float *  theta,const int sizeTheta ,float redWavelength, float greenWavelength, int polarization ,int pattern_length) {
	float maxR=0.0;
	for(int i=0; i< sizeR; i++) 
		if( r[i] > maxR ) maxR=r[i];
	float maxDiffractionParam;
	if ( polarization == 0)
		maxDiffractionParam=maxR*2.0*M_PI/redWavelength;
	if (polarization == 1)
		maxDiffractionParam=maxR*2.0*M_PI/greenWavelength;
	const int nMax=(int)ceil(maxDiffractionParam+4.0*pow(maxDiffractionParam, (float)1/3)+2.0);
	float *  u= new float[sizeTheta];
	for(int i=0;i<sizeTheta;++i)
		u[i]=cos(theta[i]);
	float *  p =new float[sizeTheta*nMax];
	float *  t =new float[sizeTheta*nMax];
	calculateMiePT(sizeTheta, nMax, u, p, t);
	if( polarization==0 ) 
		GeneratePatternPT(sizeTheta, II, r, sizeR, m, p, t, nMax, redWavelength, 0, pattern_length); 
	if( polarization==1 )
		GeneratePatternPT(sizeTheta, II, r, sizeR, m, p, t, nMax, greenWavelength, 1, pattern_length);
	delete [] u;
	delete [] p;
	delete [] t;
}
