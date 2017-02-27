#include<cstdio>
#include<complex>
#include<cstdlib>
#include<ctime>
#include<cmath>
#include<omp.h>
#include<cstring>
#include"auxillary.h"
#include"MiePT.h"
#include"MieAB.h"
#include"ReferenceDistance.h"
#include"GeneratePattern.h"
#include"Calculate_m.h"
#include"RunningRadius.h"
#include"cudaReferenceDistance.h"
#include"globals.h"
#ifdef CUDA
#include"ccuda.h"
#endif

using namespace std;

int main() {
#ifdef TIMING
#ifdef _OPENMP
	double init;
	init=omp_get_wtime();
#else
	clock_t init;
	init=clock();
#endif
#endif
	// Read data from files, size, ipp, iss, mTp, mTs, diafragma, hccd's
	size_t dummy;
	FILE * pFile;
	pFile = fopen("Cdata/setupMatlab","rb");
	int mIpp, nIpp, mIss, nIss, sizeTp, sizeTs; //mIppi=mIss to liczba klatek filmu
	float diafragma, hccd_max_G, hccd_max_R;
	dummy=fread((void*)(&mIpp)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&nIpp)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&mIss)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&nIss)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&sizeTp), sizeof(int), 1, pFile);
	dummy=fread((void*)(&sizeTs), sizeof(int), 1, pFile);
	dummy=fread((void*)(&diafragma) , sizeof(float), 1, pFile);
	dummy=fread((void*)(&hccd_max_G), sizeof(float), 1, pFile);
	dummy=fread((void*)(&hccd_max_R), sizeof(float), 1, pFile);
	fclose(pFile);
#ifndef CUDA
 float * ipp = new float[mIpp*nIpp];
 float * iss = new float[mIss*nIss];
#endif
#ifdef CUDA
 float * ipp;
 float * iss;
 allocCudaPointer(&ipp, mIpp*nIpp*sizeof(float));
 allocCudaPointer(&iss, mIss*nIss*sizeof(float));
#endif
	float * mTp = new float[sizeTp];
	float * mTs = new float[sizeTs];

	pFile = fopen("Cdata/ipp", "rb");
	dummy=fread((void*)ipp, sizeof(float), mIpp*nIpp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/iss", "rb");
	dummy=fread((void*)iss, sizeof(float), mIss*nIss,pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTp", "rb");
	dummy=fread((void*)mTp, sizeof(float), sizeTp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTs", "rb");
	dummy=fread((void*)mTs, sizeof(float), sizeTs, pFile);
	fclose(pFile);

	complex<float> mR;
	complex<float> mG;
	complex<float> shiftM;
	complex<float> shiftMRed;
	float rMin, rMax, rStep, wavelengthR, wavelengthG;
	float scale;
	float shiftG;
	float shiftR;
	int frameBegin;
	int frameEnd;
	int frameStep;
	// Reading the setup file
	pFile=fopen("Cdata/setup","rb");
	dummy=fread((void*)(&mR), 2*sizeof(float), 1, pFile);
	dummy=fread((void*)(&mG), 2*sizeof(float), 1, pFile);
	dummy=fread((void*)(&rMin), sizeof(float), 1, pFile);
	dummy=fread((void*)(&rMax), sizeof(float), 1, pFile);
	dummy=fread((void*)(&rStep), sizeof(float), 1, pFile);
	dummy=fread((void*)(&scale), sizeof(float), 1, pFile);
	dummy=fread((void*)(&shiftR), sizeof(float), 1, pFile);
	dummy=fread((void*)(&shiftG), sizeof(float), 1, pFile);
	dummy=fread((void*)(&shiftM), 2*sizeof(float), 1, pFile);
	dummy=fread((void*)(&shiftMRed), 2*sizeof(float), 1, pFile);
	dummy=fread((void*)(&frameBegin), sizeof(int), 1, pFile);
	dummy=fread((void*)(&frameStep), sizeof(int), 1, pFile);
	dummy=fread((void*)(&frameEnd), sizeof(int), 1, pFile);
	dummy=fread((void*)(&wavelengthR), sizeof(float), 1, pFile);
	dummy=fread((void*)(&wavelengthG), sizeof(float), 1, pFile);
	fclose(pFile);

	frameBegin-=1;

#ifdef TIMING 
#ifdef _OPENMP 
	printf("%f CZAS WCZYTYWANIA PLIKOW\n", omp_get_wtime()-init);
#else
	printf("%f CZAS WCZYTYWANIA PLIKOW\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC);
#endif 
#endif 

#ifdef TIMING
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
	shiftR = shiftR * 2.0 * M_PI /(float)360; //Now shifts in RAD
	shiftG = shiftG * 2.0 * M_PI /(float)360;

	int rSize=(int)floor( (rMax-rMin)/rStep)+1;
	rSize = (rSize+3)/4*4; // UWAGA TUTAJ ZWIEKSZAM ZAKRES R ZEBY SIE DZIELILO PRZEZ 4
	float *r = new float[rSize];
	for(int i=0;i<rSize;++i) 
		r[i]=rMin+rStep*i;


	float * Tp = new float[sizeTp];
	float * Ts = new float[sizeTs];
	float * rrp = new float[sizeTp];
	float * rrs = new float[sizeTs];
#pragma omp parallel sections 
	{
#pragma omp section
		{
			for(int i=0;i<sizeTp;++i)
				Tp[i]=atan ( tan( mTp[i] - M_PI*0.5 ) * scale ) + M_PI*0.5 + shiftR;
			RunningRadius( rrp, Tp, sizeTp, hccd_max_R, diafragma, wavelengthR );
		}
#pragma omp section
		{
			for(int i=0;i<sizeTs;++i)
				Ts[i]=atan ( tan( mTs[i] - M_PI*0.5 ) * scale ) + M_PI*0.5 + shiftG;
			RunningRadius( rrs, Ts, sizeTs, hccd_max_G, diafragma, wavelengthG ); 
		}
	}

	delete [] mTp;
	delete [] mTs;
	mTp=mTs=0;
#ifdef TIMING
#ifdef _OPENMP
	printf("%f CZAS RUNNING RADIUS\n", omp_get_wtime()-init);
#else
	printf("%f CZAS RUNNING RADIUS\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC);
#endif
#endif
#ifdef TIMING
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
#ifndef CUDA
	float * Ittp = new float[rSize*sizeTp];
	float * Itts = new float[rSize*sizeTs];
#endif
#ifdef CUDA
	float * Ittp;
	float * Itts;
	allocCudaPointer(&Ittp, rSize*sizeTp*sizeof(float));
	allocCudaPointer(&Itts, rSize*sizeTs*sizeof(float));
#endif 


	complex<float> *m = new  complex<float>[rSize];
	GeneratePattern( Ittp, r, rSize, mR, Tp, sizeTp, wavelengthR, wavelengthG, 0, sizeTp);
	GeneratePattern( Itts, r, rSize, mG, Ts, sizeTs, wavelengthR, wavelengthG, 1, sizeTs);
	delete [] Tp;
	delete [] Ts;
	Tp=0;
	Ts=0;
#ifdef TIMING
#ifdef _OPENMP
	printf("%f CZAS GENERATE PATTERN\n", omp_get_wtime()-init);
#else
	printf("%f CZAS GENERATE PATTERN\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC);
#endif
#endif
	for(int i=0;i<rSize;++i)
		for(int j=0;j<sizeTp;++j)
			Ittp[i*sizeTp+j]*=rrp[j];
	for(int i=0;i<rSize;++i)
		for(int j=0;j<sizeTs;++j)
			Itts[i*sizeTs+j]*=rrs[j];
	delete [] rrp;
	delete [] rrs;
	rrs=rrp=0;

#ifndef CUDA
	float * errp = new float[mIpp * rSize];
	float * errs = new float[mIss * rSize];
#endif
#ifdef CUDA
	float * errp;
	float * errs;
#endif


#ifdef TIMING
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
#ifndef CUDA
	ReferenceDistance( errp, ipp, Ittp, mIpp, nIpp, rSize, sizeTp);
	ReferenceDistance( errs, iss, Itts, mIss, nIss, rSize, sizeTs);
#endif
#ifdef CUDA
	cudaReferenceDistance( &errp, ipp, Ittp, mIpp, nIpp, rSize, sizeTp);
	cudaReferenceDistance( &errs, iss, Itts, mIss, nIss, rSize, sizeTs);
#endif
#ifdef TIMING
#ifdef _OPENMP
	printf("%f CZAS REFERENCE DISTANCE\n", omp_get_wtime()-init);
#else
	printf("%f CZAS REFERENCE DISTANCE\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC);
#endif
#endif
	//printVector(errp,10);
#ifndef CUDA
	delete [] ipp;
	delete [] iss;
	delete [] Ittp;
	delete [] Itts;
#endif
#ifdef CUDA
	freeCudaPointer(&Ittp);
	freeCudaPointer(&Itts);
	freeCudaPointer(&ipp);
	freeCudaPointer(&iss);
#endif
	iss=Ittp=Itts=ipp=0;

#ifdef TIMING
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
	float * err = new float[mIpp*rSize];
	float * inv_median_errp = new float[mIpp];
	float * inv_median_errs= new float[mIss];
	float * errpSorted = new float[mIpp*rSize];
	float * errsSorted = new float[mIss*rSize];
	memcpy(errpSorted, errp, mIpp*rSize*sizeof(float));
	memcpy(errsSorted, errs, mIss*rSize*sizeof(float));
#pragma omp parallel
	{
#pragma omp for
		for(int i=0;i<mIpp;++i) {
			qsort(errpSorted+i*rSize, rSize, sizeof(float), compare);
			inv_median_errp[i]=1.0/errpSorted[i*rSize+rSize/2];
		}
#pragma omp for nowait
		for(int i=0;i<mIss;++i) {
			qsort(errsSorted+i*rSize, rSize, sizeof(float), compare);
			inv_median_errs[i]=1.0/errsSorted[i*rSize+rSize/2];
		}
	}

	for(int j=0;j<mIpp;++j)
		for(int i=0;i<rSize;++i)
			err[j*rSize+i] = errp[j*rSize+i] * inv_median_errp[j] * errs[j*rSize+i] * inv_median_errs[j];

	delete [] inv_median_errp;
	delete [] inv_median_errs;
	delete [] errpSorted;
	delete [] errsSorted;
	errpSorted=errsSorted=inv_median_errs=inv_median_errp=0;
	int * irmp = new int[mIpp];
	int * irms = new int[mIss];
	int * irm  = new int[mIpp];
	//Finding the minimal values
#pragma omp parallel sections
	{
#pragma omp section
		{

			for(int i=0;i<mIpp;++i) {
				int index=0;
				float value=errp[i*rSize];
				for(int j=0;j<rSize;++j) {
					if( errp[i*rSize+j] < value ) {
						value = errp[i*rSize+j];
						index = j;
					}
				}
				irmp[i] = index;
			}
		}
#pragma omp section
		{

			for(int i=0;i<mIss;++i) {
				int index=0;
				float value=errs[i*rSize];
				for(int j=0;j<rSize;++j) {
					if( errs[i*rSize +j] < value ) {
						value = errs[i*rSize+j];
						index = j;
					}
				}
				irms[i] = index;
			}
		}
#pragma omp section
		{

			for(int i=0;i<mIss;++i) {
				int index=0;
				float value=err[i*rSize];
				for(int j=0;j<rSize;++j) {
					if( err[i*rSize+j] < value ) {
						value = err[i*rSize+j];
						index = j;
					}
				}
				irm[i] = index;
			}
		}
	}
#ifndef CUDA
	delete [] errp;
	delete [] errs;
#endif
#ifdef CUDA
	freeCudaPointer(&errp);
	freeCudaPointer(&errs);
#endif
	delete [] err;
	errp=errs=err=0;

#ifdef TIMING
#ifdef _OPENMP
	printf("%f CZAS SZUKANIA MINIMOW\n", omp_get_wtime()-init);
#else
	printf("%f CZAS SZUKANIA MINIMOW\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC);
#endif
#endif

	float * rIrmp = new float[mIpp];
	float * rIrms = new float[mIpp];
	float * rIrm  = new float[mIpp];
	for(int i=0;i<mIpp;++i) {
		rIrmp[i] = r[irmp[i]];
		rIrms[i] = r[irms[i]];
		rIrm[i]  = r[irm[i]];
	}

	//Saving the results
	pFile = fopen("Cdata/results", "wb");
	dummy=fwrite((void*)(&mIpp), sizeof(int), 1, pFile);
	dummy=fwrite((void*)rIrmp, sizeof(float), mIpp, pFile);
	dummy=fwrite((void*)rIrms, sizeof(float), mIpp, pFile);
	dummy=fwrite((void*)rIrm,  sizeof(float), mIpp, pFile);
	fclose(pFile);

	delete [] rIrmp;
	delete [] rIrms;
	delete [] rIrm;
	delete [] irmp;
	delete [] irms;
	delete [] irm;
	delete [] r;
	delete [] m;
	return 0;
}
