#include<cstdio>
#include<complex.h>
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


void mainFunction() {
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
	real diafragma, hccd_max_G, hccd_max_R;
	dummy=fread((void*)(&mIpp)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&nIpp)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&mIss)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&nIss)  , sizeof(int), 1, pFile);
	dummy=fread((void*)(&sizeTp), sizeof(int), 1, pFile);
	dummy=fread((void*)(&sizeTs), sizeof(int), 1, pFile);
	dummy=fread((void*)(&diafragma) , sizeof(real), 1, pFile);
	dummy=fread((void*)(&hccd_max_G), sizeof(real), 1, pFile);
	dummy=fread((void*)(&hccd_max_R), sizeof(real), 1, pFile);
	fclose(pFile);
#ifndef CUDA
 real * ipp = new real[mIpp*nIpp];
 real * iss = new real[mIss*nIss];
#endif
#ifdef CUDA
 real * ipp;
 real * iss;
 allocCudaPointer(&ipp, mIpp*nIpp*sizeof(real));
 allocCudaPointer(&iss, mIss*nIss*sizeof(real));
#endif
	real * mTp = new real[sizeTp];
	real * mTs = new real[sizeTs];

	pFile = fopen("Cdata/ipp", "rb");
	dummy=fread((void*)ipp, sizeof(real), mIpp*nIpp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/iss", "rb");
	dummy=fread((void*)iss, sizeof(real), mIss*nIss,pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTp", "rb");
	dummy=fread((void*)mTp, sizeof(real), sizeTp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTs", "rb");
	dummy=fread((void*)mTs, sizeof(real), sizeTs, pFile);
	fclose(pFile);

	real complex mR;
	real complex mG;
	real complex shiftM;
	real complex shiftMRed;
	real rMin, rMax, rStep, wavelengthR, wavelengthG;
	real scale;
	real shiftG;
	real shiftR;
	int frameBegin;
	int frameEnd;
	int frameStep;
	// Reading the setup file
	pFile=fopen("Cdata/setup","rb");
	dummy=fread((void*)(&mR), 2*sizeof(real), 1, pFile);
	dummy=fread((void*)(&mG), 2*sizeof(real), 1, pFile);
	dummy=fread((void*)(&rMin), sizeof(real), 1, pFile);
	dummy=fread((void*)(&rMax), sizeof(real), 1, pFile);
	dummy=fread((void*)(&rStep), sizeof(real), 1, pFile);
	dummy=fread((void*)(&scale), sizeof(real), 1, pFile);
	dummy=fread((void*)(&shiftR), sizeof(real), 1, pFile);
	dummy=fread((void*)(&shiftG), sizeof(real), 1, pFile);
	dummy=fread((void*)(&shiftM), 2*sizeof(real), 1, pFile);
	dummy=fread((void*)(&shiftMRed), 2*sizeof(real), 1, pFile);
	dummy=fread((void*)(&frameBegin), sizeof(int), 1, pFile);
	dummy=fread((void*)(&frameStep), sizeof(int), 1, pFile);
	dummy=fread((void*)(&frameEnd), sizeof(int), 1, pFile);
	dummy=fread((void*)(&wavelengthR), sizeof(real), 1, pFile);
	dummy=fread((void*)(&wavelengthG), sizeof(real), 1, pFile);
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
	shiftR = shiftR * 2.0 * M_PI /(real)360; //Now shifts in RAD
	shiftG = shiftG * 2.0 * M_PI /(real)360;

	int rSize=(int)floor( (rMax-rMin)/rStep)+1;
	rSize = (rSize+3)/4*4; // UWAGA TUTAJ ZWIEKSZAM ZAKRES R ZEBY SIE DZIELILO PRZEZ 4
	real *r = new real[rSize];
	for(int i=0;i<rSize;++i) 
		r[i]=rMin+rStep*i;


	real * Tp = new real[sizeTp];
	real * Ts = new real[sizeTs];
	real * rrp = new real[sizeTp];
	real * rrs = new real[sizeTs];
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
	real * Ittp = new real[rSize*sizeTp];
	real * Itts = new real[rSize*sizeTs];
#endif
#ifdef CUDA
	real * Ittp;
	real * Itts;
	allocCudaPointer(&Ittp, rSize*sizeTp*sizeof(real));
	allocCudaPointer(&Itts, rSize*sizeTs*sizeof(real));
#endif 


	real complex *m = new  real complex[rSize];
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
	real * errp = new real[mIpp * rSize];
	real * errs = new real[mIss * rSize];
#endif
#ifdef CUDA
	real * errp;
	real * errs;
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
	real * err = new real[mIpp*rSize];
	real * inv_median_errp = new real[mIpp];
	real * inv_median_errs= new real[mIss];
	real * errpSorted = new real[mIpp*rSize];
	real * errsSorted = new real[mIss*rSize];
	memcpy(errpSorted, errp, mIpp*rSize*sizeof(real));
	memcpy(errsSorted, errs, mIss*rSize*sizeof(real));
#pragma omp parallel
	{
#pragma omp for
		for(int i=0;i<mIpp;++i) {
			qsort(errpSorted+i*rSize, rSize, sizeof(real), compare);
			inv_median_errp[i]=1.0/errpSorted[i*rSize+rSize/2];
		}
#pragma omp for nowait
		for(int i=0;i<mIss;++i) {
			qsort(errsSorted+i*rSize, rSize, sizeof(real), compare);
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
				real value=errp[i*rSize];
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
				real value=errs[i*rSize];
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
				real value=err[i*rSize];
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

	real * rIrmp = new real[mIpp];
	real * rIrms = new real[mIpp];
	real * rIrm  = new real[mIpp];
	for(int i=0;i<mIpp;++i) {
		rIrmp[i] = r[irmp[i]];
		rIrms[i] = r[irms[i]];
		rIrm[i]  = r[irm[i]];
	}

	//Saving the results
	pFile = fopen("Cdata/results", "wb");
	dummy=fwrite((void*)(&mIpp), sizeof(int), 1, pFile);
	dummy=fwrite((void*)rIrmp, sizeof(real), mIpp, pFile);
	dummy=fwrite((void*)rIrms, sizeof(real), mIpp, pFile);
	dummy=fwrite((void*)rIrm,  sizeof(real), mIpp, pFile);
	fclose(pFile);

	delete [] rIrmp;
	delete [] rIrms;
	delete [] rIrm;
	delete [] irmp;
	delete [] irms;
	delete [] irm;
	delete [] r;
	delete [] m;
}
