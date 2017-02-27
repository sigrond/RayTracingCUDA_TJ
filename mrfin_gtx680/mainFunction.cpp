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
#include"globals.h"
#include<algorithm>
#ifdef CUDA
#include"ccuda.h"
#include"cudaReferenceDistance.h"
#include"cudaFindMin.h"
#endif

#define readVariable(a1,a2,a3,a4) __readVariable(a1, a2, a3, a4, __FILE__, __LINE__, #a1)
#define writeVariable(a1,a2,a3,a4) __writeVariable(a1, a2, a3, a4, __FILE__, __LINE__, #a1)

using namespace std;

extern bool noRunningRadius;

void mainFunction() {
	double init;
	startTimer(init);
/*********************************************************************
 *                       Read data from files                        *
 *********************************************************************/
	size_t dummy;
	FILE * pFile=NULL;
	pFile = fopen("Cdata/setupMatlab","rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/setupMatlab\n");
		return;
	}
	int mIpp, nIpp, mIss, nIss, sizeTp, sizeTs; //mIppi=mIss to liczba klatek filmu
	float diafragma, hccd_max_G, hccd_max_R;
	readVariable((void*)(&mIpp)       , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&nIpp)       , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&mIss)       , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&nIss)       , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&sizeTp)     , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&sizeTs)     , sizeof(int)  , 1 , pFile);
	readVariable((void*)(&diafragma)  , sizeof(float) , 1 , pFile);
	readVariable((void*)(&hccd_max_G) , sizeof(float) , 1 , pFile);
	readVariable((void*)(&hccd_max_R) , sizeof(float) , 1 , pFile);
	fclose(pFile);

	float * ipp;
	float * iss;
	allocMemory((void**)&ipp, mIpp*nIpp*sizeof(float));
	allocMemory((void**)&iss, mIss*nIss*sizeof(float));
	float * mTp = new float[sizeTp];
	float * mTs = new float[sizeTs];

	pFile = fopen("Cdata/ipp", "rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/ipp\n");
		return;
	}
	readVariable((void*)ipp, sizeof(float), mIpp*nIpp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/iss", "rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/iss\n");
		return;
	}
	readVariable((void*)iss, sizeof(float), mIss*nIss,pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTp", "rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/mTp\n");
		return;
	}
	readVariable((void*)mTp, sizeof(float), sizeTp, pFile);
	fclose(pFile);
	pFile = fopen("Cdata/mTs", "rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/mTs\n");
		return;
	}
	readVariable((void*)mTs, sizeof(float), sizeTs, pFile);
	fclose(pFile);

	//complex<float> mR;
	//complex<float> mG;
	complex<float> shiftM;
	complex<float> shiftMRed;
	float rMin, rMax, rStep, wavelengthR, wavelengthG;
	float scale;
	float shiftG;
	float shiftR;
	int frameBegin;
	int frameEnd;
	int frameStep;
	int externalM;
	complex<float> * mR;
	complex<float> * mG;
	complex<float>  mRsingle;
	complex<float>  mGsingle;
/*********************************************************************
 *                            Read setup                             *
 *********************************************************************/

	pFile=fopen("Cdata/setup","rb");
	if (pFile == NULL)
	{
		printf("error opening Cdata/setup\n");
		return;
	}
	readVariable((void*)(&externalM)   ,  sizeof(int)    , 1 , pFile); 
	if ( externalM != 1 ) {
		printf("Wczytuje pojedyncze wspolczynniki zalamania\n");
		readVariable((void*)(&mRsingle)          , 2*sizeof(float) , 1 , pFile);
		readVariable((void*)(&mGsingle)          , 2*sizeof(float) , 1 , pFile);
	}
	readVariable((void*)(&rMin)        , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&rMax)        , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&rStep)       , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&scale)       , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&shiftR)      , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&shiftG)      , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&shiftM)      , 2*sizeof(float) , 1 , pFile);
	readVariable((void*)(&shiftMRed)   , 2*sizeof(float) , 1 , pFile);
	readVariable((void*)(&frameBegin)  , sizeof(int)    , 1 , pFile);
	readVariable((void*)(&frameStep)   , sizeof(int)    , 1 , pFile);
	readVariable((void*)(&frameEnd)    , sizeof(int)    , 1 , pFile);
	readVariable((void*)(&wavelengthR) , sizeof(float)   , 1 , pFile);
	readVariable((void*)(&wavelengthG) , sizeof(float)   , 1 , pFile);
	fclose(pFile);

	int rSize=(int)floor( (rMax-rMin)/rStep)+1;
	rSize = (rSize+3)/4*4; // UWAGA TUTAJ ZWIEKSZAM ZAKRES R ZEBY SIE DZIELILO PRZEZ 4
	mR =(complex<float> *) malloc( rSize*sizeof(float)*2 );
	mG =(complex<float> *) malloc( rSize*sizeof(float)*2 );

	if ( externalM == 1 ) {
		printf("Wczytuje wspolczynniki zalamania z pliku\n");
		pFile = fopen("Cdata/mR" , "rb");
		if (pFile == NULL)
		{
			printf("error opening Cdata/mR\n");
			return;
		}
		readVariable((void*)mR, 2*sizeof(float), rSize, pFile);
		fclose(pFile);

		pFile = fopen("Cdata/mG", "rb");
		if (pFile == NULL)
		{
			printf("error opening Cdata/mG\n");
			return;
		}
		readVariable((void*)mG, 2*sizeof(float), rSize, pFile);
		fclose(pFile);

	}
	else {
		for(int i=0; i<rSize; ++i) {
			mR[i] = mRsingle;
			mG[i] = mGsingle;
		}
	}

	frameBegin-=1;

	stopTimer("CZAS WCZYTYWANIA PLIKOW", init);
/*********************************************************************
 *                          Running Radius                           *
 *********************************************************************/
	startTimer(init);
	shiftR = shiftR * 2.0 * M_PI /(float)360; //Now shifts in RAD
	shiftG = shiftG * 2.0 * M_PI /(float)360;

	float *r = new float[rSize];
	for(int i=0;i<rSize;++i) 
		r[i]=rMin+rStep*i;


	float * Tp  = new float[sizeTp];
	float * Ts  = new float[sizeTs];
	float * rrp = new float[sizeTp];
	float * rrs = new float[sizeTs];
	if (!noRunningRadius)
	{
		//#pragma omp parallel sections //TODO: sprawdzic czy to cos przyspiesza (04.04.13 by szmigacz)
		//{
		//#pragma omp section
		//{
		for (int i = 0; i < sizeTp; ++i)
			Tp[i] = atan(tan(mTp[i] - M_PI*0.5) * scale) + M_PI*0.5 + shiftR;
		RunningRadius(rrp, Tp, sizeTp, hccd_max_R, diafragma, wavelengthR);
		//}
		//#pragma omp section
		//{
		for (int i = 0; i < sizeTs; ++i)
			Ts[i] = atan(tan(mTs[i] - M_PI*0.5) * scale) + M_PI*0.5 + shiftG;
		RunningRadius(rrs, Ts, sizeTs, hccd_max_G, diafragma, wavelengthG);
		//}
		//}
	}

	for(int i=0; i<sizeTp; i++) 
		if( rrp[i]!=rrp[i] ) rrp[i]=1.0;
	for(int i=0; i<sizeTs; i++)
		if( rrs[i]!=rrs[i] ) rrs[i]=1.0;


	delete [] mTp;
	delete [] mTs;
	mTp=mTs=0;
	stopTimer("CZAS RUNNING RADIUS", init);
	startTimer(init);
/*********************************************************************
 *                         Generate pattern                          *
 *********************************************************************/
	float * Ittp;
	float * Itts;
	allocMemory((void**)&Ittp, rSize*sizeTp*sizeof(float));
	allocMemory((void**)&Itts, rSize*sizeTs*sizeof(float));


	complex<float> *m = new  complex<float>[rSize];
	GeneratePattern( Ittp, r, rSize, mR, Tp, sizeTp, wavelengthR, wavelengthG, 0, sizeTp);
	GeneratePattern( Itts, r, rSize, mG, Ts, sizeTs, wavelengthR, wavelengthG, 1, sizeTs);

	#ifdef CUDA
		freeCudaMemory();
	#endif //CUDA
	delete [] Tp;
	delete [] Ts;
	Tp=0;
	Ts=0;
	stopTimer("CZAS GENERATE PATTERN", init);
	if (noRunningRadius)
	{
		for (int i = 0; i < rSize; ++i)
			for (int j = 0; j < sizeTp; ++j)
				Ittp[i*sizeTp + j] *= rrp[j];
		for (int i = 0; i < rSize; ++i)
			for (int j = 0; j < sizeTs; ++j)
				Itts[i*sizeTs + j] *= rrs[j];
	}
	delete [] rrp;
	delete [] rrs;
	rrs=rrp=0;

/*********************************************************************
 *                        Reference distance                         *
 *********************************************************************/

	startTimer(init);
	int * irmp;
	int * irms;
	int * irm;
	allocMemory((void**)&irmp , mIpp*sizeof(int));
	allocMemory((void**)&irms , mIss*sizeof(int));
	allocMemory((void**)&irm  , mIpp*sizeof(int));

#ifndef CUDA
	float * errp;
	float * errs;
	allocMemory((void**)&errp, mIpp*rSize*sizeof(float));
	allocMemory((void**)&errs, mIss*rSize*sizeof(float));
	float * inv_median_errp = new float[mIpp];
	float * inv_median_errs = new float[mIss];
	ReferenceDistance( errp, ipp, Ittp, mIpp, nIpp, rSize, sizeTp);
	ReferenceDistance( errs, iss, Itts, mIss, nIss, rSize, sizeTs);
	float * err        = new float[mIpp*rSize];
	float * errpSorted = new float[mIpp*rSize];
	float * errsSorted = new float[mIss*rSize];
	memcpy(errpSorted, errp, mIpp*rSize*sizeof(float));
	memcpy(errsSorted, errs, mIss*rSize*sizeof(float));
#pragma omp parallel for
	for(int i=0;i<mIpp;++i) {
		//qsort(errpSorted+i*rSize, rSize, sizeof(float), compare);
		std::sort(errpSorted+i*rSize, errpSorted+i*rSize + rSize);
		inv_median_errp[i]=1.0/errpSorted[i*rSize+rSize/2];
	}
#pragma omp parallel for
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
#pragma omp parallel for
	for(int i=0;i<mIss;++i) {
		//qsort(errsSorted+i*rSize, rSize, sizeof(float), compare);
		std::sort(errsSorted+i*rSize, errsSorted+i*rSize+rSize);
		inv_median_errs[i]=1.0/errsSorted[i*rSize+rSize/2];
	}
#pragma omp parallel for
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
	for(int j=0;j<mIpp;++j)
		for(int i=0;i<rSize;++i)
			err[j*rSize+i] = errp[j*rSize+i] * inv_median_errp[j] * errs[j*rSize+i] * inv_median_errs[j];
#pragma omp parallel for
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
	delete [] inv_median_errp;
	delete [] inv_median_errs;
	delete [] errpSorted;
	delete [] errsSorted;
	errpSorted=errsSorted=inv_median_errs=inv_median_errp=0;
	freeMemory((void**)&errp);
	freeMemory((void**)&errs);
	delete [] err;
	err=0;

#endif
#ifdef CUDA
	mallocCudaReferences(0, mIpp, nIpp, rSize, sizeTp);
	mallocCudaReferences(1, mIss, nIss, rSize, sizeTs);
	cudaReferenceDistance(  ipp, Ittp, irmp,  mIpp, nIpp, rSize, sizeTp, 0);
	cudaReferenceDistance(  iss, Itts, irms,  mIss, nIss, rSize, sizeTs, 1);
	cuda1stPolarizationSync();
	cudaFindMin( irm, 2, rSize, mIss); 
	freeCudaMemoryMin();
	freeCudaRefMemory();
	cudaFinalize();
#endif

#ifdef DEBUG2
	for(int i=0 ; i< rSize; i++)
		for(int j=0; j < mIpp; j++)
			fprintf(stderr, "errp %d %d %.4e\n", i ,j , errp[i*mIpp+j]);

	for(int i=0 ; i< rSize; i++)
		for(int j=0; j < mIss; j++)
			fprintf(stderr, "errs %d %d %.4e\n", i ,j , errs[i*mIss+j]);

#endif //DEBUG2

	freeMemory((void** ) &ipp  ) ;
	freeMemory((void** ) &iss  ) ;
	freeMemory((void** ) &Ittp ) ;
	freeMemory((void** ) &Itts ) ;

	stopTimer("CZAS SZUKANIA MINIMOW I REF DISTANCE", init);

	float * rIrmp = new float[mIpp];
	float * rIrms = new float[mIpp];
	float * rIrm  = new float[mIpp];
	for(int i=0;i<mIpp;++i) {
		rIrmp[i] = r[irmp[i]] ; 
		rIrms[i] = r[irms[i]] ; 
		rIrm[i]  = r[irm[i]]  ; 
	}

/*********************************************************************
 *                        Saving the results                         *
 *********************************************************************/
	pFile = fopen("Cdata/results", "wb");
	writeVariable((void*)(&mIpp) , sizeof(int)  , 1    , pFile);
	writeVariable((void*)rIrmp   , sizeof(float) , mIpp , pFile);
	writeVariable((void*)rIrms   , sizeof(float) , mIpp , pFile);
	writeVariable((void*)rIrm    , sizeof(float) , mIpp , pFile);
	fclose(pFile);
#ifdef DEBUG
	//fprintf(stderr, "mIpp %d\n", mIpp);
	//for(int i=0; i<mIpp; i++) {
	//	fprintf(stderr, "rIrmp rIrms rIrm %d %.1f %.1f %.1f\n", i , rIrmp[i], rIrms[i], rIrm[i]);
	//}
#endif

	delete [] rIrmp ; 
	delete [] rIrms ; 
	delete [] rIrm  ; 
	delete [] r     ; 
	delete [] m     ; 
	free(mR);
	free(mG);
	freeMemory((void **)&irmp) ; 
	freeMemory((void **)&irms) ; 
	freeMemory((void **)&irm)  ; 
}
