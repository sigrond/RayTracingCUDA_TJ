#ifndef __cudaGlobals
#define __cudaGlobals
extern float * devPii[2];
extern float * devTau[2];
extern float * devAfloat[2];
extern float * devAImag[2];
extern float * devBfloat[2];
extern float * devBImag[2];
extern float * devII[2];
extern int * devNmax[2];
extern cudaStream_t stream[2];
extern cudaStream_t streamRef[2];
extern float * devReferences[2];
extern float * devErr[2];
extern float * devPatterns[2];
extern float * devInvRSquare[2];
extern float * devPSquare[2];
extern float * devMin[2];
extern float * devMax[2];
extern int * devMinIndex[2];
extern float * devMedian[2];

//extern float * devIn[3];
extern int * devOut;
//extern cudaStream_t streamMin[3];
#endif
