#ifndef __auxillary
#define __auxillary
#include"globals.h"
#include<omp.h>
#include"stdio.h"
#include"ccuda.h"
#include"stdlib.h"

#define TIME(x) {double init = omp_get_wtime(); x; printf("Czas " #x " %f\n", omp_get_wtime()-init);}

inline void startTimer(double& init) {
#ifdef TIMING
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
}
inline void stopTimer(const char nazwa[], double init){
#ifdef TIMING 
#ifdef _OPENMP 
	printf("%f %s\n", omp_get_wtime()-init, nazwa);
#else
	printf("%f %s\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC, nazwa);
#endif 
#endif 
}
void __readVariable(  void * pointer, size_t size, size_t count, FILE * file, const  char * file2, const int line, const char * name );
void __writeVariable( void * pointer, size_t size, size_t count, FILE * file, const  char * file2, const int line, const char * name );

inline void allocMemory(void ** pointer, size_t size) {
#ifdef CUDA
	allocCudaPointer( pointer, size);
#else //ifdef CUDA
	*pointer = malloc(size);
#endif //CUDA	
}

inline void freeMemory(void ** pointer) {
#ifdef CUDA
	freeCudaPointer( pointer );
#else //ifdef CUDA
	free( *pointer);
#endif //CUDA
	*pointer=0;
}


inline float sq(float a) {
	return a*a;
}
inline int compare (const void * a, const void * b)
{
	return ( *(float*)a - *(float*)b );
}
//float sq(float);
void printMatrix(float *, int, int);
void printVector(float *, int);

void printRowVector( float *a, int n);
void printDouble(float a);
#endif
