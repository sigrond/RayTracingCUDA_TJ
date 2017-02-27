#ifndef __auxillary
#define __auxillary
#include"globals.h"
#include"omp.h"
#include"stdio.h"

#define TIME(x) {double init = omp_get_wtime(); x; printf("Czas " #x " %f\n", omp_get_wtime()-init);}

inline void startTimer() {
#ifdef TIMING
	extern double init;
#ifdef _OPENMP
	init=omp_get_wtime();
#else
	init=clock();
#endif
#endif
}
inline void stopTimer(const char nazwa[]){
#ifdef TIMING 
	extern double init;
#ifdef _OPENMP 
	printf("%f %s\n", omp_get_wtime()-init, nazwa);
#else
	printf("%f %s\n", ((double)clock()-init)/(double)CLOCKS_PER_SEC, nazwa);
#endif 
#endif 
}



inline real sq(real a) {
	return a*a;
}
inline int compare (const void * a, const void * b)
{
	  return ( *(real*)a - *(real*)b );
}
//real sq(real);
void printMatrix(real *, int, int);
void printVector(real *, int);

void printRowVector( real *a, int n);
void printDouble(real a);
#endif
