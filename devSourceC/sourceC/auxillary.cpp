#include<cstdio>
#include<ctime>
#include<cstdlib>
#include"globals.h"
//double sq(double a) {
//	return a*a;
//}
void printMatrix(real *a, int m, int n) {
	for(int i=0;i<m;i++) {
		for(int j=0;j<n;j++)
			printf("%e ", a[i*n+j]);
		printf("\n");
	}
}
void printVector(real *a, int n) {
	for(int i=0;i<n;++i)
		printf("%.10e ", a[i]);
	printf("\n");
}
void printRowVector( real *a, int n) {
	for(int i=0;i<n;++i)
		printf("%.10e\n", a[i]);
}
void printDouble(real a) {
	printf("%f\n", a);
}
