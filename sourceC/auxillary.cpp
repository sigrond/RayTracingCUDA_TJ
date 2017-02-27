#include<cstdio>
#include<ctime>
#include<cstdlib>
#include"globals.h"
//double sq(double a) {
//	return a*a;
//}
void printMatrix(float *a, int m, int n) {
	for(int i=0;i<m;i++) {
		for(int j=0;j<n;j++)
			printf("%e ", a[i*n+j]);
		printf("\n");
	}
}
void printVector(float *a, int n) {
	for(int i=0;i<n;++i)
		printf("%.10e ", a[i]);
	printf("\n");
}
void printRowVector( float *a, int n) {
	for(int i=0;i<n;++i)
		printf("%.10e\n", a[i]);
}
void printDouble(float a) {
	printf("%f\n", a);
}
