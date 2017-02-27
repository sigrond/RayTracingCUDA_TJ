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
		printf("%.5e ", a[i]);
	printf("\n");
}
void printRowVector( float *a, int n) {
	for(int i=0;i<n;++i)
		printf("%.10e\n", a[i]);
}
void printDouble(float a) {
	printf("%f\n", a);
}

void __readVariable( void * pointer, size_t size, size_t count, FILE * file, const char * file2, const int line ,const char * name ) {
	size_t status;
	status = fread( pointer, size, count, file);
	if( status != count) {
		printf("Error reading a variable %s in file %s in line %d\n", name, file2, line);
		printf("Expected length: %zu but acctualy read: %zu\n", count, status);
		exit(0);
	}
}

void __writeVariable( void * pointer, size_t size, size_t count, FILE * file, const char * file2, const int line, const char * name ) {
	size_t status;
	status = fwrite( pointer, size, count, file);
	if( status != count) {
		printf("Error reading a variable %s in file %s in line %d\n", name,  file2, line);
		printf("Expected length: %zu but acctualy wrote: %zu\n", count, status);
		exit(0);
	}
}

