#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int *nmax;
	double *u;
	double *result;
	u=mxGetPr(prhs[0]);
	nmax=(int*)mxGetPr(prhs[1]);
	if(mxIsDouble(prhs[0])!=1) {
		mexErrMsgTxt("Error: U is not a double");
	}
	if(mxIsInt32(prhs[1])!=1) {
		mexErrMsgTxt("Error: nmax is not an integer");
	}
	plhs[0]=mxCreateDoubleMatrix(2,*nmax,mxREAL);
	result=mxGetPr(plhs[0]);
	int i;
	result[0]=1;
	result[1]=*u;
	result[2]=3*(*u);
	result[3]=3*cos(2*acos(*u));
	/* Evaluation loop */
	
	for(i=2;i<*nmax;i++) {
		result[2*i]=(double)(2*i+1)/i*result[2*i-2]*(*u)-(double)(i+1)/i*result[2*i-4];
		result[2*i+1]=(i+1)*(*u)*result[2*i]-(i+2)*result[2*i-2];
	}
	
	
	
}