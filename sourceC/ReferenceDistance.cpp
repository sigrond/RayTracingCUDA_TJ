#include"globals.h"
#include"auxillary.h"
void ReferenceDistance(float *   err,  float *   patterns, float *   references, int mPatterns, int nPatterns, int mReferences, int nReferences) {
#pragma omp parallel for
	for(int j=0;j<mReferences;++j) {
		float rSquared= sq(references[j*nReferences]);
		for(int i=1;i<nReferences;++i) 
			rSquared += sq(references[j*nReferences+i]);
		rSquared = 1.0/rSquared;
		for(int i=0;i<mPatterns;++i) {
			float product=patterns[i*nPatterns]*references[j*nReferences];
			for(int k=1;k<nPatterns;++k)
				product+=patterns[i*nPatterns+k]*references[j*nReferences+k];
			const float scale = product * rSquared;
			err[i*mReferences+j] = sq(patterns[i*nReferences]-scale*references[j*nReferences]);
			for(int k=1;k<nPatterns;++k) 
				err[i*mReferences+j]+= sq(patterns[i*nReferences+k]-scale*references[j*nReferences+k]);
		}
	}
}


