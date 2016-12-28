/** \file IntensCalc_CUDA_kernel.cuh
 * \author Tomasz Jakubczyk
 * \brief header file of CUDA functions for GPU
 *
 *
 *
 */

 #include "JunkStruct.h"

extern "C"
{

//__global__
//void findJunkAndHeadersD(char* DataSpace,JunkStruct* junkList,long int* junkCounter,long int* headerList,long int* headerCounter);
__global__
void findJunkAndHeadersD(char* DataSpace,long long int* junkList,long int* junkCounter,long int* headerList,long int* headerCounter);

__global__
void aviGetValueD(char* buff, unsigned short* frame, unsigned int frame_size);

__global__
void demosaicD(unsigned short* frame, unsigned int frame_size, short* outArray);

__global__
void getBgD(short* color, unsigned char* BgMask, float* BgValue);

__global__
void correctionD(short* color, int* mask, int mask_size, float* IC, float* I, float* BgValue);

__global__
void chooseRepresentativesD(float* I, int I_size, float* R, int R_size);

}
