/** \file IntensCalc_CUDA_kernel.cuh
 * \author Tomasz Jakubczyk
 * \brief plik nagłówkowy dla funkcji CUDA na GPU
 *
 *
 *
 */

extern "C"
{

__global__
void aviGetValueD(char* buff, unsigned short* frame, unsigned int frame_size);

__global__
void demosaicD(unsigned short* frame, unsigned int frame_size, short* outArray);

__global__
void correctionD(short* color, int* mask, int mask_size, float* IC, float* I);

}
