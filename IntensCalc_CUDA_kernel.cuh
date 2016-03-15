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

}
