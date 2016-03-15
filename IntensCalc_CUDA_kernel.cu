/** \file IntensCalc_CUDA_kernel.cu
 * \author Tomasz Jakubczyk
 * \brief funkcje CUDA na GPU
 *
 *
 *
 */


#define WIN32
#include<stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include "math_constants.h"

#include <stdio.h>

extern "C"
{


/** \brief wyliczanie wartości pixeli z bajtów filmu
 *
 * \param buff char*
 * \param frame unsigned short*
 * \param frame_size unsigned int
 * \return void
 *
 */
__global__
void aviGetValueD(char* buff, unsigned short* frame, unsigned int frame_size)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=frame_size)
        return;
    const unsigned char reverse6bitLookupTable[]={
0x00,0x20,0x10,0x30,0x08,0x28,0x18,0x38,0x04,0x24,0x14,0x34,0x0C,0x2C,0x1C,0x3C,
0x02,0x22,0x12,0x32,0x0A,0x2A,0x1A,0x3A,0x06,0x26,0x16,0x36,0x0E,0x2E,0x1E,0x3E,
0x01,0x21,0x11,0x31,0x09,0x29,0x19,0x39,0x05,0x25,0x15,0x35,0x0D,0x2D,0x1D,0x3D,
0x03,0x23,0x13,0x33,0x0B,0x2B,0x1B,0x3B,0x07,0x27,0x17,0x37,0x0F,0x2F,0x1F,0x3F};
/**< tablica odwracająca kolejność 6 młodszych bitów */

    unsigned short int bl,bh;
    bh=((unsigned short int)buff[2*index])<<6;
    bl=(unsigned short int)reverse6bitLookupTable[(unsigned char)(buff[2*index+1]>>2)];
    frame[index]=bh+bl;
}

}
