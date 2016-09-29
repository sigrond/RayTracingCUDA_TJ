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
#include "JunkStruct.h"

#include <stdio.h>

extern "C"
{


/** \brief każdy watek sprawdza swój bajt danych i jeśli jest to początek jednego z kodów
 * JUNK albo header to zapisuje je synchronizowanej do listy
 * \param DataSpace char* wskaźnik na dane
 * \param junkList JunkStruct* wskaźnik na listę junk
 * \param junkCounter long int* wskaźnik na licznik znalezionych sekcji junk
 * \param headerList long int* wskaźnik na listę nagłówków
 * \param headerCounter long int* wskaźnik na licznik znalezionych nagłówków
 * \return void
 *
 */
__global__
void findJunkAndHeadersD(char* DataSpace,JunkStruct* junkList,long int* junkCounter,long int* headerList,long int* headerCounter)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=655350*2-8)
        return;
    const char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
    const char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
    const char junkCode[]="JUNK";
    bool junkB=true;
    for(int i=0;i<4;i++)
    {
        junkB&=junkCode[i]==DataSpace[index+i];
        if(!junkB)
        {
            break;
        }
    }
    if(junkB)
    {
        long int tmpJunkCounter=atomicAdd(junkCounter,1);
        junkList[tmpJunkCounter].position=index;
        junkList[tmpJunkCounter].size=*(long int*)(DataSpace+index+4);
        return;
    }
    bool headerB=true;
    for(int i=0;i<header.size;i++)
    {
        headerB&=frameStartCode[i]==DataSpace[index+i] || frameStartCodeS[i]==DataSpace[index+i];
        if(!headerB)
        {
            return;
        }
    }
    if(headerB)
    {
        long int tmpHeaderCounter=atomicAdd(headerCounter,1);
        headerList[tmpHeaderCounter]=index;
    }
}


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

    unsigned short bl,bh;
    bh=0x00FF&buff[2*index];/**< CUDA zdaje się sama nie zerować starszego bajtu ze śmieci */
    bh=bh<<6;
    bl=0x00FF&buff[2*index+1];
    bl=bl>>2;
    bl=0x00FF&(unsigned short)reverse6bitLookupTable[(unsigned char)bl];
    frame[index]=bh|bl;
}

#define I frame
/** \brief demosaic GRBG bruteforce!
 *
 * \param frame unsigned short*
 * \param frame_size unsigned int
 * \param outArray short*
 * \return void
 *
 */
__global__
void demosaicD(unsigned short* frame, unsigned int frame_size, short* outArray)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=frame_size)
        return;

    int j=index/640;
    int i=index%640;
    int wid=640, len=480;
    int x_max=wid-1, y_max=len-1;
    int im1=0,ip1=0,jm1=0,jp1=0;
    int lenxwid=len*wid;

    jm1=j==0?j+1:j-1;//j-1
    jp1=j==y_max?j-1:j+1;//j+1

    im1=i==0?i+1:i-1;//i-1
    ip1=i==x_max?i-1:i+1;//i+1
    if((i&1)==0)
    {
        if((j&1)==0)//R(G)R
        {
            outArray[i+j*wid]=(I[i+jm1*wid]+I[i+jp1*wid])>>1;//B
        }
        else//G(B)G
        {
            outArray[i+j*wid]=I[i+wid*j];//B
        }
    }
    else
    {
        if((j&1)==0)//G(R)G
        {
            outArray[i+j*wid]=(I[im1+wid*jm1]+I[ip1+wid*jp1]+I[im1+wid*jp1]+I[ip1+wid*jm1])>>2;//B
        }
        else//B(G)B
        {
            outArray[i+j*wid]=(I[im1+wid*j]+I[ip1+wid*j])>>1;//B
        }
    }

    jm1=j==0?j+1:j-1;//j-1
    jp1=j==y_max?j-1:j+1;//j+1

    im1=i==0?i+1:i-1;//i-1
    ip1=i==x_max?i-1:i+1;//i+1
    if((i&1)==0)
    {
        if((j&1)==0)//R(G)R
        {
            outArray[i+j*wid+lenxwid]=I[i+j*wid];//G
        }
        else//G(B)G
        {
            outArray[i+j*wid+lenxwid]=(I[im1+wid*j]+I[ip1+wid*j]+I[i+wid*jm1]+I[i+wid*jp1])>>2;//G
        }
    }
    else
    {
        if((j&1)==0)//G(R)G
        {
            outArray[i+j*wid+lenxwid]=(I[im1+wid*j]+I[ip1+wid*j]+I[i+wid*jm1]+I[i+wid*jp1])>>2;//G
        }
        else//B(G)B
        {
            outArray[i+j*wid+lenxwid]=I[i+wid*j];//G
        }
    }

    jm1=j==0?j+1:j-1;//j-1
    jp1=j==y_max?j-1:j+1;//j+1

    im1=i==0?i+1:i-1;//i-1
    ip1=i==x_max?i-1:i+1;//i+1
    if((i&1)==0)
    {
        if((j&1)==0)//R(G)R
        {
            outArray[i+j*wid+2*lenxwid]=(I[im1+j*wid]+I[ip1+j*wid])>>1;//R
        }
        else//G(B)G
        {
            outArray[i+j*wid+2*lenxwid]=(I[im1+jm1*wid]+I[ip1+wid*jp1]+I[im1+wid*jp1]+I[ip1+wid*jm1])>>2;//R
        }
    }
    else
    {
        if((j&1)==0)//G(R)G
        {
            outArray[i+j*wid+2*lenxwid]=I[i+wid*j];//R
        }
        else//B(G)B
        {
            outArray[i+j*wid+2*lenxwid]=(I[i+wid*jm1]+I[i+wid*jp1])>>1;//R
        }
    }
}

/** \brief oblicza wartość tła
 *
 * \param color short* klatka w wybranym kolorze
 * \param BgMask unsigned char* maska tła
 * \param BgValue float* zwracana wartość tła
 * \return void
 *
 */
__global__
void getBgD(short* color, unsigned char* BgMask, float* BgValue)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=640*480)
        return;
    int j=index/640;
    int i=index%640;
    int wid=640, len=480;
    if(BgMask[j+i*len]==0)
        return;
    float value=(float)color[i+j*wid];//j+i*len
    if(i<wid/2)
    {
        atomicAdd(BgValue,value);
    }
    else
    {
        float* tmpBg=&(BgValue[1]);
        atomicAdd(tmpBg,value);
    }
}

/** \brief nałożenie maski na kolor klatki i podzielenie przez obraz korekcyjny
 *
 * \param color short* klatka w wybranym kolorze
 * \param mask int* nakładana maska
 * \param mask_size int rozmiar maski
 * \param IC float* obraz korekcyjny
 * \param I float* zwracana skorygowana klatka w wybranym kolorze
 * \return void
 *
 */
__global__
void correctionD(short* color, int* mask, int mask_size, float* IC, float* I, float* BgValue)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=mask_size)
        return;
    int maskIndex=mask[index]-1;
    #ifndef ONE_BACKGROUND
    float tmpBgValue=(maskIndex%640<320)?BgValue[0]:BgValue[1];
    #else
    float tmpBgValue=(BgValue[0]+BgValue[1])*0.5f;
    #endif // ONE_BACKGROUND
    float tmpColor=(float)color[maskIndex]-tmpBgValue;
    if(tmpColor<=0)
    {
        I[index]=0;
    }
    else
    {
        I[index]=tmpColor/IC[index];
    }
}

/** \brief wybiera równomiernie rozłożone punkty
 *
 * \param I float* duży zbiór danych
 * \param I_size int rozmiar dużego zbioru
 * \param R float* zbiór wybranych danych
 * \param R_size int rozmiar wybranych danych
 * \return void
 *
 */
__global__
void chooseRepresentativesD(float* I, int I_size, float* R, int R_size)
{
    // unique block index inside a 3D block grid
    const unsigned int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    uint index = __mul24(blockId,blockDim.x) + threadIdx.x;
    if(index>=R_size)
        return;
    R[index]=I[index*(I_size-1)/R_size+1];
}

}
