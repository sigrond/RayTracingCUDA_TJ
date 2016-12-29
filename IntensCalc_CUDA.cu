/** \file IntensCalc_CUDA.cu
 * \author Tomasz Jakubczyk
 * \brief file with functions implementations which call CUDA
 *
 *
 *
 */

#include "mex.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <exception>
#include <string>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "IntensCalc_CUDA_kernel.cuh"
#include "MovingAverage_CUDA_kernel.cuh"
#include "JunkStruct.h"

#ifdef DEBUG
extern unsigned short* previewFa;/**< frame after calculation pixels values */
unsigned short* previewFa=nullptr;

extern short* previewFb;/**< red frame after demosaic */
short* previewFb=nullptr;

extern float* previewFc;/**< red frame after applaying correction image */
float* previewFc=nullptr;

extern float* previewFd;/**< red frame after summing pixels */
float* previewFd=nullptr;

extern int frameToPrev;
#endif // DEBUG

extern int SubBg;

__host__
//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__host__
/** \brief compute grid and thread block size for a given number of elements
 *
 * \param n uint
 * \param blockSize uint
 * \param numBlocks uint&
 * \param numThreads uint&
 * \return void
 *
 */
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

cudaError_t err;
char* dev_buff=NULL;
unsigned short* dev_frame=NULL;
short* dev_outArray=NULL;

int* dev_ipR=NULL;
int ipR_Size=0;
int* dev_ipG=NULL;
int ipG_Size=0;
int* dev_ipB=NULL;
int ipB_Size=0;
float* dev_ICR_N=NULL;
float* dev_ICG_N=NULL;
float* dev_ICB_N=NULL;
int* dev_I_S_R=NULL;
int* dev_I_S_G=NULL;
int* dev_I_S_B=NULL;
float* dev_IR=NULL;
float* dev_IG=NULL;
float* dev_IB=NULL;
float* dev_sIR=NULL;
float* dev_sIG=NULL;
float* dev_sIB=NULL;
float* dev_RR=NULL;
float* dev_RG=NULL;
float* dev_RB=NULL;
unsigned char* dev_BgMaskR=NULL;
unsigned char* dev_BgMaskG=NULL;
unsigned char* dev_BgMaskB=NULL;
float* dev_BgValue=NULL;
float BgMask_SizeR[2]={0.0f, 0.0f};
float BgMask_SizeG[2]={0.0f, 0.0f};
float BgMask_SizeB[2]={0.0f, 0.0f};
float lastProbablyCorrectBgValue=60;
char* dev_DataSpace=NULL;
long int headerPosition=0;/**< header position in DataSpace from before which previously frame was copied */
#define MAX_JUNK_CHUNKS_PER_DATA_SPACE 128
//JunkStruct* dev_junkList=NULL;
long long int* dev_junkList=NULL;
//JunkStruct junkList[MAX_JUNK_CHUNKS_PER_DATA_SPACE];
long long int junkList[MAX_JUNK_CHUNKS_PER_DATA_SPACE];
long int* dev_junkCounter=NULL;
long int junkCounter=0;
#define MAX_HEADERS_PER_DATA_SPACE 8
long int* dev_headerList=NULL;
long int headerList[MAX_HEADERS_PER_DATA_SPACE];
long int* dev_headerCounter=NULL;
long int headerCounter=0;

int licznik_klatek=0;
short previewFb2[640*480];

extern "C"
{

void setupCUDA_IC()
{
    /**< prepare CUDA */

    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaFree(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree(0)): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_buff, sizeof(char)*640*480*2));
    checkCudaErrors(cudaMalloc((void**)&dev_frame, sizeof(unsigned short)*640*480));
    checkCudaErrors(cudaMalloc((void**)&dev_outArray, sizeof(short)*640*480*3));
    checkCudaErrors(cudaMalloc((void**)&dev_BgValue, sizeof(float)*2));
    checkCudaErrors(cudaMalloc((void**)&dev_DataSpace, sizeof(char)*65535*10*2));
    //checkCudaErrors(cudaMalloc((void**)&dev_junkList, sizeof(JunkStruct)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMalloc((void**)&dev_junkList, sizeof(long long int)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMalloc((void**)&dev_junkCounter, sizeof(long int)));
    checkCudaErrors(cudaMalloc((void**)&dev_headerList, sizeof(long int)*MAX_HEADERS_PER_DATA_SPACE));
    checkCudaErrors(cudaMalloc((void**)&dev_headerCounter, sizeof(long int)));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_buff,0,sizeof(char)*640*480*2));
    checkCudaErrors(cudaMemset(dev_frame,0,sizeof(unsigned short)*640*480));
    checkCudaErrors(cudaMemset(dev_outArray,0,sizeof(short)*640*480*3));
    checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)*2));
    //checkCudaErrors(cudaMemset(dev_junkList,0,sizeof(JunkStruct)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_junkList,0,sizeof(long long int)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_junkCounter,0,sizeof(long int)));
    checkCudaErrors(cudaMemset(dev_headerList,0,sizeof(long int)*MAX_HEADERS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_headerCounter,0,sizeof(long int)));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
}

void setMasksAndImagesAndSortedIndexes(
    int* ipR,int ipR_size,int* ipG,int ipG_size,int* ipB, int ipB_size,
    float* ICR_N, float* ICG_N, float* ICB_N,
    int* I_S_R, int* I_S_G, int* I_S_B,
    unsigned char* BgMaskR, float* BgMaskSizeR,
    unsigned char* BgMaskG, float* BgMaskSizeG,
    unsigned char* BgMaskB, float* BgMaskSizeB)
{
    ipR_Size=ipR_size;
    ipG_Size=ipG_size;
    ipB_Size=ipB_size;

    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_ipR, sizeof(int)*ipR_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_ipG, sizeof(int)*ipG_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_ipB, sizeof(int)*ipB_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_ICR_N, sizeof(float)*ipR_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_ICG_N, sizeof(float)*ipG_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_ICB_N, sizeof(float)*ipB_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_I_S_R, sizeof(int)*ipR_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_I_S_G, sizeof(int)*ipG_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_I_S_B, sizeof(int)*ipB_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_IR, sizeof(float)*ipR_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_IG, sizeof(float)*ipG_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_IB, sizeof(float)*ipB_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_sIR, sizeof(float)*ipR_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_sIG, sizeof(float)*ipG_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_sIB, sizeof(float)*ipB_size));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_RR, sizeof(float)*700));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_RG, sizeof(float)*700));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_RB, sizeof(float)*700));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    /** \todo pobrać zmienne z wymiarami maski tła
     */
    checkCudaErrors(cudaMalloc((void**)&dev_BgMaskR, sizeof(unsigned char)*640*480));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_BgMaskG, sizeof(unsigned char)*640*480));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMalloc((void**)&dev_BgMaskB, sizeof(unsigned char)*640*480));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMemcpy((void*)dev_ipR, ipR, sizeof(int)*ipR_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_ipG, ipG, sizeof(int)*ipG_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_ipB, ipB, sizeof(int)*ipB_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_ICR_N, ICR_N, sizeof(float)*ipR_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_ICG_N, ICG_N, sizeof(float)*ipG_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    //return;
    checkCudaErrors(cudaMemcpy((void*)dev_ICB_N, ICB_N, sizeof(float)*ipB_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    //return;
    checkCudaErrors(cudaMemcpy((void*)dev_I_S_R, I_S_R, sizeof(int)*ipR_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_I_S_G, I_S_G, sizeof(int)*ipG_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    checkCudaErrors(cudaMemcpy((void*)dev_I_S_B, I_S_B, sizeof(int)*ipB_size, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }

    checkCudaErrors(cudaMemcpy((void*)dev_BgMaskR, BgMaskR, sizeof(unsigned char)*640*480, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    BgMask_SizeR[0]=BgMaskSizeR[0];
    BgMask_SizeR[1]=BgMaskSizeR[1];

    checkCudaErrors(cudaMemcpy((void*)dev_BgMaskG, BgMaskG, sizeof(unsigned char)*640*480, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    BgMask_SizeG[0]=BgMaskSizeG[0];
    BgMask_SizeG[1]=BgMaskSizeG[1];

    checkCudaErrors(cudaMemcpy((void*)dev_BgMaskB, BgMaskB, sizeof(unsigned char)*640*480, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
    BgMask_SizeB[0]=BgMaskSizeB[0];
    BgMask_SizeB[1]=BgMaskSizeB[1];
}

/** \brief copy frame from given buffer to GPU memmory
 *
 * \param buff char* buffer with frame
 * \return void
 *
 */
void copyBuff(char* buff)
{
    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemcpy((void*)dev_buff, buff, sizeof(char)*640*480*2, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
    }
}

void loadLeft(char* buff)
{
    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemcpy((void*)dev_DataSpace, buff, sizeof(char)*655350, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
    }
}

void loadRight(char* buff)
{
    checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemcpy((void*)(dev_DataSpace+655350), buff, sizeof(char)*655350, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
    }
}

void cycleDataSpace(char* buff)
{
    if(headerPosition>=655350)
    {
        checkCudaErrors(cudaSetDevice(0));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
        }
        checkCudaErrors(cudaMemcpy((void*)(dev_DataSpace+655350), dev_DataSpace, sizeof(char)*655350, cudaMemcpyDeviceToDevice));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        }
        loadLeft(buff);
        headerPosition-=655350;
        if(headerPosition<0)
        {
            printf("Warning: headerPosition<0\n");
        }
    }
}

inline bool compJunk(const JunkStruct& a, const JunkStruct& b)
{
    return a.position<b.position;
}

void findJunkAndHeaders()
{
try
{
    //set to zero junk and headers lists
    //checkCudaErrors(cudaMemset(dev_junkList,0,sizeof(JunkStruct)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_junkList,0,sizeof(long long int)*MAX_JUNK_CHUNKS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_junkCounter,0,sizeof(long int)));
    checkCudaErrors(cudaMemset(dev_headerList,0,sizeof(long int)*MAX_HEADERS_PER_DATA_SPACE));
    checkCudaErrors(cudaMemset(dev_headerCounter,0,sizeof(long int)));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        throw std::string("cudaError(cudaMemset)");
    }

    //kernel searching for JUNK and headers 
    uint numThreads, numBlocks;
    computeGridSize(655350*2, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);
    findJunkAndHeadersD<<< dimGrid, numThreads >>>(dev_DataSpace,dev_junkList,dev_junkCounter,dev_headerList,dev_headerCounter);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(findJunkAndHeadersD): %s\n", cudaGetErrorString(err));
        throw std::string("cudaError(findJunkAndHeadersD)");
    }

    //copy junk and headers lists to host memmory
    int* tmpDst=(int*)junkList;
    int* tmpSrc=(int*)dev_junkList;
    //checkCudaErrors(cudaMemcpy((void*)tmpDst,tmpSrc,sizeof(JunkStruct)*MAX_JUNK_CHUNKS_PER_DATA_SPACE,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)tmpDst,tmpSrc,sizeof(JunkStruct)*MAX_JUNK_CHUNKS_PER_DATA_SPACE,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        throw std::string("cudaError(cudaMemcpyDeviceToHost)");
    }
    checkCudaErrors(cudaMemcpy((void*)&junkCounter,dev_junkCounter,sizeof(long int),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)headerList,dev_headerList,sizeof(long int)*MAX_HEADERS_PER_DATA_SPACE,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)&headerCounter,dev_headerCounter,sizeof(long int),cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        throw std::string("cudaError(cudaMemcpyDeviceToHost)");
    }
    #ifdef DEBUG_CUDA_DECODEC1
    printf("junkCounter: %d\n",junkCounter);
    printf("headerCounter: %d\n",headerCounter);
    #endif // DEBUG_CUDA_DECODEC1

	//select fragments of DataSpace to copy to correct frame memory segment based on junk and headers lists and position of last copied frame
    //searching for next header
    long int currentHeader=0;/**< position of header after frame we look for */
    for(int i=0;i<headerCounter;i++)
    {
        if(headerList[i]>0 && headerList[i]>headerPosition && headerList[i]>640*480*2)
        {
            if(currentHeader==0 || currentHeader>headerList[i])
            {
                currentHeader=headerList[i];
            }
        }
    }
    #ifdef DEBUG_CUDA_DECODEC1
    printf("currentHeader: %d\n",currentHeader);
    #endif // DEBUG_CUDA_DECODEC1
    //create vector with junk list and sort
    std::vector<JunkStruct> junkVector;/**< vector with positions and sizes of JUNK sections */
    for(int i=0;i<junkCounter;i++)
    {
        JunkStruct* tmpJunkStruct=(JunkStruct*)junkList+i;
        junkVector.push_back(*tmpJunkStruct);
    }
    std::sort(junkVector.begin(),junkVector.end(),compJunk);
    //select fragments and copy
    long int copiedBytes=0;/**< already copied bytes */
    long int dstOffset=0;/**< copy to */
    long int srcOffset=0;/**< copy from */
    long int bytesToCopy=0;/**< bytes to copy */
    long int lastSkipedPossition=currentHeader;/**< right stop of copying fragment */
    for(int i=junkVector.size()-1;i>=0;i--)
    {
        if(junkVector.at(i).position<currentHeader && junkVector.at(i).position>headerPosition && copiedBytes<(640*480*2))
        {
            bytesToCopy=lastSkipedPossition-(junkVector.at(i).position+junkVector.at(i).size+8);
            if(bytesToCopy>(640*480*2-copiedBytes))
            {
                bytesToCopy=640*480*2-copiedBytes;
            }
            if(bytesToCopy<0)
            {
                break;
            }
            srcOffset=junkVector.at(i).position+junkVector.at(i).size+8;
            if(srcOffset+bytesToCopy>655350*2)
            {
                printf("srcOffset+bytesToCopy>655350*2\n");
                throw std::string("srcOffset+bytesToCopy>655350*2");
            }
            if(srcOffset<0)
            {
                printf("srcOffset<0\n");
                throw std::string("srcOffset<0");
            }
            dstOffset=640*480*2-copiedBytes-bytesToCopy;
            if(dstOffset+bytesToCopy>640*480*2)
            {
                printf("dstOffset+bytesToCopy>640*480*2\n");
                throw std::string("dstOffset+bytesToCopy>640*480*2");
            }
            if(dstOffset<0)
            {
                printf("dstOffset<0\n");
                throw std::string("dstOffset<0");
            }
            #ifdef DEBUG_CUDA_DECODEC1
            printf("dstOffset: %d\n",dstOffset);
            printf("srcOffset: %d\n",srcOffset);
            printf("bytesToCopy: %d\n",bytesToCopy);
            printf("headerPosition: %d\n",headerPosition);
            printf("currentHeader: %d\n",currentHeader);
            #endif // DEBUG_CUDA_DECODEC1
            checkCudaErrors(cudaMemcpy((void*)(dev_buff+dstOffset), dev_DataSpace+srcOffset , sizeof(char)*bytesToCopy , cudaMemcpyDeviceToDevice));
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
                throw std::string("cudaError(Memcpy)");
            }
            lastSkipedPossition=junkVector.at(i).position;
            copiedBytes+=bytesToCopy;
        }
    }
    if(copiedBytes<640*480*2)
    {
        bytesToCopy=640*480*2-copiedBytes;
        if(lastSkipedPossition-bytesToCopy<0)
        {
            bytesToCopy=lastSkipedPossition;
        }
        if(bytesToCopy<=0)
        {
            printf("bytesToCopy<=0\n");
            headerPosition=currentHeader;
            return;
        }
        srcOffset=lastSkipedPossition-bytesToCopy;
        if(srcOffset+bytesToCopy>655350*2)
        {
            printf("srcOffset+bytesToCopy>655350*2\n");
            headerPosition=currentHeader;
            return;
        }
        if(srcOffset<0)
        {
            srcOffset=0;
        }
        dstOffset=640*480*2-copiedBytes-bytesToCopy;
        if(dstOffset+bytesToCopy>640*480*2)
        {
            printf("dstOffset+bytesToCopy>640*480*2\n");
            headerPosition=currentHeader;
            return;
        }
        if(dstOffset<0)
        {
            printf("dstOffset<0\n");
            headerPosition=currentHeader;
            return;
        }
        #ifdef DEBUG_CUDA_DECODEC1
        printf("dstOffset: %d\n",dstOffset);
        printf("srcOffset: %d\n",srcOffset);
        printf("bytesToCopy: %d\n",bytesToCopy);
        printf("headerPosition: %d\n",headerPosition);
        printf("currentHeader: %d\n",currentHeader);
        #endif // DEBUG_CUDA_DECODEC1

        checkCudaErrors(cudaMemcpy((void*)(dev_buff+dstOffset), dev_DataSpace+srcOffset , sizeof(char)*bytesToCopy , cudaMemcpyDeviceToDevice));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
            throw std::string("cudaError(Memcpy)");
        }
    }
    headerPosition=currentHeader;
}
catch(const std::exception& e)
{
    printf("findJunkAndHeaders std::exception: %s\n",((std::string)e.what()).c_str());
    //throw e;
}
catch(const std::string& s)
{
    printf("findJunkAndHeaders std::string exception: %s\n",s.c_str());
    //throw s;
}
catch(...)
{
    printf("findJunkAndHeaders exception\n");
    //throw;
}
}

//extern unsigned short previewFa[640*480];

float avgBgValueR[2]={0.0f,0.0f};
float avgBgValueG[2]={0.0f,0.0f};
float avgBgValueB[2]={0.0f,0.0f};


void doIC(float* I_Red, float* I_Green, float* I_Blue)
{
    uint numThreads, numBlocks;
    computeGridSize(640*480, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);

    /**< possible speedup by use of streams */
    aviGetValueD<<< dimGrid, numThreads >>>(dev_buff,dev_frame,640*480);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(aviGetValueD): %s\n", cudaGetErrorString(err));
    }
    #ifdef DEBUG
    if(licznik_klatek>=frameToPrev && licznik_klatek<(frameToPrev+100))
    checkCudaErrors(cudaMemcpy((void*)(previewFa+640*480*(licznik_klatek-frameToPrev)),dev_frame,sizeof(unsigned short)*640*480,cudaMemcpyDeviceToHost));
    #endif // DEBUG

    /**< demosaic */
    demosaicD<<< dimGrid, numThreads >>>(dev_frame,640*480,dev_outArray);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(demosaicD): %s\n", cudaGetErrorString(err));
    }

    #ifdef DEBUG
    if(licznik_klatek>=frameToPrev && licznik_klatek<(frameToPrev+100))
    checkCudaErrors(cudaMemcpy((void*)(previewFb+640*480*(licznik_klatek-frameToPrev)),dev_outArray+640*480*2,sizeof(short)*640*480,cudaMemcpyDeviceToHost));
    #endif // DEBUG


    if(ipR_Size>0)
    {
        #ifdef DEBUG2
        if(licznik_klatek<20)/**< debug */
        {
            printf("frame: %d\n",licznik_klatek);
            checkCudaErrors(cudaMemcpy((void*)previewFb2,dev_outArray+640*480*2,sizeof(short)*640*480,cudaMemcpyDeviceToHost));
            for(int i=0;i<480;i++)//480
            {
                for(int j=0;j<640;j++)//640
                {
                    if(i%16==8 && j%16==8)
                    printf("%2d",previewFb2[i*640+j]/100);
                }
                if(i%16==8)
                printf("\n");
            }
            printf("\n");
        }
        #endif // DEBUG
        /**< calculate background value */
        computeGridSize(640*480, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
        dim3 dimGrid0(dimGridX,dimGridY);
        checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)*2));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        }
        getBgD<<< dimGrid0, numThreads >>>(dev_outArray+640*480*2,dev_BgMaskR,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(getBgD R): %s\n", cudaGetErrorString(err));
        }
        //dev_BgValue[0]=(float)dev_BgValue[0]/(float)dev_BgMaskSize[0];
        float tmpBgValue[2]={0.0f,0.0f};
        checkCudaErrors(cudaMemcpy((void*)tmpBgValue,dev_BgValue,sizeof(float)*2,cudaMemcpyDeviceToHost));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        }
        tmpBgValue[0]/=BgMask_SizeR[0];
        tmpBgValue[1]/=BgMask_SizeR[1];
        avgBgValueR[0]+=tmpBgValue[0];
        avgBgValueR[1]+=tmpBgValue[1];
		
        if(SubBg==0)
        {
            tmpBgValue[0]=0.0f;
            tmpBgValue[1]=0.0f;
        }
        checkCudaErrors(cudaMemcpy((void*)dev_BgValue, tmpBgValue, sizeof(float)*2, cudaMemcpyHostToDevice));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpy): %s\n", cudaGetErrorString(err));
        }

        /**< apply mask and correction image */
        computeGridSize(ipR_Size, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid(dimGridX,dimGridY);
        correctionD<<< dimGrid, numThreads >>>(dev_outArray+640*480*2,dev_ipR,ipR_Size,dev_ICR_N,dev_IR,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(correctionD R): %s\n", cudaGetErrorString(err));
        }

        #ifdef DEBUG
        if(licznik_klatek==frameToPrev)
        checkCudaErrors(cudaMemcpy((void*)previewFc,dev_IR,sizeof(float)*ipR_Size,cudaMemcpyDeviceToHost));
        #endif // DEBUG


        /**< moving average kernel call */
        MovingAverageD<<< dimGrid, numThreads >>>(dev_IR,ipR_Size,dev_I_S_R,dev_sIR,64.0f);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(MovingAverageD): %s\n", cudaGetErrorString(err));
        }

        #ifdef DEBUG
        if(licznik_klatek==frameToPrev)
        checkCudaErrors(cudaMemcpy((void*)previewFd,dev_sIR,sizeof(float)*ipR_Size,cudaMemcpyDeviceToHost));
        #endif // DEBUG

        DivD<<< dimGrid, numThreads >>>(ipR_Size,dev_sIR,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(DivD): %s\n", cudaGetErrorString(err));
        }

        /**< select representatives */
        computeGridSize(700, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid2(dimGridX,dimGridY);

        chooseRepresentativesD<<< dimGrid2, numThreads >>>(dev_sIR,ipR_Size,dev_RR,700);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(chooseRepresentativesD): %s\n", cudaGetErrorString(err));
        }
    }
    if(ipG_Size>0)
    {
        /**< calculate background value */
        computeGridSize(640*480, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
        dim3 dimGrid0(dimGridX,dimGridY);
        checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)*2));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        }
        getBgD<<< dimGrid0, numThreads >>>(dev_outArray+640*480,dev_BgMaskG,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(getBgD G): %s\n", cudaGetErrorString(err));
        }
        //dev_BgValue[0]=(float)dev_BgValue[0]/(float)dev_BgMaskSize[0];
        float tmpBgValue[2]={0.0f,0.0f};
        checkCudaErrors(cudaMemcpy((void*)tmpBgValue,dev_BgValue,sizeof(float)*2,cudaMemcpyDeviceToHost));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        }
        tmpBgValue[0]/=BgMask_SizeG[0];
        tmpBgValue[1]/=BgMask_SizeG[1];
        avgBgValueG[0]+=tmpBgValue[0];
        avgBgValueG[1]+=tmpBgValue[1];
        /*if(licznik_klatek<50)
        {
            printf("(G)tmpBgValue[0]: %f, ",tmpBgValue[0]);
            printf("(G)tmpBgValue[1]: %f\n",tmpBgValue[1]);
        }*/
        if(SubBg==0)
        {
            tmpBgValue[0]=0.0f;
            tmpBgValue[1]=0.0f;
        }
        checkCudaErrors(cudaMemcpy((void*)dev_BgValue, tmpBgValue, sizeof(float)*2, cudaMemcpyHostToDevice));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpy): %s\n", cudaGetErrorString(err));
        }

        /**< apply mask and correction image */
        computeGridSize(ipG_Size, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid(dimGridX,dimGridY);
        correctionD<<< dimGrid, numThreads >>>(dev_outArray+640*480,dev_ipG,ipG_Size,dev_ICG_N,dev_IG,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(correctionD G): %s\n", cudaGetErrorString(err));
        }

        MovingAverageD<<< dimGrid, numThreads >>>(dev_IG,ipG_Size,dev_I_S_G,dev_sIG,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(MovingAverageD): %s\n", cudaGetErrorString(err));
        }

        DivD<<< dimGrid, numThreads >>>(ipG_Size,dev_sIG,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(DivD): %s\n", cudaGetErrorString(err));
        }

        computeGridSize(700, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid2(dimGridX,dimGridY);

        chooseRepresentativesD<<< dimGrid2, numThreads >>>(dev_sIG,ipG_Size,dev_RG,700);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(chooseRepresentativesD): %s\n", cudaGetErrorString(err));
        }
    }
    if(ipB_Size>0)
    {
        /**< calculate background value */
        computeGridSize(640*480, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
        dim3 dimGrid0(dimGridX,dimGridY);
        checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)*2));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        }
        getBgD<<< dimGrid0, numThreads >>>(dev_outArray,dev_BgMaskB,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(getBgD B): %s\n", cudaGetErrorString(err));
        }
        //dev_BgValue[0]=(float)dev_BgValue[0]/(float)dev_BgMaskSize[0];
        float tmpBgValue[2]={0.0f,0.0f};
        checkCudaErrors(cudaMemcpy((void*)tmpBgValue,dev_BgValue,sizeof(float)*2,cudaMemcpyDeviceToHost));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        }
        tmpBgValue[0]/=BgMask_SizeB[0];
        tmpBgValue[1]/=BgMask_SizeB[1];
        avgBgValueB[0]+=tmpBgValue[0];
        avgBgValueB[1]+=tmpBgValue[1];
        /*if(licznik_klatek<50)
        {
            printf("(B)tmpBgValue[0]: %f, ",tmpBgValue[0]);
            printf("(B)tmpBgValue[1]: %f\n",tmpBgValue[1]);
        }*/
        if(SubBg==0)
        {
            tmpBgValue[0]=0.0f;
            tmpBgValue[1]=0.0f;
        }
        checkCudaErrors(cudaMemcpy((void*)dev_BgValue, tmpBgValue, sizeof(float)*2, cudaMemcpyHostToDevice));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpy): %s\n", cudaGetErrorString(err));
        }

        /**< apply mask and correction image */
        computeGridSize(ipB_Size, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid(dimGridX,dimGridY);
        correctionD<<< dimGrid, numThreads >>>(dev_outArray,dev_ipB,ipB_Size,dev_ICB_N,dev_IB,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(correctionD B): %s\n", cudaGetErrorString(err));
        }

        MovingAverageD<<< dimGrid, numThreads >>>(dev_IB,ipB_Size,dev_I_S_B,dev_sIB,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(MovingAverageD): %s\n", cudaGetErrorString(err));
        }

        DivD<<< dimGrid, numThreads >>>(ipB_Size,dev_sIB,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(DivD): %s\n", cudaGetErrorString(err));
        }

        computeGridSize(700, 512, numBlocks, numThreads);
        dimGridX=numBlocks<65535?numBlocks:65535;
        dimGridY=numBlocks/65535+1;
        dim3 dimGrid2(dimGridX,dimGridY);

        chooseRepresentativesD<<< dimGrid2, numThreads >>>(dev_sIB,ipB_Size,dev_RB,700);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(chooseRepresentativesD): %s\n", cudaGetErrorString(err));
        }
    }

    checkCudaErrors(cudaMemcpy((void*)I_Red,dev_RR,sizeof(float)*700,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)I_Green,dev_RG,sizeof(float)*700,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)I_Blue,dev_RB,sizeof(float)*700,cudaMemcpyDeviceToHost));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
    licznik_klatek++;
}

void freeCUDA_IC()
{
    printf("avgBgValueR[0]: %f, ",(avgBgValueR[0]/(float)licznik_klatek));
    printf("avgBgValueR[1]: %f\n",(avgBgValueR[1]/(float)licznik_klatek));
    printf("avgBgValueG[0]: %f,",(avgBgValueG[0]/(float)licznik_klatek));
    printf("avgBgValueG[1]: %f\n",(avgBgValueG[1]/(float)licznik_klatek));
    printf("avgBgValueB[0]: %f, ",(avgBgValueB[0]/(float)licznik_klatek));
    printf("avgBgValueB[1]: %f\n",(avgBgValueB[1]/(float)licznik_klatek));

    checkCudaErrors(cudaFree(dev_buff));
    checkCudaErrors(cudaFree(dev_frame));
    checkCudaErrors(cudaFree(dev_outArray));

    checkCudaErrors(cudaFree(dev_ipR));
    checkCudaErrors(cudaFree(dev_ipG));
    checkCudaErrors(cudaFree(dev_ipB));
    checkCudaErrors(cudaFree(dev_ICR_N));
    checkCudaErrors(cudaFree(dev_ICG_N));
    checkCudaErrors(cudaFree(dev_ICB_N));
    checkCudaErrors(cudaFree(dev_I_S_R));
    checkCudaErrors(cudaFree(dev_I_S_G));
    checkCudaErrors(cudaFree(dev_I_S_B));
    checkCudaErrors(cudaFree(dev_IR));
    checkCudaErrors(cudaFree(dev_IG));
    checkCudaErrors(cudaFree(dev_IB));
    checkCudaErrors(cudaFree(dev_sIR));
    checkCudaErrors(cudaFree(dev_sIG));
    checkCudaErrors(cudaFree(dev_sIB));
    checkCudaErrors(cudaFree(dev_RR));
    checkCudaErrors(cudaFree(dev_RG));
    checkCudaErrors(cudaFree(dev_RB));
    checkCudaErrors(cudaFree(dev_BgMaskR));
    checkCudaErrors(cudaFree(dev_BgMaskG));
    checkCudaErrors(cudaFree(dev_BgMaskB));
    checkCudaErrors(cudaFree(dev_BgValue));
    checkCudaErrors(cudaFree(dev_DataSpace));
    checkCudaErrors(cudaFree(dev_junkList));
    checkCudaErrors(cudaFree(dev_junkCounter));
    checkCudaErrors(cudaFree(dev_headerList));
    checkCudaErrors(cudaFree(dev_headerCounter));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
    cudaProfilerStop();
}

}
