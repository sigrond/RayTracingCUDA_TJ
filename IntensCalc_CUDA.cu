/** \file IntensCalc_CUDA.cu
 * \author Tomasz Jakubczyk
 * \brief plik z implementacjami funkcji wywołujących CUDA'ę
 *
 *
 *
 */

#include "mex.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "IntensCalc_CUDA_kernel.cuh"
#include "MovingAverage_CUDA_kernel.cuh"

#ifdef DEBUG
extern unsigned short* previewFa;/**< klatka po obliczeniu wartości pixeli */
unsigned short* previewFa=nullptr;

extern short* previewFb;/**< czerwona klatka po demosaicu */
short* previewFb=nullptr;

extern float* previewFc;/**< czerwona klatka po nałożeniu obrazu korekcyjnego */
float* previewFc=nullptr;

extern float* previewFd;/**< czerwona klatka po sumowaniu pixeli */
float* previewFd=nullptr;
#endif // DEBUG

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
unsigned char* dev_BgMask=NULL;
float* dev_BgValue=NULL;
float BgMask_Size=0;
float lastProbablyCorrectBgValue=60;

int licznik_klatek=0;
short previewFb2[640*480];

extern "C"
{

void setupCUDA_IC()
{
    /**< przygotowanie CUDA'y */

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
    checkCudaErrors(cudaMalloc((void**)&dev_BgValue, sizeof(float)));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_buff,0,sizeof(char)*640*480*2));
    checkCudaErrors(cudaMemset(dev_frame,0,sizeof(unsigned short)*640*480));
    checkCudaErrors(cudaMemset(dev_outArray,0,sizeof(short)*640*480*3));
    checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)));
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
    unsigned char* BgMask, float BgMaskSize)
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
    checkCudaErrors(cudaMalloc((void**)&dev_BgMask, sizeof(unsigned char)*640*480));
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
    checkCudaErrors(cudaMemcpy((void*)dev_BgMask, BgMask, sizeof(unsigned char)*640*480, cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }

    BgMask_Size=BgMaskSize;
}

void copyBuff(char* buff)
{
    /**< kopiujemy na kartę */
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

//extern unsigned short previewFa[640*480];


void doIC(float* I_Red, float* I_Green, float* I_Blue)
{
    uint numThreads, numBlocks;
    computeGridSize(640*480, 512, numBlocks, numThreads);
    unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
    unsigned int dimGridY=numBlocks/65535+1;
    dim3 dimGrid(dimGridX,dimGridY);

    /**< Jeśli tutaj będzie działało za wolno, to można wykozystać dodatkowy wątek CPU i CUDA streams */
    aviGetValueD<<< dimGrid, numThreads >>>(dev_buff,dev_frame,640*480);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(aviGetValueD): %s\n", cudaGetErrorString(err));
    }
    #ifdef DEBUG
    if(licznik_klatek==1)
    checkCudaErrors(cudaMemcpy((void*)previewFa,dev_frame,sizeof(unsigned short)*640*480,cudaMemcpyDeviceToHost));
    #endif // DEBUG

    /**< demosaic */
    demosaicD<<< dimGrid, numThreads >>>(dev_frame,640*480,dev_outArray);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(demosaicD): %s\n", cudaGetErrorString(err));
    }

    #ifdef DEBUG
    if(licznik_klatek==1)
    checkCudaErrors(cudaMemcpy((void*)previewFb,dev_outArray+640*480*2,sizeof(short)*640*480,cudaMemcpyDeviceToHost));
    #endif // DEBUG

    if(ipR_Size>0)
    {
        if(licznik_klatek++<20)/**< debug */
        {
            printf("frame: %d\n",licznik_klatek);
            checkCudaErrors(cudaMemcpy((void*)previewFb2,dev_outArray+640*480*2,sizeof(short)*640*480,cudaMemcpyDeviceToHost));
            for(int i=0;i<480;i++)//480
            {
                for(int j=0;j<640;j++)//640
                {
                    if(i%16==8 && j%16==8)
                    printf("%d ",previewFb2[i*640+j]>=1000?1:0);
                }
                if(i%16==8)
                printf("\n");
            }
            printf("\n");
        }
        /**< obliczyć wartość tła */
        computeGridSize(640*480, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
        dim3 dimGrid0(dimGridX,dimGridY);
        checkCudaErrors(cudaMemset(dev_BgValue,0,sizeof(float)));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        }
        getBgD<<< dimGrid0, numThreads >>>(dev_outArray+640*480*2,dev_BgMask,dev_BgValue);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(getBgD R): %s\n", cudaGetErrorString(err));
        }
        //dev_BgValue[0]=(float)dev_BgValue[0]/(float)dev_BgMaskSize[0];
        float tmpBgValue=0.0f;
        checkCudaErrors(cudaMemcpy((void*)&tmpBgValue,dev_BgValue,sizeof(float),cudaMemcpyDeviceToHost));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
        }
        tmpBgValue/=BgMask_Size;
        if(tmpBgValue>=200.0f)
        {
            //printf("tmpBgValue: %f, ",tmpBgValue);
            tmpBgValue=lastProbablyCorrectBgValue;
        }
        else
        {
            lastProbablyCorrectBgValue+=tmpBgValue;
            lastProbablyCorrectBgValue/=2.0f;
        }
        checkCudaErrors(cudaMemset(dev_BgValue,tmpBgValue,sizeof(float)));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
        }

        /**< nałożyć maskę i skorygować */
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
        checkCudaErrors(cudaMemcpy((void*)previewFc,dev_IR,sizeof(float)*ipR_Size,cudaMemcpyDeviceToHost));
        #endif // DEBUG
        /**< przydatna sztuczka do podglądania w matlabie:
        tmpIM=zeros(640,480,'single');
        tmpIM=reshape(tmpIM,640*480,[]);
        tmpIM(ipR)=prevRC;
        tmpIM=reshape(tmpIM,640,480);
        imtool(tmpIM')
         */

        /**< średnia krocząca */
        MovingAverageD<<< dimGrid, numThreads >>>(dev_IR,ipR_Size,dev_I_S_R,dev_sIR,64.0f);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(MovingAverageD): %s\n", cudaGetErrorString(err));
        }

        #ifdef DEBUG
        checkCudaErrors(cudaMemcpy((void*)previewFd,dev_sIR,sizeof(float)*ipR_Size,cudaMemcpyDeviceToHost));
        #endif // DEBUG

        DivD<<< dimGrid, numThreads >>>(ipR_Size,dev_sIR,64.0f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("cudaError(DivD): %s\n", cudaGetErrorString(err));
        }

        /**< wybór reprezentantów */
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
        computeGridSize(ipG_Size, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
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
        computeGridSize(ipB_Size, 512, numBlocks, numThreads);
        unsigned int dimGridX=numBlocks<65535?numBlocks:65535;
        unsigned int dimGridY=numBlocks/65535+1;
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
}

void freeCUDA_IC()
{
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
    checkCudaErrors(cudaFree(dev_BgMask));
    checkCudaErrors(cudaFree(dev_BgValue));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
    cudaProfilerStop();
}

}
