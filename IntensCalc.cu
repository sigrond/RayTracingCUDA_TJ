/** \file IntensCalc.cu
 * \author Tomasz Jakubczyk
 * \brief
 * kompilacja w matlabie:
 * nvmex -f nvmexopts64.bat IntensCalc.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"
 */

#define WIN32
#include "mex.h"
#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include "helper_math.h"
#include <fstream>
#include <thread>
#include <mutex>

using namespace std;

const unsigned char reverse6bitLookupTable[]={
0x00,0x20,0x10,0x30,0x08,0x28,0x18,0x38,0x04,0x24,0x14,0x34,0x0C,0x2C,0x1C,0x3C,
0x02,0x22,0x12,0x32,0x0A,0x2A,0x1A,0x3A,0x06,0x26,0x16,0x36,0x0E,0x2E,0x1E,0x3E,
0x01,0x21,0x11,0x31,0x09,0x29,0x19,0x39,0x05,0x25,0x15,0x35,0x0D,0x2D,0x1D,0x3D,
0x03,0x23,0x13,0x33,0x0B,0x2B,0x1B,0x3B,0x07,0x27,0x17,0x37,0x0F,0x2F,0x1F,0x3F};
/**< tablica odwracająca kolejność 6 młodszych bitów */

/** \brief wydaje się, że nvcc prawidłowo wyołuje wątki
 *
 * \return void
 *
 */
void test()
{
    printf("hello test!\n");
}

/** \brief
 * function [I_Red,I_Green,I_Blue] = IntensCalc(handles,count_step,NumFrames,ipR,ipG,ipB,ICR_N,ICG_N,ICB_N)
 * \param nlhs int
 * \param plhs[] mxArray*
 * \param nrhs int
 * \param prhs[] const mxArray*
 * \return void
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int count_step=1;/**< co która klatka */
    int NumFrames;/**< liczba klatek */
    int* ipR;/**< indeksy czerwonej maski */
    int ipR_size=0;
    int* ipG;/**< indeksy zielonej maski */
    int ipG_size=0;
    int* ipB;/**< indeksy niebieskiej maski */
    int ipB_size=0;
    char* name;/**< nazwa pliku z pe³n¹ œcierzk¹ */

    /**< sprawdzanie argumentów */
    if(nlhs!=3)
    {
    	printf("function returns [I_Red,I_Green,I_Blue] \n");
    	return;
    }
    if(nrhs!=12)
    {
        printf("function arguments are (handles,count_step,NumFrames,ipR,ipG,ipB,ICR_N,ICG_N,ICB_N,I_S_R,I_S_G,I_S_B) \n");
        return;
    }
    int tmpb=mxIsStruct(prhs[0]);
    printf("mxIsStruct:%d\n",tmpb);
    if(!tmpb)/**< handles */
    {
        printf("1st argument must be a struct\n");
        return;
    }
    if(!mxIsInt32(prhs[1]))/**< count_step */
    {
        printf("2nd argument needs to be Int32 number\n");
        return;
    }
    if(!mxIsInt32(prhs[2]))/**< NumFrames */
    {
        printf("3rd argument needs to be Int32 number\n");
        return;
    }
    if(!mxIsInt32(prhs[3]))/**< ipR */
    {
        printf("4th argument needs to be Int32 vector\n");
        return;
    }
    if(!mxIsInt32(prhs[4]))/**< ipB */
    {
        printf("5th argument needs to be Int32 vector\n");
        return;
    }
    if(!mxIsInt32(prhs[5]))/**< ipG */
    {
        printf("6th argument needs to be Int32 vector\n");
        return;
    }
    if(!mxIsSingle(prhs[6]))/**< ICR_N */
    {
        printf("7th argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[7]))/**< ICG_N */
    {
        printf("8th argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsSingle(prhs[8]))/**< ICB_N */
    {
        printf("9th argument needs to be single precision vector\n");
        return;
    }
    if(!mxIsInt32(prhs[9]))/**< I_S_R */
    {
        printf("10th argument needs to be Int32 vector\n");
        return;
    }
    if(!mxIsInt32(prhs[10]))/**< I_S_G */
    {
        printf("11th argument needs to be Int32 vector\n");
        return;
    }
    if(!mxIsInt32(prhs[11]))/**< I_S_B */
    {
        printf("12th argument needs to be Int32 vector\n");
        return;
    }

    /**< pobieranie danych */
    mxArray* tmp;
    tmp=mxGetField(prhs[0],0,"fn");/**< nazwa pliku */
    name=mxArrayToString(tmp);
    printf("name: %s\n",name);
    count_step=*((int*)mxGetPr(prhs[1]));
    if(mxGetN(prhs[1])*mxGetM(prhs[1])!=1)
    {
        printf("2nd argument (count_step) must be a number\n");
        return;
    }
    printf("count_step: %d\n",count_step);
    NumFrames=*((int*)mxGetPr(prhs[2]));
    if(mxGetN(prhs[2])*mxGetM(prhs[2])!=1)
    {
        printf("3rd argument (NumFrames) must be a number\n");
        return;
    }

    /**< przygotowanie zwracanych macierzy */
    int dimsI_Red[2]={NumFrames,700};
    plhs[0]=mxCreateNumericArray(2,dimsI_Red,mxSINGLE_CLASS,mxREAL);
    float* I_Red=(float*)mxGetPr(plhs[0]);
    int dimsI_Green[2]={NumFrames,700};
    plhs[1]=mxCreateNumericArray(2,dimsI_Green,mxSINGLE_CLASS,mxREAL);
    float* I_Green=(float*)mxGetPr(plhs[1]);
    int dimsI_Blue[2]={NumFrames,700};
    plhs[2]=mxCreateNumericArray(2,dimsI_Blue,mxSINGLE_CLASS,mxREAL);
    float* I_Blue=(float*)mxGetPr(plhs[2]);
    printf("I_Blue NxM: %dx%d\n",mxGetN(plhs[2]),mxGetM(plhs[2]));
    //return;


    /**< wczytaæ klatkê */
    char* buff1=new char[65535*10];/**< podwójne buforowanie z dysku */
    char* buff2=new char[65535*10];
    bool buffNo=true;/**< który bufor */
    /**< zakładam, że najwolniejszą operacją będzie czytanie z dysku */
    mutex buff1Mutex;/**< \todo zrobić wzorzec konsument producent */
    mutex buff2Mutex;
    bool frame1Ready=false;

    thread t1(test);
    t1.join();
    //return;
    /**< wątek z wyrażenia lmbda wykonuje się poprawnie :D */
    thread readMovieThread([&]
    {/**< uwaga wyra¿enie lambda w w¹tku */
        printf("readMovieThread\n");
        //return;
        ifstream file (name, ios::in|ios::binary);
        const int skok = (640*480*2)+8;
        char* buff=nullptr;/**< aktualny adres zapisu z dysku */
        for(int i=0;i<NumFrames;i+=count_step)/**< czytanie klatek */
        {
            file.seekg((34824+(skok*(i))),ios::beg);
/**< \todo można poprawić, żeby przesunięcie było względem obecnej pozycji i dało się czytać filmy >4GB */
            if(buffNo)
            {
                buff1Mutex.lock();
                buff=buff1;
            }
            else
            {
                buff2Mutex.lock();
                buff=buff2;
            }
            buffNo=!buffNo;
            for(int j=0;j<10;j++)
            {
                file.read(buff+j*65535,65535);/**< 64KB to optymalny rozmiar bloku czytanego z dysku */
            }
            if(buffNo)
                buff2Mutex.unlock();
            else
            {
                buff1Mutex.unlock();
                frame1Ready=true;
            }
        }
        file.close();

    });/**< readMovieThread lambda */

    /**< napisaæ szybsze odwracanie bajtu przy wyko¿ystaniu lookuptable */

    while(!frame1Ready);/**< czekamy aż wczyta się pierwsza klatka */
    /**< \todo można wykożystać condition variable, żeby było ładniej */
    bool bNo=true;
    char* tmpBuff=nullptr;
    unsigned short int klatka[307200];
    unsigned short int bl,bh;
    for(int k=0;k<NumFrames;k+=count_step)
    {
        if(bNo)
        {
            buff1Mutex.lock();
            tmpBuff=buff1;
        }
        else
        {
            buff2Mutex.lock();
            tmpBuff=buff2;
        }
        bNo=!bNo;
        //for(int l=0;l<307200;l++)/**< może bardziej opłacać się zrobić to na GPU */
        //{
        //    bh=((unsigned short int)tmpBuff[2*l])<<6;
        //    bl=(unsigned short int)reverse6bitLookupTable[(unsigned char)(tmpBuff[2*l+1]>>2)];
        //    klatka[l]=bh+bl;
        //}
        if(bNo)
            buff2Mutex.unlock();
        else
            buff1Mutex.unlock();
    }
    readMovieThread.join();
    /**< czytaæ wêksze bloki danych z pliku ni¿ po jednym znaku */
    /**< dla klatki zastosowaæ demosaic */
    /**< uzyskaæ wymaskowan¹ klatkê */
    /**< podzieliæ wymaskowan¹ klatkê przez macierz korekcyjn¹ */
    /**< u¿ywaæ strumieni CUDA i lub w¹tków, ¿eby jednoczeœnie czytaæ plik i liczyæ */
    /**< ka¿d¹ posortowan¹ klatkê wyg³adziæ œredni¹ krocz¹c¹ */
    /**< zwróciæ 700 równomiernie wybranych punktów dla ka¿dej klatki */
}
