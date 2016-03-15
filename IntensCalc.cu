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
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

const unsigned char reverse6bitLookupTable[]={
0x00,0x20,0x10,0x30,0x08,0x28,0x18,0x38,0x04,0x24,0x14,0x34,0x0C,0x2C,0x1C,0x3C,
0x02,0x22,0x12,0x32,0x0A,0x2A,0x1A,0x3A,0x06,0x26,0x16,0x36,0x0E,0x2E,0x1E,0x3E,
0x01,0x21,0x11,0x31,0x09,0x29,0x19,0x39,0x05,0x25,0x15,0x35,0x0D,0x2D,0x1D,0x3D,
0x03,0x23,0x13,0x33,0x0B,0x2B,0x1B,0x3B,0x07,0x27,0x17,0x37,0x0F,0x2F,0x1F,0x3F};
/**< tablica odwracająca kolejność 6 młodszych bitów */

struct buffId
{
    buffId(int id,char* pt):id(id),pt(pt) {};
    int id;
    char* pt;
};


#define CBUFFS 8
/** \brief monitor dla bufora cyklicznego
 */
class CyclicBuffer
{
private:
    const int cBuffS;/**< rozmiar bufora cyklicznego */
    int cBeg,cEnd;/**< początek i koniec bufora cyklicznego */
    int itemCount;
    condition_variable full;/**< bufor cykliczny pełny */
    condition_variable empty;/**< bufor cykliczny pusty */
    bool buffReady[CBUFFS];/**< czy bufor nie jest już używany */
    condition_variable buffReadyCond[CBUFFS];
    char* cBuff[CBUFFS];/**< bufor cykliczny z buforami odczytu z dysku */
    condition_variable monitorCond;
    mutex monitorMtx;
public:
    /** \brief konstruktor
     */
    CyclicBuffer() :
        cBuffS(CBUFFS), cBeg(0), cEnd(0), itemCount(0)
    {
        for(int i=0;i<cBuffS;i++)/**< alokowanie pamięci bufora */
        {
            cBuff[i]=new char[65535*10];
            buffReady[i]=true;/**< na początku żaden bufor nie jest używany */
        }
    }
    /** \brief destruktor
     */
    ~CyclicBuffer()
    {
        for(int i=0;i<cBuffS;i++)
        {
            delete[] cBuff[i];
        }
    }
    /** \brief zajmij wskaźnik bufora do zapisu
     * \return char*
     */
    buffId* claimForWrite()
    {
        unique_lock<mutex> lck(monitorMtx);
		//printf("claimForWrite cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        while(itemCount==cBuffS)
        {
            printf("claimForWrite full cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
            full.wait(lck);/**< czekamy jeśli bufor cykliczny jest pełny */
        }
        unsigned int tmpEnd=cEnd;
        if(itemCount>0)
        {
            tmpEnd=(cEnd+1)%cBuffS;
        }
        if(tmpEnd==cBeg && itemCount!=0)
        {
            printf("błąd krytyczny, przepełnienie bufora");
            throw string("błąd krytyczny, przepełnienie bufora");
        }
        while(!buffReady[tmpEnd])
        {
            buffReadyCond[tmpEnd].wait(lck);/**< jeśli coś używa bufora to czekamy poza monitorem */
        }
        buffReady[tmpEnd]=false;/**< zaznaczamy, że bufor jest używany */
        lck.unlock();
        return new buffId(tmpEnd,cBuff[tmpEnd]);
    }
    /** \brief zwolnienie bufora po zapisaniu
     * \param id buffId*
     * \return void
     */
    void writeEnd(buffId* id)
    {
        monitorMtx.lock();
        //printf("writeEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        cEnd=id->id;
        itemCount++;
        buffReady[id->id]=true;/**< zaznaczamy, że nie używamy już bufora */
        monitorMtx.unlock();/**< odblokowujemy monitor */
        buffReadyCond[id->id].notify_one();/**< powiadamiamy, że bufor jest nie używany */
        delete id;/**< zwalniamy wskaźnik strukturę */
        empty.notify_one();/**< powiadamiamy, że bufor cykliczny nie jest już pusty */
    }
    /** \brief zajmij wskaźnik bufora do odczytu
     * \return buffId*
     */
    buffId* claimForRead()
    {
        unique_lock<mutex> lck(monitorMtx);
		//printf("claimForRead cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        while(itemCount==0)
        {
            //printf("claimForRead empty cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
            empty.wait(lck);
        }
        unsigned int tmpBeg=cBeg;
        while(!buffReady[tmpBeg])
        {
            buffReadyCond[tmpBeg].wait(lck);/**< jeśli coś używa bufora to czekamy poza monitorem */
        }
        buffReady[tmpBeg]=false;/**< zaznaczamy, że bufor jest używany */
        lck.unlock();
        return new buffId(tmpBeg,cBuff[tmpBeg]);
    }
    /** \brief zwolnienie bufora po odczytaniu
     * \param id buffId*
     * \return void
     */
    void readEnd(buffId* id)
    {
        monitorMtx.lock();
        //printf("readEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        cBeg=(id->id+1)%cBuffS;
        itemCount--;
        if(itemCount<0)
        {
            printf("błąd krytyczny, ujemna liczba elementów bufora");
            throw string("błąd krytyczny, ujemna liczba elementów bufora");
        }
        buffReady[id->id]=true;/**< zaznaczamy, że nie używamy już bufora */
        buffReadyCond[id->id].notify_one();/**< powiadamiamy, że bufor jest nie używany */
        delete id;
        monitorMtx.unlock();
        full.notify_one();
    }
};



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

try
{

    /**< wczytaæ klatkê */
    /**< wzorzec konsument producent */
    CyclicBuffer cyclicBuffer;

    /**< wątek z wyrażenia lmbda wykonuje się poprawnie :D */
    thread readMovieThread([&]
    {/**< uwaga wyra¿enie lambda w w¹tku */
        printf("readMovieThread\n");
        //return;
        ifstream file (name, ios::in|ios::binary);
        const int skok = (640*480*2)+8;
        char* buff=nullptr;/**< aktualny adres zapisu z dysku */
        buffId* bId=nullptr;
        for(int i=0;i<NumFrames;i+=count_step)/**< czytanie klatek */
        {
            file.seekg((34824+(skok*(i))),ios::beg);
/**< \todo można poprawić, żeby przesunięcie było względem obecnej pozycji i dało się czytać filmy >4GB */
            bId=cyclicBuffer.claimForWrite();
            buff=bId->pt;
            for(int j=0;j<10;j++)
            {
                file.read(buff+j*65535,65535);/**< 64KB to optymalny rozmiar bloku czytanego z dysku */
            }
            cyclicBuffer.writeEnd(bId);
        }
        file.close();

    });/**< readMovieThread lambda */

    /**< napisaæ szybsze odwracanie bajtu przy wyko¿ystaniu lookuptable */

    bool bNo=true;
    char* tmpBuff=nullptr;
    buffId* bID=nullptr;
    unsigned short int klatka[307200];
    unsigned short int bl,bh;
    long double licz=0.0f;
    for(int k=0;k<NumFrames;k+=count_step)
    {
        bID=cyclicBuffer.claimForRead();
        tmpBuff=bID->pt;
        for(int l=0;l<307200;l++)/**< może bardziej opłacać się zrobić to na GPU */
        {
            bh=((unsigned short int)tmpBuff[2*l])<<6;
            bl=(unsigned short int)reverse6bitLookupTable[(unsigned char)(tmpBuff[2*l+1]>>2)];
            klatka[l]=bh+bl;
            licz+=(long double)klatka[l];
        }
        cyclicBuffer.readEnd(bID);
        //printf("%lf",licz);
        /*if(k==1)
        {
            int jj=0;
            for(int ii=0;ii<307200;ii++)
            {
                printf("%4d ",klatka[ii]);
                if(jj++==640)
                {
                    jj=0;
                    printf("\n");
                }
            }
        }*/
    }
    readMovieThread.join();
}
catch(string& e)
{
    printf("wyjątek: %s",e.c_str());
}
catch(...)
{
    printf("nieznany wyjątek");
}
    /**< czytaæ wêksze bloki danych z pliku ni¿ po jednym znaku */
    /**< dla klatki zastosowaæ demosaic */
    /**< uzyskaæ wymaskowan¹ klatkê */
    /**< podzieliæ wymaskowan¹ klatkê przez macierz korekcyjn¹ */
    /**< u¿ywaæ strumieni CUDA i lub w¹tków, ¿eby jednoczeœnie czytaæ plik i liczyæ */
    /**< ka¿d¹ posortowan¹ klatkê wyg³adziæ œredni¹ krocz¹c¹ */
    /**< zwróciæ 700 równomiernie wybranych punktów dla ka¿dej klatki */
}
