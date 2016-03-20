/** \file avireadtest.cpp
 * \author Tomasz Jakubczyk
 * \brief plik do testowania poza matlabem
 * kompilacja:
 * nvcc avireadtest.cpp IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu CyclicBuffer.cpp MovingAverage_CUDA_kernel.cu -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30 -std=c++11 --use-local-env --cl-version 2012 -IC:\CUDA\include -IC:\CUDA\inc -lcufft -lcudart -lcuda
 *
C:\Our_soft\RayTracingCUDA_TJ>nvcc avireadtest.cpp IntensCalc_CUDA_kernel.cu Int
ensCalc_CUDA.cu CyclicBuffer.cpp -gencode=arch=compute_30,code=sm_30 -gencode=ar
ch=compute_30,code=compute_30 -std=c++11 --use-local-env --cl-version 2012 -IC:\
CUDA\include -IC:\CUDA\inc -lcufft -lcudart -lcuda -v -o avireadtest2.exe
 *
 */


#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <string>
#include <thread>
#include <exception>
#include <ctime>

#include "CyclicBuffer.hpp"
#include "IntensCalc_CUDA.cuh"

using namespace std;


int main(int argc,char* argv[])
{
    srand(time(0));
    string s="E:\\DEG_clean40"+to_string(rand()%9+1)+".avi";
    char* name=(char*)s.c_str();
	int NumFrames=1484;
	int count_step=1;

	int* ipR;/**< indeksy czerwonej maski */
    int ipR_size=30220;/**< rozmiar czerwonej maski */
    int* ipG;/**< indeksy zielonej maski */
    int ipG_size=33230;/**< rozmiar zielonej maski */
    int* ipB;/**< indeksy niebieskiej maski */
    int ipB_size=38077;/**< rozmiar niebieskiej maski */

    float* ICR_N;/**< czerwony wymaskowany obraz */
    float* ICG_N;/**< zielony wymaskowany obraz */
    float* ICB_N;/**< niebieski wymaskowany obraz */
    int* I_S_R;/**< indexy według wymaskowanej posortowanej thety */
    int* I_S_G;/**< indexy według wymaskowanej posortowanej thety */
    int* I_S_B;/**< indexy według wymaskowanej posortowanej thety */

    ipR=new int[ipR_size];
    for(int i=0;i<ipR_size;i++)
        ipR[i]=i;
    ipG=new int[ipG_size];
    for(int i=0;i<ipG_size;i++)
        ipG[i]=i;
    ipB=new int[ipB_size];
    for(int i=0;i<ipB_size;i++)
        ipB[i]=i;

    ICR_N=new float[ipR_size];
    for(int i=0;i<ipR_size;i++)
        ICR_N[i]=1.0f;
    ICG_N=new float[ipG_size];
    for(int i=0;i<ipG_size;i++)
        ICG_N[i]=1.0f;
    ICB_N=new float[ipB_size];
    for(int i=0;i<ipB_size;i++)
        ICB_N[i]=1.0f;

    I_S_R=new int[ipR_size];
    for(int i=0;i<ipR_size;i++)
        I_S_R[i]=i;
    I_S_G=new int[ipG_size];
    for(int i=0;i<ipG_size;i++)
        I_S_G[i]=i;
    I_S_B=new int[ipB_size];
    for(int i=0;i<ipB_size;i++)
        I_S_B[i]=i;

try
{
    printf("avireadtest - CUDA\n");
    /**< wczytaæ klatkê */
    /**< wzorzec konsument producent */
    CyclicBuffer cyclicBuffer;

    ifstream file (name, ios::in|ios::binary);
    if(!file.is_open())
    {
        throw string("file open failed");
    }
    if(!file.good())
    {
        throw string("opened file is not good");
    }

    /**< wątek z wyrażenia lmbda wykonuje się poprawnie :D */
    thread readMovieThread([&]
    {/**< uwaga wyra¿enie lambda w w¹tku */
        try
        {
        //throw 0;
        printf("readMovieThread\n");
        printf("name: %s\n",name);
        //return;

        const int skok = (640*480*2)+8;
        char* buff=nullptr;/**< aktualny adres zapisu z dysku */
        buffId* bId=nullptr;
        for(int i=0;i<NumFrames;i+=count_step)/**< czytanie klatek */
        {
            file.seekg((34824+(skok*(i))),ios::beg);
/**< \todo można poprawić, żeby przesunięcie było względem obecnej pozycji i dało się czytać filmy >4GB */
            bId=cyclicBuffer.claimForWrite();
            buff=bId->pt;
            bId->frameNo=i;
            for(int j=0;j<10;j++)
            {
                file.read(buff+j*65535,65535);/**< 64KB to optymalny rozmiar bloku czytanego z dysku */
            }
            cyclicBuffer.writeEnd(bId);
        }
        file.close();
        }
        catch(string& e)
        {
            printf("wyjątek: %s",e.c_str());
            exit(0);
        }
        catch(exception& e)
        {
            printf("wyjątek: %s",e.what());
            exit(0);
        }
        catch(...)
        {
            printf("nieznany wyjątek");
            exit(0);
        }

    });/**< readMovieThread lambda */

    setupCUDA_IC();

    setMasksAndImagesAndSortedIndexes(ipR,ipR_size,ipG,ipG_size,ipB,ipB_size,ICR_N,ICG_N,ICB_N,I_S_R,I_S_G,I_S_B);
    //setMasksAndImagesAndSortedIndexes(nullptr,0,nullptr,0,nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);

    /**< napisaæ szybsze odwracanie bajtu przy wyko¿ystaniu lookuptable */

    char* tmpBuff=nullptr;/**< tymczasowy adres bufora do odczytu */
    buffId* bID=nullptr;

    float R[700];
    float G[700];
    float B[700];

    long double licz=0.0f;
    int tmpFrameNo=-3;
    for(int k=0;k<NumFrames;k+=count_step)
    {
        bID=cyclicBuffer.claimForRead();
        tmpBuff=bID->pt;
        tmpFrameNo=bID->frameNo;
        copyBuff(tmpBuff);
        cyclicBuffer.readEnd(bID);
        doIC(R,G,B);
    }
    readMovieThread.join();

    freeCUDA_IC();
}
catch(string& e)
{
    printf("wyjątek: %s",e.c_str());
}
catch(exception& e)
{
    printf("wyjątek: %s",e.what());
}
catch(...)
{
    printf("nieznany wyjątek");
}

}
