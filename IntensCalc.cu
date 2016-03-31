/** \file IntensCalc.cu
 * \author Tomasz Jakubczyk
 * \brief
 * kompilacja w matlabie:
 * nvmex -f nvmexopts64.bat IntensCalc.cu IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu CyclicBuffer.cpp MovingAverage_CUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"
 */

#define WIN32
#include "Windows.h"
#include "mex.h"
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <string>
#include <thread>
#include <exception>

#include "CyclicBuffer.hpp"
#include "IntensCalc_CUDA.cuh"

using namespace std;

#ifdef DEBUG
extern unsigned short* previewFa;
//unsigned short* previewFa;
extern short* previewFb;

extern float* previewFc;

extern float* previewFd;
#endif // DEBUG


/** \brief
 * function [I_Red,I_Green,I_Blue] = IntensCalc(handles,count_step,NumFrames,ipR,ipG,ipB,ICR_N,ICG_N,ICB_N,I_S_R,I_S_G,I_S_B)
 * \param nlhs int
 * \param plhs[] mxArray*
 * \param nrhs int
 * \param prhs[] const mxArray*
 * \return void
 *
 */
//__host__
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    #ifdef DEBUG
    printf("DEBUG ver.\n");
    #endif // DEBUG

    int count_step=1;/**< co która klatka */
    int NumFrames;/**< liczba klatek */
    int* ipR;/**< indeksy czerwonej maski */
    int ipR_size=0;/**< rozmiar czerwonej maski */
    int* ipG;/**< indeksy zielonej maski */
    int ipG_size=0;/**< rozmiar zielonej maski */
    int* ipB;/**< indeksy niebieskiej maski */
    int ipB_size=0;/**< rozmiar niebieskiej maski */
    char* name;/**< nazwa pliku z pe³n¹ œcierzk¹ */
    float* ICR_N;/**< czerwony wymaskowany obraz */
    float* ICG_N;/**< zielony wymaskowany obraz */
    float* ICB_N;/**< niebieski wymaskowany obraz */
    int* I_S_R;/**< indexy według wymaskowanej posortowanej thety */
    int* I_S_G;/**< indexy według wymaskowanej posortowanej thety */
    int* I_S_B;/**< indexy według wymaskowanej posortowanej thety */

    /**< sprawdzanie argumentów */
    if(nlhs!=
    #ifdef DEBUG
        7)
    #else
        3)
    #endif // DEBUG
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
    bool isR=false, isG=false, isB=false;/**< czy jest zaznaczony kolor */
    double* value;
    mxArray* tmp;
    tmp=mxGetField(prhs[0],0,"fn");/**< nazwa pliku */
    name=mxArrayToString(tmp);
    printf("name: %s\n",name);

    tmp=mxGetField(prhs[0],0,"chR");
    if(tmp==NULL)
        printf("NULL chR");
    tmp=mxGetProperty(tmp,0,"Value");
    if(tmp==NULL)
        printf("NULL chR Value");
    value=(double*)mxGetPr(tmp);
    //printf("isR value: %lf\n",*value);
    isR=(bool)*value;
    printf("isR: %d\n",isR);

    tmp=mxGetField(prhs[0],0,"chG");
    tmp=mxGetProperty(tmp,0,"Value");
    value=(double*)mxGetPr(tmp);
    isG=(bool)*value;
    printf("isG: %d\n",isG);

    tmp=mxGetField(prhs[0],0,"chB");
    tmp=mxGetProperty(tmp,0,"Value");
    value=(double*)mxGetPr(tmp);
    isB=(bool)*value;
    printf("isB: %d\n",isB);

    count_step=*((int*)mxGetPr(prhs[1]));
    if(mxGetN(prhs[1])*mxGetM(prhs[1])!=1)
    {
        printf("2nd argument (count_step) must be a number\n");
        return;
    }
    printf("count_step: %d\n",count_step);
    NumFrames=*((int*)mxGetPr(prhs[2]));
    printf("NumFrames: %d\n",NumFrames);
    if(mxGetN(prhs[2])*mxGetM(prhs[2])!=1)
    {
        printf("3rd argument (NumFrames) must be a number\n");
        return;
    }

    ipR=(int*)mxGetPr(prhs[3]);
    ipR_size=mxGetN(prhs[3])*mxGetM(prhs[3]);
    printf("ipR_size: %d\n",ipR_size);
    ipG=(int*)mxGetPr(prhs[4]);
    ipG_size=mxGetN(prhs[4])*mxGetM(prhs[4]);
    printf("ipG_size: %d\n",ipG_size);
    ipB=(int*)mxGetPr(prhs[5]);
    ipB_size=mxGetN(prhs[5])*mxGetM(prhs[5]);
    printf("ipB_size: %d\n",ipB_size);

    ICR_N=(float*)mxGetPr(prhs[6]);
    printf(": %d\n",mxGetN(prhs[6])*mxGetM(prhs[6]));
    ICG_N=(float*)mxGetPr(prhs[7]);
    printf(": %d\n",mxGetN(prhs[7])*mxGetM(prhs[7]));
    ICB_N=(float*)mxGetPr(prhs[8]);
    printf(": %d\n",mxGetN(prhs[8])*mxGetM(prhs[8]));
    //for(int i=0;i<ipB_size;i++)
    //    printf("%f,",ICB_N[i]);

    I_S_R=(int*)mxGetPr(prhs[9]);
    printf(": %d\n",mxGetN(prhs[9])*mxGetM(prhs[9]));
    I_S_G=(int*)mxGetPr(prhs[10]);
    printf(": %d\n",mxGetN(prhs[10])*mxGetM(prhs[10]));
    I_S_B=(int*)mxGetPr(prhs[11]);
    printf(": %d\n",mxGetN(prhs[11])*mxGetM(prhs[11]));

    /**< przygotowanie zwracanych macierzy */
    int dimsI_Red[2]={700,NumFrames};
    plhs[0]=mxCreateNumericArray(2,dimsI_Red,mxSINGLE_CLASS,mxREAL);
    float* I_Red=(float*)mxGetPr(plhs[0]);
    int dimsI_Green[2]={700,NumFrames};
    plhs[1]=mxCreateNumericArray(2,dimsI_Green,mxSINGLE_CLASS,mxREAL);
    float* I_Green=(float*)mxGetPr(plhs[1]);
    int dimsI_Blue[2]={700,NumFrames};
    plhs[2]=mxCreateNumericArray(2,dimsI_Blue,mxSINGLE_CLASS,mxREAL);
    float* I_Blue=(float*)mxGetPr(plhs[2]);
    printf("I_Blue NxM: %dx%d\n",mxGetN(plhs[2]),mxGetM(plhs[2]));

    #ifdef DEBUG
    int dimsP[2]={640,480};
    plhs[3]=mxCreateNumericArray(2,dimsP,mxUINT16_CLASS,mxREAL);
    previewFa=(unsigned short*)mxGetPr(plhs[3]);

    plhs[4]=mxCreateNumericArray(2,dimsP,mxINT16_CLASS,mxREAL);
    previewFb=(short*)mxGetPr(plhs[4]);

    int dimsV[1]={ipR_size};
    plhs[5]=mxCreateNumericArray(1,dimsV,mxSINGLE_CLASS,mxREAL);
    previewFc=(float*)mxGetPr(plhs[5]);

    plhs[6]=mxCreateNumericArray(1,dimsV,mxSINGLE_CLASS,mxREAL);
    previewFd=(float*)mxGetPr(plhs[6]);
    #endif // DEBUG

    //return;

    if(!(isR||isG||isB))
    {
        printf("no color is chosen\n");
        return;
    }

try
{

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
    const unsigned int bigFileFirstFrame=64564;
    const unsigned int smallFileFirstFrame=34824;
    unsigned int fileFirstFrame=0;
    char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
	char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
	char* FrameStartCode=nullptr;
	char codeBuff[8];

	file.seekg(smallFileFirstFrame-8,ios::beg);
	file.read(codeBuff,8);
	bool b=true;
	bool smallf=true;
	for(int i=0;i<8;i++)
    {
        b&=frameStartCodeS[i]==codeBuff[i];
        //printf("0x%X ",codeBuff[i]);
    }
    if(b)
    {
        fileFirstFrame=smallFileFirstFrame;
        FrameStartCode=frameStartCodeS;
        printf("mały plik\n");
    }
    else
    {
        //przypadek 2 - duży plik
        file.seekg(bigFileFirstFrame-8,ios::beg);
        file.read(codeBuff,8);
        b=true;
        for(int i=0;i<8;i++)
        {
            b&=frameStartCode[i]==codeBuff[i];
            //printf("0x%X ",codeBuff[i]);
        }
        if(b)
        {
            fileFirstFrame=bigFileFirstFrame;
            FrameStartCode=frameStartCode;
            printf("duży plik\n");
            smallf=false;
        }
        else
        {
            //przypadek 3 - trzeba przejżeć nagłówek
            printf("format pliku jeszcze nie obsługiwany\n");
            return;
        }
    }

    bool finished=false;

    /**< wątek z wyrażenia lmbda wykonuje się poprawnie :D */
    thread readMovieThread([&]
    {/**< uwaga wyra¿enie lambda w w¹tku */
        buffId* bId=nullptr;
        try
        {
        //throw 0;
        printf("readMovieThread\n");
        //return;
        const int skok = (640*480*2)+8;
        char* buff=nullptr;/**< aktualny adres zapisu z dysku */
        file.seekg(fileFirstFrame,ios::beg);
        for(int i=0;i<NumFrames && !finished;i+=count_step)/**< czytanie klatek */
        {
            //file.seekg((34824+(skok*(i))),ios::beg);
/**< \todo można poprawić, żeby przesunięcie było względem obecnej pozycji i dało się czytać filmy >4GB */
            bId=cyclicBuffer.claimForWrite();
            buff=bId->pt;
            bId->frameNo=i;
            for(int j=0;j<10 && file.good();j++)
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
            string s="wyjątek: "+e;
            MessageBox(NULL,s.c_str(),NULL,NULL);
            system("pause");
            cyclicBuffer.writeEnd(bId);
            file.close();
            //mexEvalString("drawnow;");
            //exit(0);
        }
        catch(exception& e)
        {
            printf("wyjątek: %s",e.what());
            string s=e.what();
            MessageBox(NULL,s.c_str(),NULL,NULL);
            system("pause");
            cyclicBuffer.writeEnd(bId);
            file.close();
            //mexEvalString("drawnow;");
            //exit(0);
        }
        catch(...)
        {
            printf("nieznany wyjątek");
            string s="nieznany wyjątek";
            MessageBox(NULL,s.c_str(),NULL,NULL);
            system("pause");
            cyclicBuffer.writeEnd(bId);
            file.close();
            //mexEvalString("drawnow;");
            //exit(0);
        }

    });/**< readMovieThread lambda */

    setupCUDA_IC();

    setMasksAndImagesAndSortedIndexes(ipR,ipR_size,ipG,ipG_size,ipB,ipB_size,ICR_N,ICG_N,ICB_N,I_S_R,I_S_G,I_S_B);

    /**< napisaæ szybsze odwracanie bajtu przy wyko¿ystaniu lookuptable */

    char* tmpBuff=nullptr;/**< tymczasowy adres bufora do odczytu */
    buffId* bID=nullptr;

    long double licz=0.0f;
    int tmpFrameNo=-3;
    int frameEnd=614400;/**< miejsce od którego w buforze może wystąpić nagłówek następnej klatki */
    char* currentFrame=new char[614400];
    char* nextFrame=new char[655350];
    int nextFrameElements=0;
    int garbageElements=0;
    int dstOff=0;
    int srcOff=0;
    int cpyNum=0;
    char* tmpFrame=nullptr;
    for(int k=0;k<NumFrames;k+=count_step)
    {
        bID=cyclicBuffer.claimForRead();
        tmpBuff=bID->pt;
        tmpFrameNo=bID->frameNo;
        if(tmpFrameNo!=k)
        {
            //printf("tmpFrameNo: %d k: %d\n",tmpFrameNo,k);
            //throw string("zgubiona numeracja klatek");
        }

        for(int j=frameEnd;j<65535*10-8;j++)
        {
            b=true;
            for(int i=0;i<8 && b;i++)
            {
                b&=FrameStartCode[i]==tmpBuff[j+i];
            }
            if(b)
            {
                if(k==0)
                {
                    srcOff=j-640*480*2;
                    if(srcOff<0 || srcOff>65535*10)
                    {
                        printf("error1 k: %d srcOff: %d\n",k,srcOff);
                        break;
                    }
                    memcpy(currentFrame,tmpBuff+srcOff,640*480*2);/**< wszystko co jest klatką */
                    srcOff=j+8;
                    if(srcOff<0 || srcOff>65535*10)
                    {
                        printf("error2 k: %d srcOff: %d\n",k,srcOff);
                        break;
                    }
                    cpyNum=65535*10-(j+8);
                    if(cpyNum<0 || cpyNum>614400)
                    {
                        printf("error3 k: %d cpyNum: %d\n",k,cpyNum);
                        break;
                    }
                    memcpy(nextFrame,tmpBuff+srcOff,cpyNum);/**< nadmiar do następnej klatki */
                    nextFrameElements=65535*10-(j+8);/**< ile elementów weszło do następnej klatki */
                    garbageElements=j-640*480*2;/**< ile śmieci mamy za nagłówkiem klatki */
                }
                else
                {
                    //if(j<=40950)
                    //{
                    //    printf("debug1 k: %d j: %d\n",k,j);
                    //}
                    garbageElements=nextFrameElements+j-640*480*2;
                    if(garbageElements<0 || garbageElements>614400)
                    {
                        //printf("debug2 k: %d garbageElements: %d nextFrameElements: %d j: %d\n",k,garbageElements,nextFrameElements,j);
                        //break;
                        garbageElements=nextFrameElements;
                    }
                    cpyNum=nextFrameElements-garbageElements;
                    if(cpyNum<0 || cpyNum>614400)
                    {
                        printf("error5 k: %d cpyNum: %d\n",k,cpyNum);
                        break;
                    }
                    if(cpyNum>0)
                        memcpy(currentFrame,nextFrame+garbageElements,cpyNum);/**< obecną klatkę dopełniamy tym co zostało z poprzedniego odczytu */
                    dstOff=nextFrameElements-garbageElements;
                    if(dstOff<0 || dstOff>614400)
                    {
                        printf("error6 k: %d dstOff: %d\n",k,dstOff);
                        break;
                    }
                    //srcOff=j-640*480*2-(nextFrameElements-garbageElements);
                    srcOff=j>614400?j-614400:0;
                    if(srcOff<0 || srcOff>65535*10)
                    {
                        printf("error7 k: %d srcOff: %d\n",k,srcOff);
                        break;
                    }
                    cpyNum=j+dstOff<=614400?j:614400-dstOff;/**< większa manifestacja chaosu */
                    if(cpyNum<0 || cpyNum>614400)
                    {
                        printf("error8 k: %d cpyNum: %d\n",k,cpyNum);
                        break;
                    }
                    if((dstOff+cpyNum)<0 || (dstOff+cpyNum)>614400)
                    {
                        printf("error9 k: %d dstOff: %d cpyNum: %d\n",k,dstOff,cpyNum);
                        break;
                    }
                    if((srcOff+cpyNum)<0 || (srcOff+cpyNum)>655350)
                    {
                        printf("error10 k: %d srcOff: %d cpyNum: %d\n",k,srcOff,cpyNum);
                        break;
                    }
                    if(cpyNum>0)
                        memcpy(currentFrame+dstOff,tmpBuff+srcOff,cpyNum);/**< następnie dopełniamy obecną klatkę tym co właśnie przeczytaliśmy */
                    srcOff=j+8;
                    if(srcOff<0 || srcOff>65535*10)
                    {
                        printf("error11 k: %d srcOff: %d\n",k,srcOff);
                        break;
                    }
                    cpyNum=65535*10-(j+8);
                    if(cpyNum<0 || cpyNum>65535*10)
                    {
                        printf("error12 k: %d cpyNum: %d\n",k,cpyNum);
                        break;
                    }
                    if(cpyNum>0)
                        memcpy(nextFrame,tmpBuff+j+8,cpyNum);/**< zapisujemy odczytany nadmiar */
                    nextFrameElements=65535*10-(j+8);
                    if(nextFrameElements>=614400)
                    {
                        //printf("debug3 k: %d nextFrameElements: %d\n",k,nextFrameElements);
                        copyBuff(currentFrame);
                        doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);
                        k++;
                        j+=614400;
                        continue;
                    }
                }
                frameEnd=j+8-40950;
                //if(frameEnd<40950)
                //{
                //    printf("debug4 k: %d frameEnd: %d\n",k,frameEnd);
                    //break;
                //}
                break;
            }
        }
        copyBuff(currentFrame);

        //copyBuff(tmpBuff);
        cyclicBuffer.readEnd(bID);
        doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);
        //if((k*100/NumFrames)%10==0)
        //{
        //    printf("%d%% ",k*100/NumFrames);
            //mexEvalString("drawnow;");
        //}
    }
    finished=true;
    printf("finshed reading from cyclic bufor\n");
    mexEvalString("drawnow;");
    bID=cyclicBuffer.claimForRead();
    cyclicBuffer.readEnd(bID);
    readMovieThread.join();
    printf("readMovieThread joined\n");
    //mexEvalString("drawnow;");

    delete[] currentFrame;
    delete[] nextFrame;
    //cyclicBuffer.~CyclicBuffer();

    freeCUDA_IC();
}
catch(string& e)
{
    printf("wyjątek: %s",e.c_str());
    string s="wyjątek: "+e;
    MessageBox(NULL,s.c_str(),NULL,NULL);
    system("pause");
    //readMovieThread.join();
}
catch(exception& e)
{
    printf("wyjątek: %s",e.what());
    string s=e.what();
    MessageBox(NULL,s.c_str(),NULL,NULL);
    system("pause");
    //readMovieThread.join();
}
catch(...)
{
    printf("nieznany wyjątek");
    string s="nieznany wyjątek";
    MessageBox(NULL,s.c_str(),NULL,NULL);
    system("pause");
    //readMovieThread.join();
}
    /**< czytaæ wêksze bloki danych z pliku ni¿ po jednym znaku */
    /**< dla klatki zastosowaæ demosaic */
    /**< uzyskaæ wymaskowan¹ klatkê */
    /**< podzieliæ wymaskowan¹ klatkê przez macierz korekcyjn¹ */
    /**< u¿ywaæ strumieni CUDA i lub w¹tków, ¿eby jednoczeœnie czytaæ plik i liczyæ */
    /**< ka¿d¹ posortowan¹ klatkê wyg³adziæ œredni¹ krocz¹c¹ */
    /**< zwróciæ 700 równomiernie wybranych punktów dla ka¿dej klatki */
}
