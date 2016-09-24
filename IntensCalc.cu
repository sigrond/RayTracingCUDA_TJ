/** \file IntensCalc.cu
 * \author Tomasz Jakubczyk
 * \brief
 * kompilacja w matlabie:
 * nvmex -f nvmexopts64.bat IntensCalc.cu IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu CyclicBuffer.cpp MovingAverage_CUDA_kernel.cu FrameReader.cpp -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"
 */

#define WIN32
#include "Windows.h"
#include "mex.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <string>
#include <thread>
#include <exception>
#include <chrono>

#include "FrameReader.hpp"
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
    mxArray* tmp2;
    tmp=mxGetField(prhs[0],0,"fn");/**< nazwa pliku */
    name=mxArrayToString(tmp);
    printf("name: %s\n",name);

    tmp=mxGetField(prhs[0],0,"chR");
    if(tmp==NULL)
        printf("NULL chR");
    tmp2=mxGetProperty(tmp,0,"Value");
    if(tmp2==NULL)
    {
        printf("NULL chR Value");
    }
    else
    {
        tmp=tmp2;
    }
    value=(double*)mxGetPr(tmp);
    //printf("isR value: %lf\n",*value);
    isR=(bool)*value;
    printf("isR: %d\n",isR);

    tmp=mxGetField(prhs[0],0,"chG");
    tmp2=mxGetProperty(tmp,0,"Value");
    if(tmp2==NULL)
    {
        printf("NULL chG Value");
    }
    else
    {
        tmp=tmp2;
    }
    value=(double*)mxGetPr(tmp);
    isG=(bool)*value;
    printf("isG: %d\n",isG);

    tmp=mxGetField(prhs[0],0,"chB");
    tmp2=mxGetProperty(tmp,0,"Value");
    if(tmp2==NULL)
    {
        printf("NULL chB Value");
    }
    else
    {
        tmp=tmp2;
    }
    value=(double*)mxGetPr(tmp);
    isB=(bool)*value;
    printf("isB: %d\n",isB);

    unsigned char * BgMaskR=nullptr;
    unsigned char * BgMaskG=nullptr;
    unsigned char * BgMaskB=nullptr;
    tmp=mxGetField(prhs[0],0,"BackgroundMaskR");
    if(tmp==nullptr)
        printf("no handles.BackgroundMaskR?\n");
    BgMaskR=(unsigned char*)mxGetPr(tmp);
    if(BgMaskR==nullptr)
        printf("BgMaskR==nullptr\n");
    int BgM_NR=mxGetN(tmp);
    int BgM_MR=mxGetM(tmp);
    printf("BgM_NR: %d, BgM_MR: %d\n",BgM_NR,BgM_MR);

    tmp=mxGetField(prhs[0],0,"BackgroundMaskG");
    if(tmp==nullptr)
        printf("no handles.BackgroundMaskG?\n");
    BgMaskG=(unsigned char*)mxGetPr(tmp);
    if(BgMaskG==nullptr)
        printf("BgMaskG==nullptr\n");
    int BgM_NG=mxGetN(tmp);
    int BgM_MG=mxGetM(tmp);
    printf("BgM_NG: %d, BgM_MG: %d\n",BgM_NG,BgM_MG);

    tmp=mxGetField(prhs[0],0,"BackgroundMaskB");
    if(tmp==nullptr)
        printf("no handles.BackgroundMaskB?\n");
    BgMaskB=(unsigned char*)mxGetPr(tmp);
    if(BgMaskB==nullptr)
        printf("BgMaskB==nullptr\n");
    int BgM_NB=mxGetN(tmp);
    int BgM_MB=mxGetM(tmp);
    printf("BgM_NB: %d, BgM_MB: %d\n",BgM_NB,BgM_MB);

    #ifdef DEBUG
    for(int i=0;i<BgM_M;i++)//480
    {
        for(int j=0;j<BgM_N;j++)//640
        {
            if(i%16==8 && j%16==8)
            printf("%d ",BgMask[j*BgM_M+i]);
        }
        if(i%16==8)
        printf("\n");
    }
    #endif // DEBUG


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
        printf("mały plik #1\n");
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
            printf("duży plik #2\n");
            smallf=false;
        }
        else
        {
            //przypadek 3 - trzeba przejżeć nagłówek
            printf("format pliku #3\n");
            char* buff0=new char[65535+8];
            int ct=0;
            file.seekg(0,ios::beg);
            while(file.good() && !b)
            {
                file.read(buff0,65535);
                for(int j=0;j<65535 && !b;j++)
                {
                    b=true;
                    ct++;
                    for(int i=0;i<8;i++)
                    {
                        b&=frameStartCode[i]==buff0[j+i];
                    }
                    if(b)
                    {
                        printf("first frame header ct=%d\n",ct);
                        file.seekg(65535-j,ios::cur);
                        //printf("tellg: %lld\n",file.tellg());
                    }
                }
            }
            delete[] buff0;
            FrameStartCode=frameStartCode;
            fileFirstFrame=ct;
            printf("Warning! JUNK skipping is still experimental!\n");
            //return;
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

    /**< szybkie obliczenie z ilu pikseli składa się tło */
    float BgMaskSizeR[2]={0.0f, 0.0f};
    for(int i=0;i<BgM_MR;i++)//480
    {
        for(int j=0;j<BgM_NR;j++)//640
        {
            if(BgMaskR[j*BgM_MR+i]==1)
            {
                if(j<BgM_NR/2)
                {
                    BgMaskSizeR[0]+=1.0f;
                }
                else
                {
                    BgMaskSizeR[1]+=1.0f;
                }
            }
        }
    }
    printf("BgMaskSizeR[0]: %f\n",BgMaskSizeR[0]);
    printf("BgMaskSizeR[1]: %f\n",BgMaskSizeR[1]);

    float BgMaskSizeG[2]={0.0f, 0.0f};
    for(int i=0;i<BgM_MG;i++)//480
    {
        for(int j=0;j<BgM_NG;j++)//640
        {
            if(BgMaskG[j*BgM_MG+i]==1)
            {
                if(j<BgM_NG/2)
                {
                    BgMaskSizeG[0]+=1.0f;
                }
                else
                {
                    BgMaskSizeG[1]+=1.0f;
                }
            }
        }
    }
    printf("BgMaskSizeG[0]: %f\n",BgMaskSizeG[0]);
    printf("BgMaskSizeG[1]: %f\n",BgMaskSizeG[1]);

    float BgMaskSizeB[2]={0.0f, 0.0f};
    for(int i=0;i<BgM_MB;i++)//480
    {
        for(int j=0;j<BgM_NB;j++)//640
        {
            if(BgMaskB[j*BgM_MB+i]==1)
            {
                if(j<BgM_NB/2)
                {
                    BgMaskSizeB[0]+=1.0f;
                }
                else
                {
                    BgMaskSizeB[1]+=1.0f;
                }
            }
        }
    }
    printf("BgMaskSizeB[0]: %f\n",BgMaskSizeB[0]);
    printf("BgMaskSizeB[1]: %f\n",BgMaskSizeB[1]);

    setMasksAndImagesAndSortedIndexes(ipR,ipR_size,ipG,ipG_size,ipB,ipB_size,ICR_N,ICG_N,ICB_N,I_S_R,I_S_G,I_S_B,
                                      BgMaskR,BgMaskSizeR,BgMaskG,BgMaskSizeG,BgMaskB,BgMaskSizeB);

    /**< napisaæ szybsze odwracanie bajtu przy wyko¿ystaniu lookuptable */

    buffId* bID=nullptr;

    #ifndef OLD_DECODEC

    FrameReader frameReader(&cyclicBuffer);

    thread correctnessControlThread([&]
    {
        try
        {
            char* tmpFrame=nullptr;
            for(int k=0;k<NumFrames;k++)
            {
                if(!frameReader.correctnessControl.checkFrame())
                {
                    tmpFrame=frameReader.correctnessControl.decodeFrame();
                    if(tmpFrame!=nullptr)
                    {
                        copyBuff(tmpFrame);
                        doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);
                    }
                }
            }
        }
        catch(exception& e)
        {
            printf("wyjątek: %s",e.what());
            string s=e.what();
            MessageBox(NULL,s.c_str(),NULL,NULL);
            system("pause");
        }
        catch(...)
        {
            printf("nieznany wyjątek");
            string s="nieznany wyjątek";
            MessageBox(NULL,s.c_str(),NULL,NULL);
            system("pause");
        }
    });

    char* frame=nullptr;
    for(int k=0;k<NumFrames;k++)
    {
        frame=frameReader.getFrame();
        copyBuff(frame);
        doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);

        if(k%500==0 || k==NumFrames-1)
        {
            printf("\b\b\b\b\b\b\b\b\n%5.2f%%\n",(float)(k*100)/(float)NumFrames);
            //mexEvalString("drawnow;");
            mexEvalString("pause(.001);");
        }
    }

    #else

    char* tmpBuff=nullptr;/**< tymczasowy adres bufora do odczytu */

    long double licz=0.0f;
    int tmpFrameNo=-3;
    int frameEnd=614400;/**< miejsce od którego w buforze może wystąpić nagłówek następnej klatki */
    char* currentFrame=new char[655350];
    char* nextFrame=new char[655350];
    int nextFrameElements=0;
    int garbageElements=0;
    int dstOff=0;
    int srcOff=0;
    int cpyNum=0;
    char* tmpFrame=nullptr;
    int checkBuffForStartCodePosition=0;
    int lastFrameNo=-1;
    char junkCode[]="JUNK";
    //long long int junkCt=0;
    int junkSize=0;
    bool junkB=true;
    int checkBuffForJunkPosition=0;
    printf("Progress:   %5.2f%%\n",0.0f);
    for(int k=0;k<NumFrames;k+=count_step)
    {
        bID=cyclicBuffer.claimForRead();
        tmpBuff=bID->pt;
        if(tmpBuff==nullptr)
        {
            printf("Critical Error tmpBuff==nullptr k: %d\n",k);
        }
        tmpFrameNo=bID->frameNo;
        if(tmpFrameNo!=k)
        {
            //printf("tmpFrameNo: %d k: %d\n",tmpFrameNo,k);
            //throw string("zgubiona numeracja klatek");
        }

        if(lastFrameNo+1!=tmpFrameNo)
        {
            printf("lastFrameNo: %d tmpFrameNo: %d\n",lastFrameNo,tmpFrameNo);
            cyclicBuffer.printStatus();
        }
        lastFrameNo=tmpFrameNo;

        for(int j=frameEnd;j<65535*10-8;j++)
        {
            junkB=true;
            for(int i=checkBuffForJunkPosition;i<4 && junkB;i++)
            {
                junkB&=junkCode[i]==tmpBuff[j+i];
                if(junkB)
                    checkBuffForJunkPosition=i;
                else
                    checkBuffForJunkPosition=0;
            }
            if(junkB)
            {
                //printf("znaleziono JUNK ct=%d\n",ct);
                //junkCt=ct;
                junkSize=*(int*)(tmpBuff+j+4);
                junkSize+=4;
                //printf("JUNK size: %d\n",junkSize);
            }
            b=true;
            for(int i=checkBuffForStartCodePosition;i<8 && b;i++)
            {
                if(j+i<0 || j+i>=65535*10)
                {
                    printf("error-1 k: %d próba odczytu z poza zakresu tmpBuff\n",k);
                    b=false;
                    break;
                }
                //b&=FrameStartCode[i]==tmpBuff[j+i];
                b&=frameStartCode[i]==tmpBuff[j+i] || frameStartCodeS[i]==tmpBuff[j+i];
                if(b)
                {
                    checkBuffForStartCodePosition=i;/**< zapisujemy na wypadek gdyby bufor przecioł nagłówek */
                }
                else
                {
                    checkBuffForStartCodePosition=0;
                }
            }

            if(b)
            {
                if(k==0)
                {
                    srcOff=j-640*480*2-junkSize;
                    junkSize=0;
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
                    garbageElements=nextFrameElements+j-640*480*2-junkSize;/**< śmieci za nagłówkiem klatki */
                    if(garbageElements<0 || garbageElements>614400)
                    {
                        //printf("debug2 k: %d garbageElements: %d nextFrameElements: %d j: %d\n",k,garbageElements,nextFrameElements,j);
                        //break;
                        garbageElements=nextFrameElements;
                    }
                    cpyNum=nextFrameElements-garbageElements;
                    if(cpyNum<0 || cpyNum>614400)
                    {
                        printf("error5 k: %d cpyNum: %d           \n",k,cpyNum);
                        break;
                    }
                    if(cpyNum+garbageElements<0 || cpyNum+garbageElements>655350)
                    {
                        printf("error5b k: %d cpyNum: %d garbageElements: %d           \n",k,cpyNum, garbageElements);
                        break;
                    }
                    if(cpyNum>0 && cpyNum<=614400)
                        memcpy(currentFrame,nextFrame+garbageElements,cpyNum);/**< obecną klatkę dopełniamy tym co zostało z poprzedniego odczytu */
                    dstOff=nextFrameElements-garbageElements;/**< gdzie w obecnej klatce kończą się dane z poprzedniego bloku */
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
                    /**<                      v- kopiujemy do nagłówka, albo tylko do JUNK'u przed nagłówkiem */
                    /**<                            v- j wskazuje na nagłówek jeszcze następnej klatki i tylko dopychamy dane do obecnej klatki */
                    cpyNum=j-junkSize+dstOff<=614400?(j-junkSize>=0?j-junkSize:0):614400-dstOff;/**< większa manifestacja chaosu */
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
                    if(cpyNum+j+8<0 || cpyNum+j+8>65535*10)
                    {
                        printf("error12b k: %d cpyNum: %d j: %d\n",k,cpyNum,j);
                        break;
                    }
                    if(cpyNum>0)
                        memcpy(nextFrame,tmpBuff+j+8,cpyNum);/**< zapisujemy odczytany nadmiar */
                    nextFrameElements=65535*10-(j+8);
                    junkSize=0;/**< nie zawsze przed nagłowkiem musi być JUNK */
                    if(nextFrameElements>=614400)
                    {
                        //printf("debug3 k: %d nextFrameElements: %d\n",k,nextFrameElements);
                        copyBuff(currentFrame);
                        if(k<NumFrames)
                        {
                            doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);
                        }
                        else
                        {
                            printf("pominięcie operacji dla nieprzewidzianych klatek k: %d\n",k);
                        }
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

        #ifdef SIM_HEAVY_CALC
        this_thread::sleep_for (chrono::milliseconds(10));
        #endif // SIM_HEAVY_CALC

        //copyBuff(tmpBuff);
        cyclicBuffer.readEnd(bID);
        if(k<NumFrames)
        {
            doIC(I_Red+k*700,I_Green+k*700,I_Blue+k*700);
        }
        else
        {
            printf("pominięcie operacji dla nieprzewidzianych klatek k: %d\n",k);
        }
        if(k%500==0 || k==NumFrames-1)
        {
            printf("\b\b\b\b\b\b\b%5.2f%%\n",(float)(k*100)/(float)NumFrames);
            //mexEvalString("drawnow;");
            mexEvalString("pause(.001);");
        }
    }
    #endif // OLD_DECODEC
    finished=true;
    printf("finshed reading from cyclic bufor\n");
    printf("itemCount: %d\n",cyclicBuffer.tellItemCount());
    mexEvalString("pause(.001);");
    if(cyclicBuffer.tellItemCount()>0)
    {
        bID=cyclicBuffer.claimForRead();
        cyclicBuffer.readEnd(bID);
    }


    readMovieThread.join();
    printf("readMovieThread joined\n");
    //mexEvalString("drawnow;");

    correctnessControlThread.join();
    printf("correctnessControlThread joined\n");

    #ifdef OLD_DECODEC
    delete[] currentFrame;
    delete[] nextFrame;
    #endif // OLD_DECODEC
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
