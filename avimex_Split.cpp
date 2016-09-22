/*
This program reads .avi file created using AVT PIKE DS filter ADW6 and split using AVI SPLIT (16 bit RAW format)
Compiled successfully under Linux Mint 11 64bit with Matlab 2010b 64bit and Windows 7 32 bit with Matlab 2008b 32bit using VS 2005.
This code only reads a single 16bit frame 640x480. Returns a Matlab 640x480 uint16 matrix without any debayering.
It was meant to be used with AviReadPike.m matlab script, that does debayering and process a multi-frame video frame by frame returning 4dimensional matrix (frame no, height, width, RGB) of uint16 numbers.

First input argument is filename (with .avi extension), second is frame number (first frame is indexed with 0 IMPORTANT).

Code includes two functions of 8 and 6 bit number reversal using byte shifts.

If you would like to adjust this code to work with another codecs you should change variables: "pocz", which is the offset from the beginning of the file to the beginning of the first frame in bytes (including frame header '00db ....' and "skok" which is the size of a single frame in bytes.

  Author: Szymon Migacz, mailto: szmigacz(at)gmail.com
  Version 1.0 from 12.07.11
*/

/** \file avimex_Split.cpp
 * \author Tomasz Jakubczyk
 * \email sigrond93(at)gmail.com
 * \date 29.02.2016
 * \brief zmiany maj¹ce na celu doprowadziæ do prwid³owego dzia³ania
 * (bez efektu ziarna; wartoœci pixeli uciête z góry) w wersji 64 bitowej
 * \date 23.03.2016
 * zmiany mające na celu umożliwienie poprawnego czytania również
 * nie podzielonych filmów
 * \date 27.07.2016
 * próba naprawienia czytania klatek
 */



#include "mex.h"
#include<fstream>
#include<limits>
#include<exception>
#include<vector>

using namespace std;


//6 bit byte reverse

unsigned short int odwroc6(unsigned short int i) {
	unsigned short int temp=0;
	if(i%2) temp += 32;
	if((i>>1)%2) temp += 16;
	if((i>>2)%2) temp += 8;
	if((i>>3)%2) temp += 4;
	if((i>>4)%2) temp += 2;
	if((i>>5)%2) temp += 1;
	return temp;
}
//end of the reverse
//MEX function begins:

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
try{
    const unsigned char reverse6bitLookupTable[]={
0x00,0x20,0x10,0x30,0x08,0x28,0x18,0x38,0x04,0x24,0x14,0x34,0x0C,0x2C,0x1C,0x3C,
0x02,0x22,0x12,0x32,0x0A,0x2A,0x1A,0x3A,0x06,0x26,0x16,0x36,0x0E,0x2E,0x1E,0x3E,
0x01,0x21,0x11,0x31,0x09,0x29,0x19,0x39,0x05,0x25,0x15,0x35,0x0D,0x2D,0x1D,0x3D,
0x03,0x23,0x13,0x33,0x0B,0x2B,0x1B,0x3B,0x07,0x27,0x17,0x37,0x0F,0x2F,0x1F,0x3F};
/**< tablica odwracająca kolejność 6 młodszych bitów */
    const unsigned int bigFileFirstFrame=64564;
    const unsigned int smallFileFirstFrame=34824;
    unsigned int fileFirstFrame=0;
	char *filename;
	int *numer;
	numer=(int*)mxGetPr(prhs[1]);
	filename = mxArrayToString(prhs[0]);
	ifstream file (filename, ios::in|ios::binary);
	char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
	char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
	char codeBuff[8];
	//przypadek 1 - mały plik
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
            printf("duży plik\n");
            smallf=false;
        }
        else
        {
            //przypadek 3 - trzeba przejżeć nagłówek
            printf("format pliku jeszcze nie obsługiwany\n");
            smallf=false;
        }
    }


	unsigned short int *klatka;
	long long int skok = (640*480*2)+8;
	plhs[0]=mxCreateNumericMatrix(640,480,mxUINT16_CLASS,mxREAL);
	klatka=(unsigned short int*) mxGetPr(plhs[0]);
	long long int pos=((long long int)fileFirstFrame+(skok*(long long int)(*numer)));
	#ifndef _WIN64
	printf("WIN64 not defined\n");
	#endif // _WIN64
	#if 0
	//printf("sizeof(streamoff): %d\n",sizeof(streamoff));
	file.seekg(0,ios::beg);
	while(pos>=numeric_limits<long int>::max())
    {
        printf("pos: %lld\n",pos);
        file.seekg(numeric_limits<long int>::max(),ios::cur);
        pos-=numeric_limits<long int>::max();
        //file.read(codeBuff,8);
        //pos-=8;
        if(!file.good())
        {
            printf("file is not good(1)!\n");
            ios_base::iostate state=file.rdstate();
            printf("state: %d\n",state);
            if(file.eof())
            {
                printf("file.eof\n");
            }
            if(file.fail())
            {
                printf("file.fail\n");
            }
            if(file.bad())
            {
                printf("file.bad\n");
            }
            printf("tellg: %lld\n",file.tellg());
            file.clear();
            break;
        }
    }
    file.seekg(pos,ios::cur);
    #else
    file.seekg(0,file.end);
        printf("tellg end: %lld\n",file.tellg());
	file.seekg(pos,ios::beg);
	#endif // _WIN64

	if(!file.good())
    {
        printf("file is not good(2)!\n");
        ios_base::iostate state=file.rdstate();
        printf("state: %d\n",state);
        if(file.eof())
        {
            printf("file.eof\n");
        }
        if(file.fail())
        {
            printf("file.fail\n");
        }
        if(file.bad())
        {
            printf("file.bad\n");
        }
        printf("tellg: %lld\n",file.tellg());
    }
	//sprawdzam nagłówek klatki
	file.seekg(-8,ios::cur);
	if(!file.good())
    {
        printf("file is not good(3)!\n");
        ios_base::iostate state=file.rdstate();
        printf("state: %d\n",state);
        if(file.eof())
        {
            printf("file.eof\n");
        }
        if(file.fail())
        {
            printf("file.fail\n");
        }
        if(file.bad())
        {
            printf("file.bad\n");
        }
        printf("tellg: %lld\n",file.tellg());
    }
	file.read(codeBuff,8);
	if(!file.good())
    {
        printf("file is not good(4)!\n");
        ios_base::iostate state=file.rdstate();
        printf("state: %d\n",state);
        if(file.eof())
        {
            printf("file.eof\n");
        }
        if(file.fail())
        {
            printf("file.fail\n");
        }
        if(file.bad())
        {
            printf("file.bad\n");
        }
        printf("tellg: %lld\n",file.tellg());
        file.clear();
        file.close();
        file.open(filename, ios::in|ios::binary);
        if(!file.is_open())
        {
            printf("file reopen failed\n");
        }
        pos-=8;
        printf("pos: %lld\n",pos);
        file.seekg(0,file.end);
        printf("tellg end: %lld\n",file.tellg());
        file.seekg(pos,file.beg);
        printf("tellg: %lld\n",file.tellg());
        file.read(codeBuff,8);
        if(!file.good())
        {
            printf("file is not good\n");
        }
        if(file.eof())
        {
            printf("file.eof\n");
        }
        if(file.fail())
        {
            printf("file.fail\n");
        }
        if(file.bad())
        {
            printf("file.bad\n");
        }
    }

	for(int i=0;i<8;i++)
    {
        /*
        if(smallf)
            b&=frameStartCodeS[i]==codeBuff[i];
        else
            b&=frameStartCode[i]==codeBuff[i];
        */
        b&=frameStartCodeS[i]==codeBuff[i]||frameStartCode[i]==codeBuff[i];
        printf("0x%02X ",codeBuff[i]);
    }
    struct JunkInfo
    {
        long long int size;
        long long int position;
        JunkInfo(long long int s, long long int p) :
            size(s), position(p) {}
        JunkInfo() :
            size(0), position(0) {}
        JunkInfo(JunkInfo& c) :
            size(c.size), position(c.position) {}
    };
    vector<JunkInfo> v;
    char junkCode[]="JUNK";
    int junkCt=0;
    int junkSize=0;
    bool junkB=true;
    if(!b)
    {
        printf("początek klatki: %hu nie znaleziony w przewidzianym miejscu\n",*numer);
        char* buff=new char[65535+8];
        file.seekg(8*65535,ios::cur);
        long long int ct=0,ct2=0;
        while(!b && file.good())
        {
            file.read(buff,65535);
            ct2++;
            for(int j=0;j<65535;j++)
            {
                ct++;
                junkB=true;
                for(int i=0;i<4 && junkB;i++)
                {
                    junkB&=junkCode[i]==buff[j+i];
                }
                if(junkB)
                {
                    /**< Chunks with fourcc 'JUNK' can appear anywhere and should be ignored. */
                    printf("znaleziono JUNK ct=%lld\n",ct);
                    v.push_back(JunkInfo(4+(*(int*)(buff+j+4)),ct));
                    junkSize+=(*(int*)(buff+j+4));
                    junkCt=ct;
                    junkSize+=4;
                    printf("JUNK size: %d\n",junkSize);
                }
                b=true;
                for(int i=0;i<8 && b;i++)
                {
                    /*
                    if(smallf)
                        b&=frameStartCodeS[i]==buff[j+i];
                    else
                        b&=frameStartCode[i]==buff[j+i];
                    */
                    b&=frameStartCodeS[i]==buff[j+i]||frameStartCode[i]==buff[j+i];
                }
                if(b)
                {
                    file.seekg(-(65535-j)-614400-junkSize,ios::cur);
                    printf("nagłówek klatki ct=%lld\n",ct);
                    printf("tellg: %lld\n",file.tellg());
                    for(int i=0;i<v.size();i++)
                    {
                        v.at(i).position=614400-(ct-v.at(i).position);
                    }
                    break;
                }
            }
            if(!b)
            printf("nie znaleziono początku klatki, przeszukuję następne 64KB\n");
        }
    }

	//czytanie klatki
	unsigned short int bl,bh;
	int j=0;
	for(int i=0;i<307200;i++)
    {
        while(j<(v.size()-1) && (i*2==v.at(j).position || i*2+1==v.at(j).position))
        {
            printf("j: %d, i: %d, v.at(j).size: %d, v.at(j).position: %d\n",j,i,v.at(j).size,v.at(j).position);
            file.seekg(v.at(j).size,ios::cur);
            j++;
        }
        bh=((unsigned short int)file.get())<<6;
        bl=reverse6bitLookupTable[file.get()>>2];
        klatka[i]=bh+bl;
    }
    file.close();
}
catch(exception &e)
{
    printf("wyjątek: %s\n",e.what());
}
catch(...)
{
    printf("nieznany wyjątek\n");
}
}
