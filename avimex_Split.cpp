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
 *
 */



#include "mex.h"
#include<fstream>

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
	char *filename;
	int *numer;
	numer=(int*)mxGetPr(prhs[1]);
	filename = mxArrayToString(prhs[0]);
	ifstream file (filename, ios::in|ios::binary);
	unsigned short int *klatka;
	int skok = (640*480*2)+8;
	plhs[0]=mxCreateNumericMatrix(640,480,mxUINT16_CLASS,mxREAL);
	klatka=(unsigned short int*) mxGetPr(plhs[0]);
	file.seekg((34824+(skok*(*numer))),ios::beg);
	unsigned short int bl,bh;
	for(int i=0;i<307200;i++)
    {
        bh=((unsigned short int)file.get())<<6;
        bl=odwroc6(((unsigned short int)file.get())>>2);
        klatka[i]=bh+bl;
    }
}
