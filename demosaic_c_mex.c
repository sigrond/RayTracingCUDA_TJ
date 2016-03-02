#include "mex.h"
//#include <stdlib.h>
#include <string.h>
//#include <stdio>
#define I inArray

typedef enum {RGGB, GRBG, GBRG, BGGR, INVALID} Pattern;
Pattern get_pattern(char* patt)
{

    if(strcmp(patt,"grbg")==0)
    {
        return GRBG;
    }
    return INVALID;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned short*inArray;
    inArray=mxGetData(prhs[0]);
    const int* wid_len=mxGetDimensions(prhs[0]);
    int wid=wid_len[0], len=wid_len[1];
    //printf("%d %d\n", wid, len);
    int x_max=wid-1, y_max=len-1;
    int dims[3]={wid, len, 3};// 480,640
    plhs[0]=mxCreateNumericArray(3,dims,mxINT16_CLASS,mxREAL);
    short* outArray;
    outArray=mxGetData(plhs[0]);
    char* patt=mxArrayToString(prhs[1]);
    Pattern pattern=get_pattern(patt);
    int im1=0,ip1=0,jm1=0,jp1=0;
    int lenxwid=len*wid;
    switch(pattern)
    {
    case GRBG:
        for(int j=0;j<len;j++)
        {
            jm1=j==0?j+1:j-1;//j-1
            jp1=j==y_max?j-1:j+1;//j+1
            for(int i=0;i<wid;i++)
            {
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
            }
        }

        for(int j=0;j<len;j++)
        {
            jm1=j==0?j+1:j-1;//j-1
            jp1=j==y_max?j-1:j+1;//j+1
            for(int i=0;i<wid;i++)
            {
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
            }
        }

        for(int j=0;j<len;j++)
        {
            jm1=j==0?j+1:j-1;//j-1
            jp1=j==y_max?j-1:j+1;//j+1
            for(int i=0;i<wid;i++)
            {
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
        }
        break;
    case RGGB:
        break;
    case GBRG:
        break;
    case BGGR:
        break;
    case INVALID:
        //printf("invalid\n");
        break;
    }
}

















