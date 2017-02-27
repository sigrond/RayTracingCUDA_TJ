/* 
 * Ten plik MEX s³u¿y wy³¹cznie do odczytywania filmów zakodowanych kodekiem
 * PixelFLY. 
 * Sk³adnia:
 * MOVIEDATA = readavi(FILENAME,INDEX) reads from the AVI file FILENAME.  If
 * INDEX is -1, all frames in the movie are read. Otherwise, only frame number
 * INDEX is read.  
 * 
 */

#include <windows.h>
#include <vfw.h>
#include "mex.h"
/* Zaczyna siê czêœæ Matlaba */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{   
    //double *z1;
    int	i,
        j,
        start,
        end,
        frameCount,
        n,
        height=480,
        width=640,
        pad,
        paddedWidth=width*3, 
        formatSize;  
    
    char *filename;
    int dims[2];
    const char *fieldNames[1] = {"cdata"};/* Doda³em s³owo kluczowe const*/
    double *index;
    
    const void *pvdata;
    PAVIFILE pfile;
    PAVISTREAM vidStream;
    HRESULT hr;
    PBITMAPINFO bi;
    LPBITMAPINFOHEADER lpbi = NULL;
    PGETFRAME pframe[1];
    mxArray *mxframe;
    uint16_T *bit16frame;
    /* Sprawdzenie poprawnoœci argumentów */
   
if(!mxIsChar(prhs[0]))    
{       
    mexErrMsgIdAndTxt("MATLAB:Aviread12:firstinputinvalid","Nazwa pliku nie jest poprawna.");
}
/* Koniec sprawdzenia pierwszego argumentu*/
    
if(!mxIsNumeric(prhs[1]))
{
    mexErrMsgIdAndTxt("MATLAB:Aviread12:secondinputinvalid","Z³y zakres klatek. On musi byæ wektorem lub skalarem.");   
}
/* Koniec sprawdzenia drugiego argumentu*/


filename = mxArrayToString(prhs[0]);    
index = mxGetPr(prhs[1]);  
n = mxGetNumberOfElements(prhs[1]);/*D³ugoœæ wektora*/
  
/*Inicjalizacja procesu wideo*/   
AVIFileInit();    
/* Otwieranie pliku avi*/   
hr = AVIFileOpen(&pfile,filename, OF_READ, NULL);
/* Sprawdzenie czy mo¿na otwieraæ plik avi*/    
if(hr != AVIERR_OK)     
{          
    mexErrMsgIdAndTxt("MATLAB:Aviread12:fileopenfailed","Nie mo¿na otworzyæ pliku avi.");   
}
/* Koniec pierwszego sprawdzenia pliku avi*/   
mxFree(filename);
    
/* Nastêpne sprawdzenie*/   
hr = AVIFileGetStream(pfile, &vidStream, streamtypeVIDEO, 0);    
if(hr == AVIERR_NODATA)        
{        
    AVIFileRelease(pfile);         
    mexErrMsgIdAndTxt("MATLAB:Aviread12:novideostream","Nie mo¿na zlokalizowaæ strumienia wideo.");        
}  
else if (hr == AVIERR_MEMORY)     
{        
    AVIFileRelease(pfile);                
    mexErrMsgIdAndTxt("MATLAB:Aviread12:memerror","Wyczerpanie pamiêci.");      
}
/*Koniec drugiego sprawdzenia pliku avi*/ 
/* Nastêpne sprawdzenie*/  
hr = AVIStreamFormatSize(vidStream, 0, &formatSize);   
if(hr != AVIERR_OK)       
{       
    AVIStreamRelease(vidStream);       
    AVIFileRelease(pfile);       
    mexErrMsgIdAndTxt("MATLAB:m2:invalidstreamformat","Nie mo¿na odczytaæ strumienia wideo.");        
}
/*Koniec trzeciego sprawdzenia pliku avi*/    
bi = (PBITMAPINFO) mxMalloc(formatSize);

/*Sprawdzenie formatu strumienia wideo*/
hr = AVIStreamReadFormat(vidStream, 0, bi, &formatSize);   
if(hr != AVIERR_OK)        
{        
    AVIStreamRelease(vidStream);          
    AVIFileRelease(pfile);        
    mexErrMsgIdAndTxt("MATLAB:Aviread12:invalidstreamformat","Nie mo¿na odczytaæ formatu strumienia wideo.");     
}
/*Koniec sprawdzenia formatu strumienia wideo*/
bi->bmiHeader.biCompression = mmioFOURCC('N','o','n','e'); 
            
/*Sprawdzenie punktu startu strumienia wideo*/
start = AVIStreamStart(vidStream); 
if (start == -1)      
{         
    AVIStreamRelease(vidStream);
    AVIFileRelease(pfile);        
    mexErrMsgIdAndTxt("MATLAB:Aviread12:nostreamstart","Nie mo¿na zlokalizowaæ punktu startu strumienia wideo.");      
}
/*Koniec sprawdzenia punktu startu strumienia wideo*/
/*Sprawdzenie punktu konca strumienia wideo*/  
end = AVIStreamEnd(vidStream);    
if (end == -1)       
{          
    AVIStreamRelease(vidStream);          
    AVIFileRelease(pfile);      
    mexErrMsgIdAndTxt("MATLAB:Aviread12:nostreamend","Nie mo¿na zlokalizowaæ punktu startu strumienia wideo.");     
}
/*Koniec sprawdzenia punktu konca strumienia wideo*/   
if(index[0] == -1) /* Ca³y plik avi zosta³ odczytany*/    
{      
    plhs[0] = mxCreateStructMatrix(1,end-start,1,fieldNames);     
}  
else      /* Tylko czêœæ pliku avi zosta³a odczytana*/ 
{          
    plhs[0] = mxCreateStructMatrix(1,n,1,fieldNames);     
}
/*Uwaga tutaj*/     
dims[0] = 1;
dims[1] = paddedWidth*height;  
/*Sprawdzenie mo¿liwoœci otworzenia pojedynczej klatki*/		   
pframe[0] = AVIStreamGetFrameOpen(vidStream, NULL);/*NULL dla domyœlnego formatu*/ 
if (pframe[0] == NULL)       
{            
    AVIStreamRelease(vidStream);          
    AVIFileRelease(pfile);                       
    mexErrMsgIdAndTxt("MATLAB:Aviread12:invalidCodec","Nie mo¿na zlokalizowaæ dekodeka do dekompresowania strumiena wideo.");      
}
  
/*Koniec sprawdzenia mo¿liwoœci otworzenia pojedynczej klatki*/	
        
/* Czêœæ najwa¿niejsza*/ 

frameCount = 0;  
for(j=0;j<n;j++)      
{          
    if( index[0] != -1 )/* Jeœli tylko czêœæ pliku zosta³a odczytana*/             
    {             
        start = (double) index[j]-1;//index[j];                  
        end = (double) index[j];//index[j]+1;               
    }
                          
    for(i=start; i<end; i++)                               
    {                
       lpbi =AVIStreamGetFrame(pframe[0],i);
        /*Sprawdzenie danych z pojedynczej klatki*/
        
       if(lpbi == NULL)                                             
       {                                                             
           AVIStreamRelease(vidStream);                                 
           AVIFileRelease(pfile);                                     
           mexErrMsgIdAndTxt("MATLAB:Aviread12:noframedata","B³¹d uzyskania klatki wideo.");                                               
       }
         /*Koniec sprawdzenia danych z pojedynczej klatki*/                
 /*          
  A negative value for height means the bitmap is
                      
  stored top down.  Can't use negative number for
                     
  calculations.               
  */                                   
       
       //height = lpbi->biHeight > 0 ? lpbi->biHeight : -lpbi->biHeight;/*Jeœli biHeight dodatnia to bierze plus a w drugim przypadku bierze minus*/                                           
        
       //width  = lpbi->biWidth;
        
/*Sprawdzenia u¿ycia kodeka PixelFLY*/    
        
        if (lpbi->biBitCount!=48)                                                 
        {
                                                 
            AVIStreamGetFrameClose(pframe[0]);/* Zamkniêcie strumienia*/                         
            AVIStreamRelease(vidStream);                         
            AVIFileRelease(pfile);                        
            mexErrMsgIdAndTxt("MATLAB:Aviread12:invalidframetype","Ja przetwarzam tylko plik avi zakodowany kodekiem PixelFLY.");                      
                      
        }       
       pvdata=(char*)lpbi + lpbi->biSize;
        /*Koniec sprawdzenia u¿ycia kodeka PixelFLY*/   
        /*Skorygowanie rozmiaru */
        //pad = 4-(width*3)%4;                         
        //paddedWidth = width*3 + (pad==4 ? 0:pad);
        /*Koniec skorygowania rozmiaru */
        //dims[1] = paddedWidth*height;       
        mxframe = mxCreateNumericArray(2,dims,mxUINT16_CLASS,mxREAL); 
        bit16frame = mxGetData(mxframe); 
        
        /* Kopiowanie wszystkich wartoœci. Potem rozk³ada w RGB*/
        memcpy(bit16frame,pvdata,
                                   paddedWidth*height*sizeof(unsigned short));                                              
    }   
    mxSetField(plhs[0],frameCount,fieldNames[0],mxframe);                
    frameCount++;
}	     
mxFree(bi);    
AVIStreamGetFrameClose(pframe[0]);    
AVIStreamRelease(vidStream);    
AVIFileRelease(pfile);
/*Dodatkowe zmienne wyjœciowe*/
//plhs[1] = mxCreateDoubleMatrix(1,1, mxREAL);
//plhs[2] = mxCreateDoubleMatrix(1,1, mxREAL);
//z1 = mxGetPr(plhs[1]);
//z2 = mxGetPr(plhs[2]);
//*z1=(double)j  ;
//*z2=width;
}
