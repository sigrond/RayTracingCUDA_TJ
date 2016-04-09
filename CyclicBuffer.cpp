/** \file CyclicBuffer.cpp
 * \author Tomasz Jakubczyk
 * \brief plik z implementacjami metod klasy monitora z buforem cyklicznym
 *
 *
 *
 */

#include "CyclicBuffer.hpp"
#include <string>
#include <cstdio>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif // MATLAB_MEX_FILE

using namespace std;
using namespace ErrorCode;

/** \brief konstruktor
 */
CyclicBuffer::CyclicBuffer() :
    cBuffS(CBUFFS), cBeg(0), cEnd(0), itemCount(0)
{
    for(int i=0;i<cBuffS;i++)/**< alokowanie pamięci bufora */
    {
        cBuff[i]=new char[65535*10];
        frameNo[i]=-7;/**< niech oznacza to, że nigdy nie było użyte */
        buffReady[i]=true;/**< na początku żaden bufor nie jest używany */
    }
    for(int i=0;i<ERRNUM;i++)
    {
        errorCount[i]=0;
    }
}

/** \brief destruktor
 */
CyclicBuffer::~CyclicBuffer()
{
    unique_lock<mutex> lck(monitorMtx);
    #ifdef VERBOSE1
    printf("itemCount: %d cBeg: %d cEnd: %d\n",itemCount,cBeg,cEnd);
    #endif // VERBOSE1
    for(int i=0;i<cBuffS;i++)
    {
        #ifdef VERBOSE1
        printf("buffReady[%d]: %d frameNo[%d]: %d\n",i,buffReady[i],i,frameNo[i]);
        #endif // VERBOSE1
        delete[] cBuff[i];
        cBuff[i]=nullptr;
    }
    for(int i=0;i<ERRNUM;i++)
    {
        if(errorCount[i]!=0)
        printf("errorCount[%d]: %d\n",i,errorCount[i]);
    }
    lck.unlock();
}

/** \brief zajmij wskaźnik bufora do zapisu
 * \return char*
 */
buffId* CyclicBuffer::claimForWrite()
{
    unique_lock<mutex> lck(monitorMtx);
    #ifdef VERBOSE2
    printf("claimForWrite cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    while(itemCount==cBuffS || ((cEnd+1)%cBuffS==cBeg && itemCount!=0))
    {
        //printf("claimForWrite full cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        full.wait(lck);/**< czekamy jeśli bufor cykliczny jest pełny */
    }
    unsigned int tmpEnd=cEnd;
    if(itemCount>0)
    {
        tmpEnd=(cEnd+1)%cBuffS;
    }
    if(tmpEnd==cBeg && itemCount!=0)
    {
        errorCount[BufferOverflow]++;
        printf("błąd krytyczny, przepełnienie bufora");
        throw string("błąd krytyczny, przepełnienie bufora");
    }
    while(!buffReady[tmpEnd])
    {
        buffReadyCond[tmpEnd].wait(lck);/**< jeśli coś używa bufora to czekamy poza monitorem */
    }
    buffReady[tmpEnd]=false;/**< zaznaczamy, że bufor jest używany */
    lck.unlock();
    return new buffId(tmpEnd,cBuff[tmpEnd],frameNo[tmpEnd]);
}

/** \brief zwolnienie bufora po zapisaniu
 * \param id buffId*
 * \return void
 */
void CyclicBuffer::writeEnd(buffId* id)
{
    monitorMtx.lock();
    #ifdef VERBOSE2
    printf("writeEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    cEnd=id->id;
    itemCount++;
    frameNo[id->id]=id->frameNo;
    buffReady[id->id]=true;/**< zaznaczamy, że nie używamy już bufora */
    monitorMtx.unlock();/**< odblokowujemy monitor */
    buffReadyCond[id->id].notify_one();/**< powiadamiamy, że bufor jest nie używany */
    delete id;/**< zwalniamy wskaźnik strukturę */
    id=nullptr;
    empty.notify_one();/**< powiadamiamy, że bufor cykliczny nie jest już pusty */
}

/** \brief zajmij wskaźnik bufora do odczytu
 * \return buffId*
 */
buffId* CyclicBuffer::claimForRead()
{
    unique_lock<mutex> lck(monitorMtx);
    #ifdef VERBOSE2
    printf("claimForRead cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    while(itemCount==0)
    {
        if(cBeg!=cEnd && itemCount==0)
        {
            errorCount[cBegIsNotcEndAtitemCount0]++;
            printf("0 elementów, a cBeg!=cEnd");
        }
        #ifdef VERBOSE2
        printf("claimForRead empty cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        #endif // VERBOSE2
        empty.wait(lck);
    }
    if(cBeg==cEnd && itemCount>1 && itemCount!=cBuffS)
    {
        errorCount[ClaimForReadOfNotExistingElement]++;
        printf("próba czytania nieistniejącego elementu\n");
        printf("cBeg: %d cEnd: %d itemCount: %d\n               \n",cBeg,cEnd,itemCount);
        //throw string("próba czytania nieistniejącego elementu");
    }
    unsigned int tmpBeg=cBeg;
    while(!buffReady[tmpBeg])
    {
        #ifdef VERBOSE2
        printf("claimForRead buff not ready cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        #endif // VERBOSE2
        buffReadyCond[tmpBeg].wait(lck);/**< jeśli coś używa bufora to czekamy poza monitorem */
    }
    buffReady[tmpBeg]=false;/**< zaznaczamy, że bufor jest używany */
    lck.unlock();
    return new buffId(tmpBeg,cBuff[tmpBeg],frameNo[tmpBeg]);
}

/** \brief zwolnienie bufora po odczytaniu
 * \param id buffId*
 * \return void
 */
void CyclicBuffer::readEnd(buffId* id)
{
    unique_lock<mutex> lck(monitorMtx);
    #ifdef VERBOSE2
    printf("readEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    if(itemCount==0)
    {
        errorCount[ReadEndOfNotExistingElement]++;
        printf("właśnie przeczytaliśmy coś co nie istniało\n");
    }
    if(itemCount<0)
    {
        errorCount[ReadEndWithNegativeNumberOfElements]++;
        printf("próbujemy zakończyć czytanie z ujemną liczbą elementów\n");
    }
    if(itemCount==1 && buffReady[(id->id+1)%cBuffS]==true)
    {
        cBeg=(id->id+1)%cBuffS;
    }
    else if(itemCount==1 && buffReady[(id->id+1)%cBuffS]==false)
    {
        errorCount[ReadEndAndNextBuffIsNotBeingWriten]++;
        #ifdef VERBOSE1
        printf("wygląda na to, że kończymy czytać jedyny element w buforze, a następny się jak na razie nie zapisuje\n");
        printf("FrameNo: %d cBeg: %d cEnd: %d itemCount: %d         \n",id->frameNo,cBeg,cEnd,itemCount);
        #endif // VERBOSE1
        //cBeg=cEnd;
    }
    if(itemCount>1)
    {
        cBeg=(id->id+1)%cBuffS;
    }
    itemCount--;
    if(itemCount<0)
    {
        errorCount[ReadEndCousedNegativeNumberOfElements]++;
        //printf("błąd krytyczny, ujemna liczba elementów bufora");
        throw string("błąd krytyczny, ujemna liczba elementów bufora");
    }
    frameNo[id->id]=-2;
    buffReady[id->id]=true;/**< zaznaczamy, że nie używamy już bufora */
    lck.unlock();
    buffReadyCond[id->id].notify_one();/**< powiadamiamy, że bufor jest nie używany */
    delete id;
    id=nullptr;
    full.notify_one();
}


int CyclicBuffer::tellItemCount()
{
    return itemCount;
}

