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
}

/** \brief destruktor
 */
CyclicBuffer::~CyclicBuffer()
{
    unique_lock<mutex> lck(monitorMtx);
    printf("itemCount: %d cBeg: %d cEnd: %d\n",itemCount,cBeg,cEnd);
    for(int i=0;i<cBuffS;i++)
    {
        printf("buffReady[%d]: %d frameNo[%d]: %d\n",i,buffReady[i],i,frameNo[i]);
        delete[] cBuff[i];
        cBuff[i]=nullptr;
    }
    lck.unlock();
}

/** \brief zajmij wskaźnik bufora do zapisu
 * \return char*
 */
buffId* CyclicBuffer::claimForWrite()
{
    unique_lock<mutex> lck(monitorMtx);
    //printf("claimForWrite cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    while(itemCount==cBuffS || ((cEnd+1)%cBuffS && itemCount!=0))
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
        //printf("błąd krytyczny, przepełnienie bufora");
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
    //printf("writeEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
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
    //printf("claimForRead cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    while(itemCount==0)
    {
        //printf("claimForRead empty cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        empty.wait(lck);
    }
    /*if(cBeg==cEnd && itemCount!=cBuffS)
    {
        printf("próba czytania nieistniejącego elementu");
        throw string("próba czytania nieistniejącego elementu");
    }*/
    unsigned int tmpBeg=cBeg;
    while(!buffReady[tmpBeg])
    {
        //printf("claimForRead buff not ready cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
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
    monitorMtx.lock();
    //printf("readEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    if(itemCount-1>0)
    {
        cBeg=(id->id+1)%cBuffS;
    }
    itemCount--;
    if(itemCount<0)
    {
        //printf("błąd krytyczny, ujemna liczba elementów bufora");
        throw string("błąd krytyczny, ujemna liczba elementów bufora");
    }
    frameNo[id->id]=-2;
    buffReady[id->id]=true;/**< zaznaczamy, że nie używamy już bufora */
    monitorMtx.unlock();
    buffReadyCond[id->id].notify_one();/**< powiadamiamy, że bufor jest nie używany */
    delete id;
    id=nullptr;
    full.notify_one();
}
