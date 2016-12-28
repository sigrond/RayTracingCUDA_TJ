/** \file CyclicBuffer.cpp
 * \author Tomasz Jakubczyk
 * \brief file with implementations of cyclic buffer methods
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

/** \brief constructor
 */
CyclicBuffer::CyclicBuffer() :
    cBuffS(CBUFFS), cBeg(0), cEnd(0), itemCount(0), averageLoad(0.0f), loadCount(0.0f)
{
    for(int i=0;i<cBuffS;i++)/**< alocate buffer memmory */
    {
        cBuff[i]=new char[65535*10];
        frameNo[i]=-7;/**< never used */
        buffReady[i]=1;/**< no buffer is used */
    }
    for(int i=0;i<ERRNUM;i++)
    {
        errorCount[i]=0;
    }
}

/** \brief destructor
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
        printf("errorCount[%d]: %d - %s\n",i,errorCount[i],ErrorNames[i]);
    }
    averageLoad/=loadCount;
    printf("averageLoad: %f\n",averageLoad);
    lck.unlock();
}

/** \brief claim buffer pointer for writing
 * \return char*
 */
buffId* CyclicBuffer::claimForWrite()
{
    unique_lock<mutex> lck(monitorMtx);
    loadCount++;
    averageLoad+=(float)itemCount;
    #ifdef VERBOSE2
    printf("claimForWrite cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    while(itemCount==cBuffS || ((cEnd+1)%cBuffS==cBeg && itemCount!=0))
    {
        //printf("claimForWrite full cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        full.wait(lck);/**< wait if cyclic buffer is full */
    }
    unsigned int tmpEnd=cEnd;
    if(itemCount>0)
    {
        tmpEnd=(cEnd+1)%cBuffS;
    }
    else if(itemCount==0 && cBeg==(cEnd+1)%cBuffS)
    {
        tmpEnd=(cEnd+1)%cBuffS;
    }
    if(tmpEnd==cBeg && itemCount!=0)
    {
        errorCount[BufferOverflow]++;
        printf("błąd krytyczny, przepełnienie bufora");//buffer overflow
        throw string("błąd krytyczny, przepełnienie bufora");
    }
    while(buffReady[tmpEnd]!=1)
    {
        buffReadyCond[tmpEnd].wait(lck);/**< if something uses buffer then wait outside the monitor */
    }
    buffReady[tmpEnd]=-1;/**< mark that buffer is used */
    lck.unlock();
    return new buffId(tmpEnd,cBuff[tmpEnd],frameNo[tmpEnd]);
}

/** \brief unclaim buffer after writing
 * \param id buffId*
 * \return void
 */
void CyclicBuffer::writeEnd(buffId* id)
{
    monitorMtx.lock();
    loadCount++;
    averageLoad+=(float)itemCount;
    #ifdef VERBOSE2
    printf("writeEnd cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    cEnd=id->id;
    itemCount++;
    frameNo[id->id]=id->frameNo;
    buffReady[id->id]=2;/**< mark that buffer isn't used */
    monitorMtx.unlock();/**< unlock monitor */
    buffReadyCond[id->id].notify_one();/**< notify thread that buffer isn't used */
    delete id;/**< free pointer to structure */
    id=nullptr;
    empty.notify_one();/**< notify thread that cyclic buffer isn't empty */
}

/** \brief claim buffer pointer for reading
 * \return buffId*
 */
buffId* CyclicBuffer::claimForRead()
{
    unique_lock<mutex> lck(monitorMtx);
    #ifdef VERBOSE2
    printf("claimForRead cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    #endif // VERBOSE2
    loadCount++;
    averageLoad+=(float)itemCount;
    while(itemCount==0)
    {
        if(!(cBeg==cEnd || cBeg==(cEnd+1)%cBuffS) && itemCount==0)
        {
            errorCount[cBegIsNotcEndAtitemCount0]++;
            printf("0 elementów, a cBeg!=cEnd\n");
            #ifdef VERBOSE3
            _printStatus();
            #endif // VERBOSE3
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
    while(buffReady[tmpBeg]!=2)
    {
        #ifdef VERBOSE2
        printf("claimForRead buff not ready cBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
        #endif // VERBOSE2
        buffReadyCond[tmpBeg].wait(lck);/**< if something uses buffer then wait outside the monitor */
    }
    buffReady[tmpBeg]=-2;/**< mark that buffer is used */
    lck.unlock();
    return new buffId(tmpBeg,cBuff[tmpBeg],frameNo[tmpBeg]);
}

/** \brief unclaim buffer after reading
 * \param id buffId*
 * \return void
 */
void CyclicBuffer::readEnd(buffId* id)
{
    unique_lock<mutex> lck(monitorMtx);
    loadCount++;
    averageLoad+=(float)itemCount;
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
    if(itemCount==1 && buffReady[(id->id+1)%cBuffS]==-1)
    {
        cBeg=(id->id+1)%cBuffS;
    }
    else if(itemCount==1 && buffReady[(id->id+1)%cBuffS]==2)
    {
        printf("mało prawdopodobny przypadek gdy jedyny element jest tym z którego czytania właśnie schodzimy a następny jest zaznaczony jako gotowy pełny\n");
        errorCount[7]++;
        cBeg=(id->id+1)%cBuffS;
    }
    else if(itemCount==1 && buffReady[(id->id+1)%cBuffS]==-2)
    {
        errorCount[8]++;
        cBeg=(id->id+1)%cBuffS;
    }
    else if(itemCount==1 && buffReady[(id->id+1)%cBuffS]!=-1)
    {
        errorCount[ReadEndAndNextBuffIsNotBeingWriten]++;
        #ifdef VERBOSE3
        _printStatus();
        #endif // VERBOSE3
        #ifdef VERBOSE1
        printf("wygląda na to, że kończymy czytać jedyny element w buforze, a następny się jak na razie nie zapisuje\n");
        printf("FrameNo: %d cBeg: %d cEnd: %d itemCount: %d         \n",id->frameNo,cBeg,cEnd,itemCount);
        #endif // VERBOSE1
        cBeg=cEnd;
    }
    else if(itemCount>1)
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
    buffReady[id->id]=1;/**< mark that buffer isn't used */
    lck.unlock();
    buffReadyCond[id->id].notify_one();/**< notify thread that buffer isn't used */
    delete id;
    id=nullptr;
    full.notify_one();
}


int CyclicBuffer::tellItemCount()
{
    return itemCount;
}

void CyclicBuffer::printStatus()
{
    monitorMtx.lock();
    printf("                  \ncBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    for(int i=0;i<cBuffS;i++)
    {
        printf("buffReady[%d]: %d ",i,buffReady[i]);
        printf("frameNo[%d]: %d\n",i,frameNo[i]);
    }
    monitorMtx.unlock();
}

void CyclicBuffer::_printStatus()
{
    printf("                  \ncBeg: %d cEnd: %d itemCount: %d\n",cBeg,cEnd,itemCount);
    for(int i=0;i<cBuffS;i++)
    {
        printf("buffReady[%d]: %d ",i,buffReady[i]);
        printf("frameNo[%d]: %d\n",i,frameNo[i]);
    }
}

