/** \file FrameReader.cpp
 * \author Tomasz Jakubczyk
 * \brief Plik z implementacjami metod klasy FrameReader
 *
 */

#include "FrameReader.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif // MATLAB_MEX_FILE

/** \brief wypisanie tekstu do matlaba
 *
 * \param c const char*
 * \return void
 *
 */
void printm(const char* c)
{
    printf("%s\n",c);
    #ifdef MATLAB_MEX_FILE
    mexEvalString("pause(.001);");
    #endif // MATLAB_MEX_FILE
}

const char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
const char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
const char junkCode[]="JUNK";

FrameReader::FrameReader(CyclicBuffer* c) :
    cyclicBuffer(c),dataSpace(nullptr),emptyLeft(true),emptyRight(true)
{
    #ifdef DEBUG
    printm("FrameReader::FrameReader(CyclicBuffer* c)");
    #endif // DEBUG
    dataSpace=new DataSpace(65535*10*2);
}

FrameReader::~FrameReader()
{
    #ifdef DEBUG
    printm("FrameReader::~FrameReader()");
    #endif // DEBUG
    delete dataSpace;
}

void FrameReader::loadLeft()
{
    #ifdef DEBUG
    printm("void FrameReader::loadLeft()");
    #endif // DEBUG
    buffId* tmpBuff=nullptr;
    printStatus();
    if(cyclicBuffer==nullptr)
    {
        #ifdef DEBUG
        printm("cyclicBuffer==nullptr");
        #endif // DEBUG
        throw FrameReaderException("cyclicBuffer==nullptr");
    }
    tmpBuff=cyclicBuffer->claimForRead();
    if(tmpBuff==nullptr)
    {
        #ifdef DEBUG
        printm("tmpBuff==nullptr");
        #endif // DEBUG
        throw FrameReaderException("tmpBuff==nullptr");
    }
    #ifdef DEBUG
    printf("tmpBuff->frameNo: %d\n",tmpBuff->frameNo);
    printm("tmpBuff=cyclicBuffer->claimForRead()");
    #endif // DEBUG
    if(dataSpace->pt==nullptr)
    {
        #ifdef DEBUG
        printm("dataSpace->pt==nullptr");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->pt==nullptr");
    }
    if(tmpBuff->pt==nullptr)
    {
        #ifdef DEBUG
        printm("tmpBuff->pt==nullptr");
        #endif // DEBUG
        throw FrameReaderException("tmpBuff->pt==nullptr");
    }
    if(dataSpace->halfSize<=0)
    {
        #ifdef DEBUG
        printm("dataSpace->halfSize<=0");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->halfSize<=0");
    }
    if(dataSpace->halfSize>655350)
    {
        #ifdef DEBUG
        printm("dataSpace->halfSize>655350");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->halfSize>655350");
    }
    memcpy(dataSpace->pt,tmpBuff->pt,dataSpace->halfSize);
    #ifdef DEBUG
    printm("memcpy(dataSpace->pt,tmpBuff->pt,dataSpace->halfSize)");
    #endif // DEBUG
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyLeft=false;
}

void FrameReader::loadRight()
{
    #ifdef DEBUG2
    printm("void FrameReader::loadRight()");
    #endif // DEBUG
    buffId* tmpBuff=nullptr;
    tmpBuff=cyclicBuffer->claimForRead();
    if(dataSpace->ptRight==nullptr)
    {
        #ifdef DEBUG
        printm("dataSpace->ptRight==nullptr");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->ptRight==nullptr");
    }
    memcpy(dataSpace->ptRight,tmpBuff->pt,dataSpace->halfSize);
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyRight=false;
}

void FrameReader::cycleDataSpace()
{
    #ifdef DEBUG2
    printm("void FrameReader::cycleDataSpace()");
    #endif // DEBUG
    if(header.position>=dataSpace->halfSize)
    {
        header.position-=dataSpace->halfSize;
        header.pt-=dataSpace->halfSize;
        memcpy(dataSpace->ptLeft,dataSpace->ptRight,dataSpace->halfSize);
        loadRight();
    }
}

char* FrameReader::getFrame()
{
    #ifdef DEBUG2
    printm("char* FrameReader::getFrame()");
    #endif // DEBUG
    if(emptyRight && !emptyLeft)
    {
        loadRight();
    }
    if(emptyLeft)
    {
        loadLeft();
    }
    if(!header.found)
    {
        findNextHeader();
    }
    if(header.position<frame.size)
    {
        findNextHeader();
    }
    frame.position=(junk.found?junk.position:header.position)-frame.size;
    if(frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size)
    {
        #ifdef DEBUG
        printm("frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size");
        #endif // DEBUG
        throw FrameReaderException("frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size");
    }
    frame.pt=dataSpace->pt+frame.position;
    if(header.position>=dataSpace->halfSize)
    {
        cycleDataSpace();
    }
    printStatus();
    system("pause");
    return frame.pt;
}

void FrameReader::findNextHeader()
{
    #ifdef DEBUG2
    printm("void FrameReader::findNextHeader()");
    #endif // DEBUG
    bool headerB=true;
    bool junkB=true;
    junk.found=false;

    for(int j=header.found?(header.position+frame.size):header.position;
    j<(header.found?(dataSpace->size-header.size):dataSpace->halfSize-header.size);
    j++)
    {
        junkB=true;
        for(int i=0;i<junk.hSize;i++)
        {
            junkB&=junkCode[i]==dataSpace->pt[j+i];
            if(!junkB)
            {
                break;
            }
        }
        if(junkB)
        {
            junk.found=true;
            junk.number++;
            junk.position=j;
            junk.pt=dataSpace->pt+junk.position;
            junk.size=*(int*)(dataSpace->pt+j+4);
            junk.size+=4;
            j+=junk.size;
        }
        headerB=true;
        for(int i=0;i<header.size;i++)
        {
            headerB&=frameStartCode[i]==dataSpace->pt[j+i] || frameStartCodeS[i]==dataSpace->pt[j+i];
            if(!headerB)
            {
                break;
            }
        }
        if(headerB)
        {
            header.found=true;
            header.number++;
            header.position=j;
            header.pt=dataSpace->pt+header.position;
            break;
        }
    }
    if(!headerB)
    {
        header.found=false;
    }
}

FrameReader::DataSpace::DataSpace(unsigned long int s) :
    pt(nullptr), size(s), ptLeft(nullptr), ptRight(nullptr), halfSize(s/2)
{
    #ifdef DEBUG
    printm("FrameReader::DataSpace::DataSpace(unsigned long int s)");
    #endif // DEBUG
    pt=new char[size];
    ptLeft=pt;
    ptRight=pt+size/2;

}

FrameReader::DataSpace::~DataSpace()
{
    #ifdef DEBUG
    printm("FrameReader::DataSpace::~DataSpace()");
    #endif // DEBUG
    delete[] pt;
}

FrameReader::Header::Header() :
    pt(nullptr), position(0), found(false), number(0), size(8)
{
    #ifdef DEBUG
    printm("FrameReader::Header::Header()");
    #endif // DEBUG
}

FrameReader::Junk::Junk() :
    pt(nullptr), position(0), number(0), size(0), found(false)
{
    #ifdef DEBUG
    printm("FrameReader::Junk::Junk()");
    #endif // DEBUG
}

FrameReader::Frame::Frame() :
    size(614400), pt(nullptr), position(0),found(false)
{
    #ifdef DEBUG
    printm("FrameReader::Frame::Frame()");
    #endif // DEBUG
}

void FrameReader::printStatus()
{
    printm("void FrameReader::printStatus()");
    printf("cyclicBuffer: 0x%p\n",cyclicBuffer);
    printf("dataSpace: 0x%p\n",dataSpace);
    printf("dataSpace->pt: 0x%p\n",dataSpace->pt);
    printf("dataSpace->size: %lu\n",dataSpace->size);
    printf("dataSpace->ptLeft: 0x%p\n",dataSpace->ptLeft);
    printf("dataSpace->ptRight: 0x%p\n",dataSpace->ptRight);
    printf("dataSpace->halfSize: %lu\n",dataSpace->halfSize);
    printf("header.pt: 0x%p\n",header.pt);
    printf("header.position: %lu\n",header.position);
    printf(header.found?"header.found: true\n":"header.found: false\n");
    printf("header.number: %lu\n",header.number);
    printf("header.size: %lu\n",header.size);
    printf("junk.pt: 0x%p\n",junk.pt);
    printf("junk.position: %lu\n",junk.position);
    printf("junk.number: %lu\n",junk.number);
    printf("junk.size: %lu\n",junk.size);
    printf(junk.found?"junk.found: true\n":"junk.found: false\n");
    printf("junk.hSize: %lu\n",junk.hSize);
    printf("frame.size: %lu\n",frame.size);
    printf("frame.pt: 0x%p\n",frame.pt);
    printf("frame.position: %lu\n",frame.position);
    printf(frame.found?"frame.found: true\n":"frame.found: false\n");
    printf(emptyLeft?"emptyLeft: true\n":"emptyLeft: false\n");
    printf(emptyRight?"emptyRight: true\n":"emptyRight: false\n");
    printm("void FrameReader::printStatus() end");
}





