/** \file FrameReader.cpp
 * \author Tomasz Jakubczyk
 * \brief Plik z implementacjami metod klasy FrameReader
 *
 */

#include "FrameReader.hpp"
#include <cstring>
#include <cstdio>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif // MATLAB_MEX_FILE

const char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
const char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
const char junkCode[]="JUNK";

FrameReader::FrameReader(CyclicBuffer* c) :
    dataSpace(nullptr),emptyLeft(true),emptyRight(true)
{
    dataSpace=new DataSpace(65535*10*2);
}

FrameReader::~FrameReader()
{
    delete dataSpace;
}

void FrameReader::loadLeft()
{
    buffId* tmpBuff=nullptr;
    tmpBuff=cyclicBuffer->claimForRead();
    memcpy(dataSpace->pt,tmpBuff->pt,dataSpace->halfSize);
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyLeft=false;
}

void FrameReader::loadRight()
{
    buffId* tmpBuff=nullptr;
    tmpBuff=cyclicBuffer->claimForRead();
    memcpy(dataSpace->ptRight,tmpBuff->pt,dataSpace->halfSize);
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyRight=false;
}

void FrameReader::cycleDataSpace()
{
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
        throw FrameReaderException("frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size");
    }
    frame.pt=dataSpace->pt+frame.position;
    if(header.position>=dataSpace->halfSize)
    {
        cycleDataSpace();
    }
    return frame.pt;
}

void FrameReader::findNextHeader()
{
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
    pt=new char[size];
    ptLeft=pt;
    ptRight=pt+size/2;

}

FrameReader::DataSpace::~DataSpace()
{
    delete[] pt;
}

FrameReader::Header::Header() :
    pt(nullptr), position(0), found(false), number(0), size(8)
{

}

FrameReader::Junk::Junk() :
    pt(nullptr), position(0), number(0), size(0), found(false)
{

}

FrameReader::Frame::Frame() :
    size(614400), pt(nullptr), position(0),found(false)
{

}





