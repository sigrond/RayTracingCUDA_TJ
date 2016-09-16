/** \file FrameReader.cpp
 * \author Tomasz Jakubczyk
 * \brief Plik z implementacjami metod klasy FrameReader
 *
 */

#include "FrameReader.hpp"
#include <cstring>

FrameReader::FrameReader(CyclicBuffer* c) :
    dataSpace(nullptr),junkPt(nullptr),
    emptyLeft(true),emptyRight(true)
{
    dataSpace=new DataSpace(65535*10*2);
}

FrameReader::~FrameReader()
{
    delete dataSpace;
}

char* FrameReader::getFrame()
{
    char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
	char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
    buffId* tmpBuff=nullptr;
    if(emptyLeft)
    {
        tmpBuff=cyclicBuffer->claimForRead();
        memcpy(dataSpace->pt,tmpBuff->pt,65535*10);
        cyclicBuffer->readEnd(tmpBuff);
        tmpBuff=nullptr;
        emptyLeft=false;
    }
    if(!header.found)
    {
        int i=0;
        while(i<dataSpace->size+header.size)
    }
}

void FrameReader::findNextHeader()
{
    if(!header.found)
    {

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
