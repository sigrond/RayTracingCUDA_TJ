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
#include <vector>

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
unsigned long long int safetyCounter=0;

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
    #ifdef DEBUG
    printStatus();
    #endif // DEBUG
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
    if(header.position>=dataSpace->halfSize)
    {
        cycleDataSpace();
    }
    if(!header.found)/**< jeśli jeszcze nie został znaleziony nagłówek */
    {
        findNextHeader();
        #ifdef DEBUG
        printStatus();
        #endif // DEBUG
    }
    if(header.position-junk.size<frame.size)/**< jeśli przed nagłówkiem nie ma miejsca na klatkę to trzeba znaleźć następny nagłówek */
    {
        findNextHeader();
    }
    else if(header.position+frame.size+junk.size+header.size<(emptyRight?dataSpace->halfSize:dataSpace->size))
    {
        findNextHeader();
    }
    else if(!emptyRight)
    {
        printm("to nie powinno się wydażyć #001");
        printStatus();
        system("PAUSE");
        throw FrameReaderException("to nie powinno się wydażyć #001");
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
    frame.found=true;

    correctnessControl.addFrame(dataSpace,&header,&junk,&frame);

    #ifdef DEBUG
    if(safetyCounter++<=5)
    {
        printStatus();
        //system("pause");
    }
    #endif // DEBUG
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
    if(header.found)
    {
        header.position+=frame.size;/**< najwcześniejsza możliwa pozycja następnego nagłówka */
    }

    for(int j=header.found?(header.position+header.size):header.position;
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
            junk.size=*(int*)(dataSpace->pt+j+junk.hSize);
            junk.size+=junk.hSize;
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

FrameReader::DataSpace::DataSpace(const DataSpace& o) :
    pt(nullptr), size(o.size), ptLeft(nullptr), ptRight(nullptr), halfSize(o.halfSize)
{
    while(pt==nullptr)
    {
        pt=new (nothrow) char[size];
    }
    ptLeft=pt;
    ptRight=pt+size/2;
    memcpy(pt,o.pt,size);
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
    pt(nullptr), position(0), number(0), size(0), found(false), hSize(4)
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

FrameReader::CorrectnessControl::CorrectnessControl() :
    lastFrameCorrect(true), decodedFrame(nullptr)
{
    decodedFrame=new char[frame.size];
}

FrameReader::CorrectnessControl::~CorrectnessControl()
{
    if(!q.empty())
    {
        printm("kolejka korekcji poprawności nie jest pusta");
        throw FrameReaderException("kolejka korekcji poprawności nie jest pusta");
    }
    delete[] decodedFrame;
}

FrameReader::CorrectnessControl::FrameData::FrameData(DataSpace* d,Header* h,Junk* j,Frame* f) :
    dataSpacePt(nullptr),headerPt(nullptr),junkPt(nullptr),framePt(nullptr)
{
    dataSpacePt=new DataSpace(*d);
    headerPt=new Header(*h);
    junkPt=new Junk(*j);
    framePt=new Frame(*f);
}

FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)
{
    printm("FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)");
    throw FrameReaderException("FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)");
}

FrameReader::CorrectnessControl::FrameData::~FrameData()
{
    delete dataSpacePt;
    dataSpacePt=nullptr;
    delete headerPt;
    headerPt=nullptr;
    delete junkPt;
    junkPt=nullptr;
    delete framePt;
    framePt=nullptr;
}

void FrameReader::CorrectnessControl::addFrame(DataSpace* d,Header* h,Junk* j,Frame* f)
{
    FrameData* frameData=new FrameData(d,h,j,f);
    m.lock();
    q.push(frameData);
    m.unlock();
    empty.notify_one();
}

bool FrameReader::CorrectnessControl::checkFrame()
{
    unique_lock<mutex> lck(m);
    while(q.empty())
    {
        empty.wait(lck);
    }
    FrameData* frameData=q.front();
    lck.unlock();
    lck.release();
    bool headerB=true;
    bool junkB=true;
    for(int j=0;j<frameData->dataSpacePt->size;j++)
    {
        junkB=true;
        for(int i=0;i<frameData->junkPt->hSize;i++)
        {
            junkB&=junkCode[i]==frameData->dataSpacePt->pt[j+i];
        }
        if(junkB)
        {
            frameData->junkV.emplace_back();
            frameData->junkV.back().position=j;
            frameData->junkV.back().size=*(int*)(frameData->dataSpacePt->pt+j+4);
            frameData->junkV.back().size+=4;
            j+=frameData->junkV.back().size;
        }
        headerB=true;
        for(int i=0;i<frameData->headerPt->size)
        {
            headerB&=frameStartCode[i]==frameData->dataSpacePt->pt[j+i] ||
             frameStartCodeS[i]==frameData->dataSpacePt->pt[j+i];
        }
        if(headerB)
        {
            frameData->headerV.emplace_back();
            frameData->headerV.back().position=j;
            j+=frameData->headerV.back().size;
        }
    }
    for(int i=0;i<frameData->headerV.size();i++)
    {
        if(frameData->headerV.at(i).position==frameData->headerPt->position)
        {//nagłówek znaleziony na wcześniej ustalonej pozycji
            if(frameData->junkV.empty())
            {//nie znaleziono żadnego JUNK
                if(frameData->junkPt->found)
                {//było zaznaczone, że wcześniej znaleziono w danym segmncie JUNK
                    lastFrameCorrect=false;
                    return false;///dziwny przrypadek, raczej nie powinien się zdażyć
                }
                else
                {
                    if(frameData->headerV.at(i).position>=frameData->framePt->size)
                    {//przed nagłówkiem jest dość miejsca na klatkę
                        if(i==0)
                        {//jest to pierwszy nagłówek z kolei
                            m.lock();
                            q.pop();
                            m.unlock();
                            lastFrameCorrect=true;
                            return true;///nie ma JUNK, nie ma wcześniejszego nagłówka, a klatka w całości mieści się przed nagłówkiem
                        }
                        else if(frameData->headerV.at(i-1).position+frameData->headerV.at(i-1).size+frameData->framePt->size<=frameData->headerV.at(i).position)
                        {//między nagłówkami jest dość miejsca na klatkę
                            m.lock();
                            q.pop();
                            m.unlock();
                            lastFrameCorrect=true;
                            return true;///nie ma JUNK, a klatka w całości zmieśiła się mi,ędzy nagłówkami
                        }
                        else
                        {
                            lastFrameCorrect=false;
                            return false;///za mało miejsca na całą klatkę pomiędzy nagłówkami klatek, to było by dziwne
                        }
                    }
                    else
                    {
                        lastFrameCorrect=false;
                        return false;///urżnięta klatka, raczej nie powinno sie wydażyć
                    }
                }
            }
            else
            {
                for(int j=0;j<frameData->junkV.size();j++)
                {
                    if(frameData->junkV.at(j).position==frameData->junkPt->position)
                    {//zaneziono wykożystany JUNK
                        if(frameData->junkV.at(j).position+frameData->junkV.at(j).size==frameData->headerPt->position)
                        {//JUNK jest przyklejony do odpowiadającego nagłówka
                            if(j==0)
                            {//jest to pierwszy JUNK z kolei
                                if(frameData->junkV.at(j).position>=frameData->framePt->size)
                                {//przed JUNK jest dość miejsca na klatkę
                                    if(i==0)
                                    {//i jest to pierwszy nagłówek z kolei
                                        m.lock();
                                        q.pop();
                                        m.unlock();
                                        lastFrameCorrect=true;
                                        return true;///pierwszy nagłowek i pierwszy JUNK sklejone, a przed nimi jest dość miejsca na całą klatkę
                                    }
                                    else
                                    {//wcześniej był inny nagłówek
                                        if(frameData->junkV.at(j).position-(frameData->headerV.at(i).position+frameData->headerPt->size)>=frameData->framePt->size)
                                        {//między JUNK a header jest dość miejsca na klatkę
                                            m.lock();
                                            q.pop();
                                            m.unlock();
                                            lastFrameCorrect=true;
                                            return true;///między JUNK(przyklejonym do nagłówka), a poprzednim nagłówkiem jest dość miejsca na klatkę i nie ma tam dodatkowego JUNK
                                        }
                                        else
                                        {
                                            lastFrameCorrect=false;
                                            return false;///między JUNK, a poprzednim nagłówkiem jest za mało miejsca na całą klatkę, dziwny przypadek, raczej nie powinien się wydażyć
                                        }
                                    }
                                }
                                else
                                {
                                    lastFrameCorrect=false;
                                    return false;///urżnięta klatka, bo przed JUNK jest za mało miejsca, raczej nie powinno się wydażyć
                                }
                            }
                            else
                            {//nie pierwszy JUNK z kolei
                                if(i==0)
                                {//pierwszy nagłówek z kolei
                                    if(frameData->junkV.at(j-1).position+frameData->junkV.at(j-1).size+frameData->framePt->size<=frameData->junkV.at(j).position)
                                    {//między JUNK przyklejonym do nagłówka a poprzednim JUNK jest dość miejsca na klatkę
                                        m.lock();
                                        q.pop();
                                        m.unlock();
                                        lastFrameCorrect=true;
                                        return true;///oznaczało by to, że bezpośrednio za nagłówkiem, który się nie zmieścił w bloku danych był JUNK i było dość miejsca na klatkę do następnego JUNK przyklejonego do nagłówka
                                    }
                                    else
                                    {
                                        lastFrameCorrect=false;
                                        return false;///JUNK wystąpił w śrdoku klatki i wcześniejszy nagłówek nie zmieścił się w DataSpace
                                    }
                                }
                                else
                                {//nie pierwszy nagłówek z kolei
                                    if(frameData->junkV.at(j-1).position+frameData->junkV.at(j-1).size+frameData->framePt->size<=frameData->junkV.at(j).position)
                                    {
                                        m.lock();
                                        q.pop();
                                        m.unlock();
                                        lastFrameCorrect=true;
                                        return true;///przed JUNK przyklejonym do nagłówka było dość miejsca na klatkę
                                    }
                                    else
                                    {
                                        lastFrameCorrect=false;
                                        return false;///JUNK wystąpił w środku klatki
                                    }
                                }
                            }
                        }
                        else
                        {
                            lastFrameCorrect=false;
                            return false;///JUNK nie przyklejony do nagłówka klatki
                        }
                    }
                }
            }
        }
    }
    lastFrameCorrect=false;
    return false;///nie znalezion wcześniej znalezionego nagłówka, albo nie znaleziono wcześniej znalezionego JUNK, albo inny błąd
}

char* FrameReader::CorrectnessControl::decodeFrame()
{
    if(lastFrameCorrect)
    {
        printm("próbujemy zdekodować ponownie klatkę oznaczoną jako poprawnie zdekodowaną");
        throw FrameReaderException("próbujemy zdekodować ponownie klatkę oznaczoną jako poprawnie zdekodowaną");
    }
    m.lock();
    FrameData* frameData=q.front();
    m.unlock();
    if(frameData->headerV.empty())
    {
        m.lock();
        q.pop();
        m.unlock();
        printm("duża przestrzeń bez klatek?");
        return nullptr;///duża przestrzeń bez klatek?
    }
    if(frameData->junkV.empty())
    {
        if(frameData->headerPt->position>=frameData->framePt->size)
        {
            printm("klatka która powinna być dobrze zdekodowana, została oznaczona jako źle zdekodowana");
            memcpy(decodedFrame,frameData->dataSpacePt->pt,frameData->framePt->size);
            m.lock();
            q.pop();
            m.unlock();
            return decodedFrame;
        }
        else
        {
            printm("utrata części klatki?");
            memcpy(decodedFrame+(frameData->framePt->size-frameData->headerPt->position),frameData->dataSpacePt->pt,frameData->headerPt->position);
            m.lock();
            q.pop();
            m.unlock();
            return decodedFrame;
        }
    }
    int copiedFrame=0;/**< ile klatki zostało skopiowane */
    int lastSkipedPosition=frameData->headerPt->position;/**< ostatnio ominięty fragment */
    int sizeToCopy=0;
    for(int i=0;i<frameData->headerV.size();i++)
    {
        if(frameData->headerV.at(i).position==frameData->headerPt->position)
        {//jest przynajmniej jeden JUNK i header
            for(int j=frameData->junkV.size()-1;j>=0;j--)
            {
                if(frameData->junkV.at(j).position<frameData->headerPt->position)
                {//JUNK jest przed nagłówkiem
                    if(frameData->junkV.at(j).position+frameData->junkV.at(j).size>=frameData->headerPt->position)
                    {//JUNK przyklejony do nagłówka
                        lastSkipedPosition=frameData->junkV.at(j).position;
                    }
                    else
                    {
                        if(copiedFrame>=frameData->framePt->size)
                        {
                            m.lock();
                            q.pop();
                            m.unlock();
                            return decodedFrame;
                        }
                        sizeToCopy=(lastSkipedPosition-(frameData->junkV.at(j).position+frameData->junkV.at(j).size))<=(frameData->framePt->size-copiedFrame)?(lastSkipedPosition-(frameData->junkV.at(j).position+frameData->junkV.at(j).size)):(frameData->framePt->size-copiedFrame);
                        memcpy(decodedFrame+(frameData->framePt->size-copiedFrame),
                               frameData->dataSpacePt->pt+(frameData->junkV.at(j).position+frameData->junkV.at(j).size),
                               sizeToCopy);
                        lastSkipedPosition=frameData->junkV.at(j).position;
                        copiedFrame+=sizeToCopy;
                    }
                }
            }
        }
    }
    if(copiedFrame<frameData->framePt->size)
    {
        if(lastSkipedPosition-frameData->framePt->size+copiedFrame>=0)
        {//jest z kąd kopiować
            memcpy(decodedFrame,frameData->dataSpacePt->pt+(lastSkipedPosition-frameData->framePt->size+copiedFrame),frameData->framePt->size-copiedFrame);
        }
        else
        {
            memcpy(decodedFrame+(frameData->framePt->size-copiedFrame-lastSkipedPosition),rameData->dataSpacePt->pt,lastSkipedPosition);
        }
    }
    m.lock();
    q.pop();
    m.unlock();
    return decodedFrame;
}











