#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<string>

using namespace std;

int main(int argc, char* argv[])
{
    fstream file0("E:\\DEG_clean4.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean401.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean402.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean403.avi",ios::in|ios::binary);
    const int skok = (640*480*2)+8;
    char* buff=new char[640*480*2];
    int j=0;
    char c;
    long long int bindex=0;
    long long int index=0;
    long long int lindex=0;
    bool found=false;
    int k=401;
    char str[4];
    fstream file1;
    string name;
    while(file0.good() && k<=412)
    {
        name=string("E:\\DEG_clean")+string((const char*)itoa(k,str,10))+string(".avi");
        file1.open(name.c_str(),ios::in|ios::binary);
        if(file1.good())
        {
            printf("file opened: %s\n",name.c_str());
        }
        else
        {
            printf("cant't open file: %s\n",name.c_str());
        }
        file1.seekg(34824,ios::beg);
        while(file0.good() && file1.good())
        {
            //printf("search for frame: %d\n",j);
            file1.read(buff,640*480*2);
            if(!file1.good())
            {
                printf("problem reading file1 : %s\n",name.c_str());
                continue;
            }
            found=false;
            int licz=0;
            while(!found && file0.good() && file1.good() && licz<=640*480*2*3)
            {
                bindex=index;
                for(int i=0;i<640*480*2;i++)
                {
                    c=file0.get();
                    index++;
                    if(c==buff[i])
                    {
                        if(i+1==640*480*2)
                        {
                            printf("frame No. %d starts at: %lld difference: %lld\n", j, bindex,bindex-lindex);
                            lindex=bindex;
                            //return 0;
                            //file0.seekg(8,ios::cur);
                            found=true;
                            break;
                        }
                        continue;
                    }
                    else
                    {
                        licz++;
                        if(licz>640*480*2*3)
                        {
                            printf("skept 3 frames, try to find next frame\n");
                            file0.seekg(-640*480*2*3,ios::cur);
                            index-=640*480*2*3;
                            break;
                        }
                        if(c==buff[0])
                        {
                            printf("new potential begining of frame: %d at: %lld licz: %d\n",j,index,licz);
                            i=0;
                            bindex=index;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
            if(!found)
                printf("frame No. %d not found\n",j);
            file1.seekg(8,ios::cur);
            j++;
        }
        file1.close();
        k++;
        if(!file0.good())
            printf("file0 is not good\n");
        file0.seekg(-640*480*2*2,ios::cur);
        index-=640*480*2*2;
    }
    return 0;
}






