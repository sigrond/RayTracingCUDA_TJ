#include<cstdio>
#include<cstdlib>
#include<fstream>

using namespace std;

int main(int argc, char* argv[])
{
    fstream file0("E:\\DEG_clean4.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean401.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean402.avi",ios::in|ios::binary);
    //fstream file1("E:\\DEG_clean403.avi",ios::in|ios::binary);
    fstream file1("E:\\DEG_clean404.avi",ios::in|ios::binary);
    const int skok = (640*480*2)+8;
    char* buff=new char[640*480*2];
    int i=0;
    file1.seekg((34824+(skok*(i))),ios::beg);
    file1.read(buff,640*480*2);
    char c;
    long long int bindex=0;
    long long int index=0;
    while(!file0.eof())
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
                    printf("frame No. 1 starts at: %lld\n", bindex);
                    return 0;
                }
                continue;
            }
            else
            {
                if(c==buff[0])
                {
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
    printf("frame No. 1 not found\n");
    return 0;
}
