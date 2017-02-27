function Args = mySteepestDescent( Pk,PCCD,ha,hf,hp,hpb,t1,OptTime,pointsr,pointsb )
%mySteepestDescent metoda najszybszego spadku. d³ugoœæ kroku sta³a,
%kierunek - oœ z najwiêkszym spadkiem, spadek/gradient - dla tego samego
%sta³ego kroku
%   ta naiwna i brutalna implemantacja ma na celu dzia³aæ szybciej ni¿
%   generyczne metody i ignorowaæ p³ytkie minima lokalne

Args=[Pk,PCCD];
dx=0.1;%przy takim wyliczniu "gradientu" strona ma znaczenie tj poczodna lewostronna nie musi siê równaæ prawostronnej
M=1;
cumM=0;
prevI=0;

while M>1e-6 && toc(t1)<OptTime
    g=zeros(6,2);
    for i=1:6
        tmpA=Args;
        tmpA(i)=Args(i)+dx;
        g(i,1)=MeanSquaredDistance(pointsr,pointsb,Args)-MeanSquaredDistance(pointsr,pointsb,tmpA);
        tmpA(i)=Args(i)-dx;
        g(i,2)=MeanSquaredDistance(pointsr,pointsb,Args)-MeanSquaredDistance(pointsr,pointsb,tmpA);
    end
    [M,I]=max(g(:));
    if M>0
        [I_row, I_col] = ind2sub(size(g),I);
        for i=1:6
            if i==I_row
                if I_col==1
                    Args(i)=Args(i)+dx;
                else
                    Args(i)=Args(i)-dx;
                end
                break;
            end
        end
    end
    
    cumM=cumM+M;
    if prevI~=I || cumM>0.1
        r=658;
        g=532;
        b=458;
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
        delete(hp);
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
        drawnow
    end
    prevI=I;
    if cumM>1
        cumM=0;
    end
    
end

end

