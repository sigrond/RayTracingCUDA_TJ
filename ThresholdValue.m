function [ alpha1, alpha2 ] = ThresholdValue( Frame )
%ThresholdValue znajdowanie wartoœci progowej miêdzy jasnym a ciemnym pixelem
%   szukamy wyszukiwaniem binarnym zadanej proporcji jasnych do ciemnych
proportion=80402/226798;%szukana proporcja jasnych do ciemnych
epsylon=0.001;%zadana dok³adnoœæ
x=0;%aktualna proporcja od 0 do 307200 (lub infty)
alpha1=0.5;%aktualny próg od 0 do 1
al=0;%próg z lewej
ar=1;%próg z prawej
%i=0;
%proportion
while (x > proportion + epsylon || x < proportion - epsylon)
    fge=(Frame(:,:,1)./max(max(Frame(:,:,1))))>alpha1;
    bs=sum(sum(fge));
    x=bs/(307200-bs);
    %x
    if x > proportion + epsylon
        al=alpha1;
        alpha1=mean([ar,alpha1]);
    elseif x < proportion - epsylon
        ar=alpha1;
        alpha1=mean([al,alpha1]);
    end
    %alpha1
    %i=i+1
end

epsylon=0.01;
x=0;%aktualna proporcja od 0 do 307200 (lub infty)
alpha2=0.5;%aktualny próg od 0 do 1
al=0;%próg z lewej
ar=1;%próg z prawej
while x > proportion + epsylon || x < proportion - epsylon
    fge=(Frame(:,:,3)./max(max(Frame(:,:,3))))>alpha2;
    bs=sum(sum(fge));
    x=bs/(307200-bs);
    if x > proportion + epsylon
        al=alpha2;
        alpha2=mean([ar,alpha2]);
    elseif x < proportion - epsylon
        ar=alpha2;
        alpha2=mean([al,alpha2]);
    end
end

end

