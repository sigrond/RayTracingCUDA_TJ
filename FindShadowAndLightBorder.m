function [ point ] = FindShadowAndLightBorder( line )
%FindShadowAndLightBorder Summary of this function goes here
%   Detailed explanation goes here
%dl=grad(line);

%znajdowanie extremów
k=0;
extremum=zeros(80,3);
for i=2:length(line)-1
    if line(i-1)>line(i) && line(i+1)>line(i) %minimum
        k=k+1;
        extremum(k,1)=line(i);
        extremum(k,2)=-1;
        extremum(k,3)=i;
    elseif line(i-1)<line(i) && line(i+1)<line(i) %maximum
        k=k+1;
        extremum(k,1)=line(i);
        extremum(k,2)=1;
        extremum(k,3)=i;
    end
end

%figure
%pl=plot(line(:));
%hold on;
%plot(extremum(1:k,3),extremum(1:k,1),'LineStyle','none','Marker','x','MarkerEdgeColor','r');

%model = @(v)arrayfun(@(x) integral(@(t) sin(t^2), 0, x),v); 
%x = lsqcurvefit(model,0,1:length(line),line)


%znajdowanie najwy¿szej "amplitudy"
maxDifference=0;
minDiffInd=0;
maxDiffInd=0;
firstDifMin=0;
for i=2:k
    difference=abs(extremum(i-1,1)-extremum(i,1));
    if difference>maxDifference
        maxDifference=difference;
        if extremum(i,2)==1
            maxDiffInd=extremum(i,3);
            minDiffInd=extremum(i-1,3);
            firstDifMin=i+1;
        else
            maxDiffInd=extremum(i-1,3);
            minDiffInd=extremum(i,3);
            firstDifMin=i-2;
        end
    end
end

%P=(a+b)/2
%2*P=a+b
%2*P-b=a

%plot(maxDiffInd,line(maxDiffInd),'LineStyle','none','Marker','o','MarkerEdgeColor','m');
%plot(minDiffInd,line(minDiffInd),'LineStyle','none','Marker','o','MarkerEdgeColor','c');

%firstDifMin
%maxDiffInd+(maxDiffInd-minDiffInd);

%point=2*maxDiffInd-minDiffInd;
if firstDifMin>length(extremum)
    firstDifMin=length(extremum);
elseif firstDifMin<1
    firstDifMin=1;
end
point=2*maxDiffInd-extremum(firstDifMin,3);
if point>length(line)
    point=length(line);
elseif point<1
    point=1;
end
%plot(point,line(point),'LineStyle','none','Marker','+','MarkerEdgeColor','b');

end

