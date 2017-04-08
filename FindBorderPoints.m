function [ pointsr, pointsb ] = FindBorderPoints( Frame, Args )
%FindBorderPoints znajdowanie brzegu obrazu rozproszeniowego zgodnie z
%teori¹ na podstawie przybli¿onych parametrów ramki
%   Detailed explanation goes here
r=658;
g=532;
b=458;

try
    lambdas=evalin('base', 'lambdas');
catch
    lambdas=[r,g,b];
end
r=lambdas(1);
g=lambdas(2);
b=lambdas(3);

Px=Args(1);
Py=Args(2);
Pz=Args(3);
ShX=Args(4);
ShY=Args(5);
lCCD=Args(6);

k=0;

[X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,r);

v=zeros(80,2);
line=zeros(80,1);

load('BR_settings.mat','SPointsR','SPointsB','FitFresnel','DisplayedWindows');
if exist('SPointsR','var');
    selectedPointsR=SPointsR;
else
    selectedPointsR=[3:2:10 15:8:40 41:2:50 51:8:80];
end
if exist('SPointsB','var');
    selectedPointsB=SPointsB;
else
    selectedPointsB=[3:2:10 15:8:40 41:2:50 51:8:80];
end
if ~exist('FitFresnel','var');
    FitFresnel=0;
end
if ~exist('DisplayedWindows','var')
    DisplayedWindows.BrightnesWindow=1;
    DisplayedWindows.SPointsWindow=1;
    DisplayedWindows.OptimInfo=1;
    DisplayedWindows.SimAnealingWindow=1;
    DisplayedWindows.FresnelFitPlots=1;
    DisplayedWindows.FinalOptWindow=1;
end
if DisplayedWindows.OptimInfo
    showDisplay='iter';
    showDiagnostics='on';
else
    showDisplay='off';
    showDiagnostics='off';
end

for i=selectedPointsR%2:4:80%wybrane indeksy punktów na ramce w pobli¿u których szukamy brzegu
    j=0;
    d=30;%odleg³oœæ od zadanego punktu do brzegu otoczenia
    a1=(Y(i-1)-Y(i+1))/(X(i-1)-X(i+1));%wpó³czynnik kierunkowy stycznej do ramki w punkcie
    if atand(a1)>0
        a=tand(atand(a1)-90);%wsp kierunkowy prostej prostopad³ej do stycznej do ramki w punkcie
    else
        a=tand(atand(a1)+90);%wsp kierunkowy prostej prostopad³ej do stycznej do ramki w punkcie
    end
    b0=Y(i)-a*X(i);%prosta na której szukamy wartoœci jest w postaci y=ax+b
    b=b0;
    %y=a*X(i)+b0
    %Y(i)
    %b=b0-Y(i);%przesuniecie uk³adu wsp
    %do znalezienia pixeli na odcinku pos³u¿y algorytm Bresenhama
    %delta=(2*a*b)^2-4*(1+a^2)*(b^2-d^2);
    delta=(((a^2)+2*a*b-2*X(i)-2*Y(i)*a)^2)-4*(1+(a^2))*((b^2)+(X(i)^2)+(Y(i)^2)-2*Y(i)*b-(d^2));
    if delta>0
        %x1=(-(2*a*b)-(delta)^0.5)/(2*(1+a^2))+X(i);
        %y1=a*x1+b0;
        %x2=(-(2*a*b)+(delta)^0.5)/(2*(1+a^2))+X(i);
        %y2=a*x2+b0;
        x1=(-(a^2+2*a*b-2*X(i)-2*Y(i)*a)-delta^0.5)/(2*(1+a^2));
        y1=a*x1+b;
        x2=2*X(i)-x1;
        y2=2*Y(i)-y1;
    else
        %b=b0-X(i)*a;
        %delta=(-2*b/a^2)^2-4*(1+1/a^2)*(b^2/a^2-d^2);
        %y1=(2*b/a^2-(delta)^0.5)/(2*(1+1/a^2))+Y(i);
        %x1=(y1-b0)/a;
        %y2=(2*b/a^2+(delta)^0.5)/(2*(1+1/a^2))+Y(i);
        %x2=(y2-b0)/a;
        delta=((-2*b)/(a^2)-2*Y(i))^2-4*(1+1/a^2)*((b^2)/(a^2)+Y(i)^2-d^2);
        if delta>0
            y1=(-((-2*b)/(a^2)-2*Y(i))-delta^0.5)/(2*(1+(1/(a^2))));
            x1=(y1-b)/a;
            x2=2*X(i)-x1;
            y2=2*Y(i)-y1;
        else
            x1=(Y(i)+d-b)/a;
            y1=a*x1+b;
            x2=2*X(i)-x1;
            y2=2*Y(i)-y1;
        end
    end
    %y=a*x1+b0
    %y1
    %y=a*x2+b0
    %y2
    %((X(i)-x1)^2+(Y(i)-y1)^2)^0.5
    %((X(i)-x2)^2+(Y(i)-y2)^2)^0.5
    
    %x1=int32(round(x1));
    %y1=int32(round(y1));
    %x2=int32(round(x2));
    %y2=int32(round(y2));
    x=int32(round(x1));
    y=int32(round(y1));
    % ustalenie kierunku rysowania

    if x1<x2
        xi=int32(1);
        dx=int32(x2-x1);
    else
        xi=int32(-1);
        dx=int32(x1-x2);
    end
    % ustalenie kierunku rysowania
    if y1<y2
        yi=int32(1);
        dy=int32(y2-y1);
    else
        yi=int32(-1);
        dy=int32(y1-y2);
    end
    % pierwszy piksel
    j=j+1;
    v(j,1)=x;%+X(i);
    v(j,2)=y;%+Y(i);
    if x<1
        v(j,1)=1;
    end
    if x>640
        v(j,1)=640;
    end
    if y<1
        v(j,2)=1;
    end
    if y>480
        v(j,2)=480;
    end
    %j
    line(j)=Frame(v(j,2),v(j,1),1);
    % oœ wiod¹ca OX
    if dx>dy
        ai=(dy-dx)*2;
        bi=dy*2;
        d=bi-dx;
        % pêtla po kolejnych x
        while x~=int32(round(x2))
            % test wspó³czynnika
            if d>=0
                x=x+xi;
                y=y+yi;
                d=d+ai;
            else
                d=d+bi;
                x=x+xi;
            end
            j=j+1;
            v(j,1)=x;%+X(i);
            v(j,2)=y;%+Y(i);
            if x<1
                v(j,1)=1;
            end
            if x>640
                v(j,1)=640;
            end
            if y<1
                v(j,2)=1;
            end
            if y>480
                v(j,2)=480;
            end
            line(j)=Frame(v(j,2),v(j,1),1);
        end
    % oœ wiod¹ca OY
    else
        ai=(dx-dy)*2;
        bi=dx*2;
        d=bi-dy;
        % pêtla po kolejnych y
        while y~=int32(round(y2))
            % test wspó³czynnika
            if d>=0
                x=x+xi;
                y=y+yi;
                d=d+ai;
            else
                d=d+bi;
                y=y+yi;
            end
            j=j+1;
            v(j,1)=x;%+X(i);
            v(j,2)=y;%+Y(i);
            if x<1
                v(j,1)=1;
            end
            if x>640
                v(j,1)=640;
            end
            if y<1
                v(j,2)=1;
            end
            if y>480
                v(j,2)=480;
            end
            line(j)=Frame(v(j,2),v(j,1),1);
        end
    end
    if exist('pl','var')
        %plot(line(1:j));
    else
        %pl=plot(line(1:j));
        %hold on;
    end
    %max(line)
    %hf = imtool( Frame./(max(max(max(Frame)))/20) );
    %ha = get(hf,'CurrentAxes');
    %hold(ha,'on');
    %hp=plot(ha,X,Y,'-xr');
    %hs=scatter(ha,v(1:j,1),v(1:j,2),'filled','MarkerFaceColor','r');
    %hs=scatter(ha,X(i),Y(i),'filled','MarkerFaceColor','b');
    %hs=scatter(ha,x1,y1,'filled','MarkerFaceColor','m');
    %hs=scatter(ha,x2,y2,'filled','MarkerFaceColor','c');
    quality=max(line(1:j))-min(line(1:j));
    k=k+1;
    c='r';
    %point=FindShadowAndLightBorder(line(1:j));
    %dist=BorderDistance(X,Y,v(point,1),v(point,2));
    %quality=-dist;
    if FitFresnel
        %point=FindShadowAndLightBorder(line(1:j));
        %dist=BorderDistance(X,Y,v(point,1),v(point,2));
        %quality=-dist;
        %x(1) - skala wartoœci
        %x(2) - przesuniêcie argumentów
        %x(3) - gêstoœæ argumentów
        %x(4) - przesuniêcie wartoœci
        xdata=1:j;
        ydata=line(1:j)-min(line(1:j));
        ydata=ydata./max(ydata);
        %fun=@(x,xdata)x(1)*fresnelc((xdata-x(2))/x(3))-x(4);
        fun=@(x,xdata)x(1)*myDiffractionFunction((xdata-x(2))/x(3))-x(4);
        x0=[2 25 8 0];
        lb=[0 1 -16 -1];
        ub=[2 j 16 1];
        [x,resnorm,residual,exitflag,output] = lsqcurvefit(fun,x0,xdata,ydata',lb,ub,optimoptions('lsqcurvefit','Diagnostics',showDiagnostics,'Display',showDisplay,'ScaleProblem','jacobian','TolFun',1e-16));
        meanx=mean(ydata);
        %relativeError=sqrt(resnorm)/meanx*100;
        relativeError=sqrt(norm(residual,Inf))/(meanx)*100;
        if DisplayedWindows.FresnelFitPlots
            figure('Name',sprintf('Fresnel fit: %d, b³¹d wzglêdny: %f%% residuum norm: %e, x: %e %f %e %e',i,relativeError,sqrt(resnorm),x(1),x(2),x(3),x(4)))
            plot(xdata,ydata')
            hold on
            mp=plot(xdata,fun(x,xdata));
            hold off;
        end
        point=FindShadowAndLightBorder(line(1:j));
        quality=1/abs(point-x(2));
    end
    data(k)=struct('v',v,'j',j,'X',X,'Y',Y,'line',line,'quality',quality,'color',c,'inColorIndex',i);
    %waitfor(hf);
end

r=658;
g=532;
b=458;

try
    lambdas=evalin('base', 'lambdas');
catch
    lambdas=[r,g,b];
end
r=lambdas(1);
g=lambdas(2);
b=lambdas(3);

[X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,b);

v=zeros(80,2);
line=zeros(80,1);

for i=selectedPointsB%2:4:80%wybrane indeksy punktów na ramce w pobli¿u których szukamy brzegu
    j=0;
    d=30;%odleg³oœæ od zadanego punktu do brzegu otoczenia
    a1=(Y(i-1)-Y(i+1))/(X(i-1)-X(i+1));%wpó³czynnik kierunkowy stycznej do ramki w punkcie
    if atand(a1)>0
        a=tand(atand(a1)-90);%wsp kierunkowy prostej prostopad³ej do stycznej do ramki w punkcie
    else
        a=tand(atand(a1)+90);%wsp kierunkowy prostej prostopad³ej do stycznej do ramki w punkcie
    end
    b0=Y(i)-a*X(i);%prosta na której szukamy wartoœci jest w postaci y=ax+b
    b=b0;
    %y=a*X(i)+b0
    %Y(i)
    %b=b0-Y(i);%przesuniecie uk³adu wsp
    %do znalezienia pixeli na odcinku pos³u¿y algorytm Bresenhama
    %delta=(2*a*b)^2-4*(1+a^2)*(b^2-d^2);
    delta=(((a^2)+2*a*b-2*X(i)-2*Y(i)*a)^2)-4*(1+(a^2))*((b^2)+(X(i)^2)+(Y(i)^2)-2*Y(i)*b-(d^2));
    if delta>0
        %x1=(-(2*a*b)-(delta)^0.5)/(2*(1+a^2))+X(i);
        %y1=a*x1+b0;
        %x2=(-(2*a*b)+(delta)^0.5)/(2*(1+a^2))+X(i);
        %y2=a*x2+b0;
        x1=(-(a^2+2*a*b-2*X(i)-2*Y(i)*a)-delta^0.5)/(2*(1+a^2));
        y1=a*x1+b;
        x2=2*X(i)-x1;
        y2=2*Y(i)-y1;
    else
        %b=b0-X(i)*a;
        %delta=(-2*b/a^2)^2-4*(1+1/a^2)*(b^2/a^2-d^2);
        %y1=(2*b/a^2-(delta)^0.5)/(2*(1+1/a^2))+Y(i);
        %x1=(y1-b0)/a;
        %y2=(2*b/a^2+(delta)^0.5)/(2*(1+1/a^2))+Y(i);
        %x2=(y2-b0)/a;
        delta=((-2*b)/(a^2)-2*Y(i))^2-4*(1+1/a^2)*((b^2)/(a^2)+Y(i)^2-d^2);
        if delta>0
            y1=(-((-2*b)/(a^2)-2*Y(i))-delta^0.5)/(2*(1+(1/(a^2))));
            x1=(y1-b)/a;
            x2=2*X(i)-x1;
            y2=2*Y(i)-y1;
        else
            x1=(Y(i)+d-b)/a;
            y1=a*x1+b;
            x2=2*X(i)-x1;
            y2=2*Y(i)-y1;
        end
    end
    %y=a*x1+b0
    %y1
    %y=a*x2+b0
    %y2
    %((X(i)-x1)^2+(Y(i)-y1)^2)^0.5
    %((X(i)-x2)^2+(Y(i)-y2)^2)^0.5
    
    %x1=int32(round(x1));
    %y1=int32(round(y1));
    %x2=int32(round(x2));
    %y2=int32(round(y2));
    x=int32(round(x1));
    y=int32(round(y1));
    % ustalenie kierunku rysowania

    if x1<x2
        xi=int32(1);
        dx=int32(x2-x1);
    else
        xi=int32(-1);
        dx=int32(x1-x2);
    end
    % ustalenie kierunku rysowania
    if y1<y2
        yi=int32(1);
        dy=int32(y2-y1);
    else
        yi=int32(-1);
        dy=int32(y1-y2);
    end
    % pierwszy piksel
    j=j+1;
    v(j,1)=x;%+X(i);
    v(j,2)=y;%+Y(i);
    if x<1
        v(j,1)=1;
    end
    if x>640
        v(j,1)=640;
    end
    if y<1
        v(j,2)=1;
    end
    if y>480
        v(j,2)=480;
    end
    %j
    line(j)=Frame(v(j,2),v(j,1),3);
    % oœ wiod¹ca OX
    if dx>dy
        ai=(dy-dx)*2;
        bi=dy*2;
        d=bi-dx;
        % pêtla po kolejnych x
        while x~=int32(round(x2))
            % test wspó³czynnika
            if d>=0
                x=x+xi;
                y=y+yi;
                d=d+ai;
            else
                d=d+bi;
                x=x+xi;
            end
            j=j+1;
            v(j,1)=x;%+X(i);
            v(j,2)=y;%+Y(i);
            if x<1
                v(j,1)=1;
            end
            if x>640
                v(j,1)=640;
            end
            if y<1
                v(j,2)=1;
            end
            if y>480
                v(j,2)=480;
            end
            line(j)=Frame(v(j,2),v(j,1),3);
        end
    % oœ wiod¹ca OY
    else
        ai=(dx-dy)*2;
        bi=dx*2;
        d=bi-dy;
        % pêtla po kolejnych y
        while y~=int32(round(y2))
            % test wspó³czynnika
            if d>=0
                x=x+xi;
                y=y+yi;
                d=d+ai;
            else
                d=d+bi;
                y=y+yi;
            end
            j=j+1;
            v(j,1)=x;%+X(i);
            v(j,2)=y;%+Y(i);
            if x<1
                v(j,1)=1;
            end
            if x>640
                v(j,1)=640;
            end
            if y<1
                v(j,2)=1;
            end
            if y>480
                v(j,2)=480;
            end
            line(j)=Frame(v(j,2),v(j,1),3);
        end
    end
    %plot(line(1:j))
    %max(line)
    %hf = imtool( Frame./(max(max(max(Frame)))/20) );
    %ha = get(hf,'CurrentAxes');
    %hold(ha,'on');
    %hp=plot(ha,X,Y,'-xb');
    %hs=scatter(ha,v(1:j,1),v(1:j,2),'filled','MarkerFaceColor','b');
    %hs=scatter(ha,X(i),Y(i),'filled','MarkerFaceColor','m');
    %hs=scatter(ha,x1,y1,'filled','MarkerFaceColor','r');
    %hs=scatter(ha,x2,y2,'filled','MarkerFaceColor','c');
    quality=max(line(1:j))-min(line(1:j));
    k=k+1;
    c='b';
    if FitFresnel
        %point=FindShadowAndLightBorder(line(1:j));
        %dist=BorderDistance(X,Y,v(point,1),v(point,2));
        %quality=-dist;
        %x(1) - skala wartoœci
        %x(2) - przesuniêcie argumentów
        %x(3) - gêstoœæ argumentów
        %x(4) - przesuniêcie wartoœci
        xdata=1:j;
        ydata=line(1:j)-min(line(1:j));
        ydata=ydata./max(ydata);
        %fun=@(x,xdata)x(1)*fresnelc((xdata-x(2))/x(3))-x(4);
        fun=@(x,xdata)x(1)*myDiffractionFunction((xdata-x(2))/x(3))-x(4);
        x0=[2 25 -8 0];
        lb=[0 1 -16 -1];
        ub=[2 j 16 1];
        [x,resnorm,residual,exitflag,output] = lsqcurvefit(fun,x0,xdata,ydata',lb,ub,optimoptions('lsqcurvefit','Diagnostics',showDiagnostics,'Display',showDisplay,'ScaleProblem','jacobian','TolFun',1e-16));
        meanx=mean(ydata);
        %relativeError=sqrt(resnorm)/meanx*100;
        relativeError=sqrt(norm(residual,Inf))/(meanx)*100;
        if DisplayedWindows.FresnelFitPlots
            figure('Name',sprintf('Fresnel fit: %d, b³¹d wzglêdny: %f%% residuum norm: %e, x: %e %f %e %e',i,relativeError,sqrt(resnorm),x(1),x(2),x(3),x(4)))
            plot(xdata,ydata')
            hold on
            mp=plot(xdata,fun(x,xdata));
            hold off;
        end
        point=FindShadowAndLightBorder(line(1:j));
        quality=1/abs(point-x(2));
    end
    data(k)=struct('v',v,'j',j,'X',X,'Y',Y,'line',line,'quality',quality,'color',c,'inColorIndex',i);
    %waitfor(hf);
end

vq=zeros(k,1);
for i=1:k
    vq(i)=data(i).quality;
end
[Q,I]=sort(vq,'descend');
if DisplayedWindows.SPointsWindow
    hf = imtool( Frame./(max(max(max(Frame)))/20) );
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    hp=plot(ha,X,Y,'-xb');
end
r=658;
g=532;
b=458;
try
    lambdas=evalin('base', 'lambdas');
catch
    lambdas=[r,g,b];
end
r=lambdas(1);
g=lambdas(2);
b=lambdas(3);
[X Y]=BorderFunction(Px,Py,Pz,ShX,ShY,lCCD,r);
if DisplayedWindows.SPointsWindow
    hp=plot(ha,X,Y,'-xr');
end
ib=0;
ir=0;
load('BR_settings.mat','BPoints');
for i=1:BPoints%12
    if DisplayedWindows.SPointsWindow
        hs=scatter(ha,data(I(i)).v(1:data(I(i)).j,1),data(I(i)).v(1:data(I(i)).j,2),'filled','MarkerFaceColor',data(I(i)).color);
    end
    point=FindShadowAndLightBorder(data(I(i)).line(1:data(I(i)).j));
    if DisplayedWindows.SPointsWindow
        hs=scatter(ha,data(I(i)).v(point,1),data(I(i)).v(point,2),'filled','MarkerFaceColor','c');
    end
    
    x = data(I(i)).v(point,1); y = data(I(i)).v(point,2);% scatter(x,y);
    %a = selectedPoints(I(i)); b = num2str(a); c = cellstr(b);
    a = data(I(i)).inColorIndex; b = num2str(a); c = cellstr(b);
    dx = 1.1; dy = 0.1; % displacement so the text does not overlay the data points

    %text(x+dx, y+dy, c,'fontsize',18,'color',[0,1,0], 'Parent', ha);
    
    if data(I(i)).color=='r'
        ir=ir+1;
        pointsr(ir,1)=data(I(i)).v(point,1);
        pointsr(ir,2)=data(I(i)).v(point,2);
        if DisplayedWindows.SPointsWindow
            text(x+dx, y+dy, c,'fontsize',18,'color',[1,1,0], 'Parent', ha);
        end
    else
        ib=ib+1;
        pointsb(ib,1)=data(I(i)).v(point,1);
        pointsb(ib,2)=data(I(i)).v(point,2);
        if DisplayedWindows.SPointsWindow
            text(x+dx, y+dy, c,'fontsize',18,'color',[0,1,1], 'Parent', ha);
        end
    end
end

end

