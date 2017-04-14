function [ Pk, PCCD ] = BorderRecognition( Frame, initial_point, System )
%BORDERRECOGNITION Procedura dobierania parametrów ramki do filmu
%   Detailed explanation goes here

if(exist('initial_point','var'))
    if(length(initial_point)~=6)
        initial_point=[0,0,0,0,0,82];
    end
else
    initial_point=[0,0,0,0,0,82];
end


load('BR_settings.mat','BP','Op','VFch','BrightTime','OptTime','DisplayedWindows','ManualPointCorrection','SPointsR','SPointsB','FitFresnel');
myMaxTime=BrightTime;

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

global efDr efDg efDb GSystem wb;
if exist('System', 'var')
    handles.S=System;
else
    handles.S=SetSystem;
end
GSystem=handles.S;
efDr  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,r,25);
efDg  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,g,25);
efDb  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,b,25);

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

%rêczne wybieranie punktów brzegowych
if BP==2
    %hf = imtool( Frame./(max(max(max(Frame)))/20) );
    hf = imtool( Frame(:,:,1), [ min(min(Frame(:,:,1))) max(max(Frame(:,:,1))) ]);
    set(hf,'name','Select Border Points for red color!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    h = impoly(ha);
    positionr = wait(h);
    delete(h);

    delete(hf);
    hf = imtool( Frame(:,:,3), [ min(min(Frame(:,:,3))) max(max(Frame(:,:,3))) ]);
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');

    hs=scatter(ha,positionr(:,1),positionr(:,2),'filled','MarkerFaceColor','m');

    set(hf,'name','Select Border Points for blue color!')
    hb = impoly(ha);
    positionb = wait(hb);
    delete(hb);

    delete(hf);

    hf = imtool( Frame./(max(max(max(Frame)))/20) );
end
%znalezienie i wyœwietlenie obrazu binarnego do optymalizacji po funkcji
%skalaryzuj¹cej jasnoœci
if BP==1 || BP==3
    [a1 a2]=ThresholdValue(Frame);
    %hf = imtool( (Frame(:,:,1)./max(max(Frame(:,:,1)))>a1)|(Frame(:,:,3)./max(max(Frame(:,:,3)))>a2) );
    tmpF=Frame;
    tmpF(:,:,1)=(Frame(:,:,1)./max(max(Frame(:,:,1)))>a1).*0.5;
    tmpF(:,:,2)=zeros(size(Frame(:,:,2)));
    tmpF(:,:,3)=(Frame(:,:,3)./max(max(Frame(:,:,3)))>a2).*0.5;
    hf = imtool( tmpF );
    %imtool( (Frame(:,:,1)./max(max(Frame(:,:,1)))<a1)&(Frame(:,:,3)./max(max(Frame(:,:,3)))<a2) );
    %imtool( (Frame(:,:,3)./max(max(Frame(:,:,3)))>0.09) );
end
ha = get(hf,'CurrentAxes');
hold(ha,'on');

%narysowanie rêcznie wybranych punktów brzegowych
if BP==2
    hs=scatter(ha,positionr(:,1),positionr(:,2),'filled','MarkerFaceColor','m');

    hs=scatter(ha,positionb(:,1),positionb(:,2),'filled','MarkerFaceColor','c');
end

t0=tic;
t1=0;

%Position=position;


%wstêpne narysowanie ramek
[X Y]=BorderFunction(0,0,0,0,0,82,r);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(0,0,0,0,0,82,b);
hpb=plot(ha,X,Y,'-xb');

allTime=BrightTime+OptTime+10*(size(SPointsR,2)+size(SPointsB,2))*FitFresnel;
timeLeft=allTime;
wb=waitbar(0,sprintf('Progress: 0%% Estimated time left: %f s / %d s \n Brightnes scalarisation optimization...',allTime,allTime));
t5=tic;

%funkcja wyœwietlania
function stop = myoutfun(x, optimValues, state)
stop = false;
[X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),r);
%[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);
delete(hp);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),b);
%[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);
delete(hpb);
hpb=plot(ha,X,Y,'-xb');
set(hf,'name',sprintf('%f %f %f %f %f %f',x(1),x(2),x(3),x(4),x(5),x(6)))
drawnow
tp1=toc(t5);
timeLeft=timeLeft-tp1;
waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n Local optimizer..',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
t5=tic;
%display(Br+Bb+Dr+Db);
if(toc(t0)>OptTime)
    stop=true;
end
if(toc(t1)>myMaxTime)
    stop=true;
end
end

%funkcja wyœwietlania i rysowania wykresu dla symulowanego wy¿arzania
function stop = saplotfun(options,optimvalues,flag)
    stop = false;
x=optimvalues.x;
[X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),r);
%[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);
delete(hp);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),b);
%[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);
delete(hpb);
hpb=plot(ha,X,Y,'-xb');
set(hf,'name',sprintf('%f %f %f %f %f %f',x(1),x(2),x(3),x(4),x(5),x(6)))
drawnow
tp1=toc(t5);
timeLeft=timeLeft-tp1;
waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n Simulated Anealing..',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
t5=tic;
%display(Br+Bb+Dr+Db);
if(toc(t0)>OptTime)
    stop=true;
end
if(toc(t1)>myMaxTime)
    stop=true;
end
if DisplayedWindows.SimAnealingWindow
    switch flag
        case 'init'
            plotBest = plot(optimvalues.iteration,optimvalues.bestfval, '.b');
            set(plotBest,'Tag','saplotbestf');
            xlabel('Iteration','interp','none');
            ylabel('Function value','interp','none')
            title(sprintf('Best Function Value: %g',optimvalues.bestfval),'interp','none');
        case 'iter'
            plotBest = findobj(get(gca,'Children'),'Tag','saplotbestf');
            newX = [get(plotBest,'Xdata') optimvalues.iteration];
            newY = [get(plotBest,'Ydata') optimvalues.bestfval];
            set(plotBest,'Xdata',newX, 'Ydata',newY);
            set(get(gca,'Title'),'String',sprintf('Best Function Value: %g',optimvalues.bestfval));
    end
end
end

function stop = myoutfun2(x, optimValues, state)
stop = false;
[X Y]=BorderFunction(Args(1),Args(2),Args(3),x(1),Args(5),x(2),r);
%[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);
delete(hp);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(Args(1),Args(2),Args(3),x(1),Args(5),x(2),b);
%[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);
delete(hpb);
hpb=plot(ha,X,Y,'-xb');
set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),x(1),Args(5),x(2)))
drawnow
tp1=toc(t5);
timeLeft=timeLeft-tp1;
waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n X-Z optimization..',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
t5=tic;
%display(Br+Bb+Dr+Db);
if mod(optimValues.iteration,1000)==0 || strcmp(state,'done')
    map2dXZ=zeros(20,20);
    y=zeros(20);
    z=zeros(20);
    y=linspace(x(1)-1,x(1)+1,20);
    z=linspace(x(2)-1,x(2)+1,20);
    for i=1:20
        for j=1:20
            %XZ
            point=[0,0,0,x(1)+(j-10)*0.1,0,x(2)+(i-10)*0.1];
            map2dXZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        end
        str=sprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\t\t%6.3f%%\n',((i-1)/0.2));
        display(str);
    end
    figure('Name','XZ')
    mesh(y,z,map2dXZ);
end
if(toc(t0)>300)
    stop=true;
end
if(toc(t1)>150)
    stop=true;
end
end

function stop = myoutfun3(x, optimValues, state)
stop = false;
[X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),x(1),x(2),r);
%[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);
delete(hp);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),x(1),x(2),b);
%[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);
delete(hpb);
hpb=plot(ha,X,Y,'-xb');
set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),x(1),x(2)))
drawnow
%display(Br+Bb+Dr+Db);
tp1=toc(t5);
timeLeft=timeLeft-tp1;
waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n Y-Z optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
t5=tic;
if mod(optimValues.iteration,1000)==0 || strcmp(state,'done')
    map2dYZ=zeros(20,20);
    y=zeros(20);
    z=zeros(20);
    y=linspace(x(1)-1,x(1)+1,20);
    z=linspace(x(2)-1,x(2)+1,20);
    for i=1:20
        for j=1:20
            %YZ
            point=[0,0,0,0,x(1)+(j-10)*0.1,x(2)+(i-10)*0.1];
            map2dYZ(i,j)=MeanSquaredDistance(pointsr,pointsb,point);
        end
        str=sprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\t\t%6.3f%%\n',((i-1)/0.2));
        display(str);
    end
    figure('Name','YZ')
    mesh(y,z,map2dYZ);
end
if(toc(t0)>400)
    stop=true;
end
if(toc(t1)>300)
    stop=true;
end
end

%funkcja wyœwietlania dla algorytmu genetycznego
function [state, options,optchanged] = mygaoutputfcn(options,state,flag)
optchanged = false;

switch flag
 case 'init'
        disp('Starting the algorithm');
        t5=tic;
    case {'iter','interrupt'}
        %disp('Iterating ...')
        ibest = state.Best(end);
        ibest = find(state.Score == ibest,1,'last');
        bestx = state.Population(ibest,:);
        x=bestx;

        [X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),r);
        %[Br, Dr]=BrightInDimOut(Frame(:,:,1),X,Y);
        delete(hp);
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(x(1),x(2),x(3),x(4),x(5),x(6),b);
        %[Bb, Db]=BrightInDimOut(Frame(:,:,3),X,Y);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',x(1),x(2),x(3),x(4),x(5),x(6)))
        drawnow
        tp1=toc(t5);
        timeLeft=timeLeft-tp1;
        waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n Genetic Algorithm...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
        t5=tic;
    case 'done'
        disp('Performing final task');
end
end

t5=tic;

if 0
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);

    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);

elseif 0
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
elseif 0
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','dfp','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','steepdesc','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
elseif BP==2 && Op==1 %dwueatpowa optymalizacja dla rêcznie wybranych punktów
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    tp1=toc(t1)
    
    timeLeft=timeLeft-tp1;
    waitbar(tp1/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of 1 of 2 step optimization...',tp1/allTime*100,timeLeft,allTime));
    
    t1=tic;
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9, 'DiffMinChange', 1e-2);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    tp1=toc(t1)
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of 2 of 2 step optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op~=3 && BP==1%fminsearch funkcji skalaryzuj¹cej po jasnoœci
    t1=tic;
    if DisplayedWindows.BrightnesWindow
        options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    else
        delete(hf);
        options = optimset('Display',showDisplay,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    end
    [Args, f,exitflag,output]=fminsearch(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    tp1=toc(t1)
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of fminsearch brightnes scalarization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op~=3 && BP==3%symulowane wy¿arzanie funkcji skalaryzuj¹cej po jasnoœci
    t1=tic;
    %options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    %[Args, f,exitflag,output]=fminsearch(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,options);
    
    hybridopts = optimset('Display',showDisplay);
    if DisplayedWindows.OptimInfo
        options=optimset('Diagnostics',showDiagnostics,'Display',showDisplay);
    else
        options=optimset();
    end
    if DisplayedWindows.BrightnesWindow
        saoptions=saoptimset(options,'HybridFcn',{@patternsearch,hybridopts},'PlotFcns',@saplotfun);
    else
        delete(hf);
        saoptions=saoptimset(options,'HybridFcn',{@patternsearch,hybridopts});
    end
    [Args, f,exitflag,output]=simulannealbnd(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,[-0.6,-0.6,-0.6,-1.5,-1.5,initial_point(6)-3],[0.6,0.6,0.6,1.5,1.5,initial_point(6)+9],saoptions);
    
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    tp1=toc(t1)
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of Simulated anealing by brightnes scalarization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
end
myMaxTime=OptTime;

if Op~=3 && DisplayedWindows.BrightnesWindow
    if ~exist('Args','var')
        Args=initial_point;
    end
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    hp=plot(ha,X,Y,'-xr');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
    delete(hpb);
    hpb=plot(ha,X,Y,'-xb');

%    output

    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
end
%wybieranie punktów brzegowych
if (BP==1 || BP==3) && Op~=3
    t0=tic;
    [pointsr, pointsb]=FindBorderPoints(Frame, [Pk,PCCD]);
    %rêczne korygowanie wybranych punktów brzegowych
    if ManualPointCorrection
        hf = imtool( Frame(:,:,1), [ min(min(Frame(:,:,1))) max(max(Frame(:,:,1))) ]);
        set(hf,'name','Modify border points selection in red chanel!');
        ha = get(hf,'CurrentAxes');
        hold(ha,'on');
        h = impoly(ha,pointsr);
        pointsr = wait(h);
        delete(h);
        delete(hf);
        hf = imtool( Frame(:,:,3), [ min(min(Frame(:,:,3))) max(max(Frame(:,:,3))) ]);
        set(hf,'name','Modify border points selection in blue chanel!');
        ha = get(hf,'CurrentAxes');
        hold(ha,'on');
        h = impoly(ha,pointsb);
        pointsb = wait(h);
        delete(h);
        delete(hf);
    end
    tp1=toc(t0);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of border points selection',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
    t0=tic;
    t1=tic;
elseif BP==2
    pointsr(:,1)=positionr(:,1);
    pointsr(:,2)=positionr(:,2);
    pointsb(:,1)=positionb(:,1);
    pointsb(:,2)=positionb(:,2);
end
if Op~=3 && (DisplayedWindows.FinalOptWindow || VFch)
    hf = imtool( Frame./(max(max(max(Frame)))/20) );
    %hf = imtool( Frame(:,:,1), [ min(min(Frame(:,:,1))) max(max(Frame(:,:,1))) ]);
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    hs=scatter(ha,pointsr(:,1),pointsr(:,2),'filled','MarkerFaceColor','m');
    hs=scatter(ha,pointsb(:,1),pointsb(:,2),'filled','MarkerFaceColor','c');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    hp=plot(ha,X,Y,'-xr');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
    hpb=plot(ha,X,Y,'-xb');
end

t0=tic;
t1=tic;
t5=tic;
if Op==1

    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'Diagnostics',showDiagnostics,'MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    tp1=toc(t1)
    
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of Gradient optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));

    t1=tic;

    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun,'MaxIter',2400,'MaxFunEvals',4800,'TolFun',1e-9,'TolX',1e-9);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of Non-gradient optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op==2

    [Args, f,exitflag,output]=ga(@(x)MeanSquaredDistance(pointsr,pointsb,x),6,[1,0,0,0,0,0;0,1,0,0,0,0;0,0,1,0,0,0;0,0,0,1,0,0;0,0,0,0,1,0;0,0,0,0,0,1],[1,1,1,1.5,1.5,Args(6)+3],[],[],[-1,-1,-1,-1.5,-1.5,Args(6)-3],[1,1,1,1.5,1.5,Args(6)+3],[],gaoptimset('Display',showDisplay,'OutputFcn',@mygaoutputfcn,'TimeLimit',OptTime));
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of Genetic Algorithm optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op==3
    if exist('myNeuralNetworkFunction','file')
        Args = myNeuralNetworkFunction(reshape(Frame,480*640*3,1));
        x=Args;
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
            delete(hp);
        if (DisplayedWindows.FinalOptWindow || VFch)
            hp=plot(ha,X,Y,'-xr');
            [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
            delete(hpb);
            hpb=plot(ha,X,Y,'-xb');
            set(hf,'name',sprintf('%f %f %f %f %f %f',x(1),x(2),x(3),x(4),x(5),x(6)))
            drawnow
        end
    else
        display('Brak sieci neuronowej!');
    end
elseif Op==4
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun2,'MaxIter',2400,'MaxFunEvals',4800,'TolFun',1e-20,'TolX',1e-20);
    [Args([4,6]), f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,[Args(1),Args(2),Args(3),x(1),Args(5),x(2)]),[Args(4),Args(6)],options);
    
    options = optimset('Display',showDisplay,'OutputFcn',@myoutfun3,'MaxIter',4800,'MaxFunEvals',9600,'TolFun',1e-20,'TolX',1e-20);
    [Args([5,6]), f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,[Args(1),Args(2),Args(3),Args(4),x(1),x(2)]),[Args(5),Args(6)],options);
    
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op==5
    [Args, f,exitflag,output]=lsqnonlin(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],[],[],optimset('Algorithm','trust-region-reflective','Diagnostics',showDiagnostics,'Display',showDisplay,'OutputFcn',@myoutfun));
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    if (DisplayedWindows.FinalOptWindow || VFch)
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
        drawnow
    end
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op==6
    if ~DisplayedWindows.FinalOptWindow && VFch==0
        ha=-1;
    end
    Args = mySteepestDescent( Pk,PCCD,ha,hf,hp,hpb,t1,OptTime,pointsr,pointsb )
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    if (DisplayedWindows.FinalOptWindow || VFch)
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
        drawnow
    end
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of My Steepest Descent optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
elseif Op==7
    hybridopts = optimset('Display',showDisplay);
    options=optimset('Diagnostics','on','Display',showDisplay);
    if (DisplayedWindows.FinalOptWindow || VFch)
        saoptions=saoptimset(options,'HybridFcn',{@patternsearch,hybridopts},'PlotFcns',@saplotfun);
    else
        saoptions=saoptimset(options,'HybridFcn',{@patternsearch,hybridopts});
    end
    [Args, f,exitflag,output]=simulannealbnd(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],[-0.6,-0.6,-0.6,-1.5,-1.5,Args(6)-3],[0.6,0.6,0.6,1.5,1.5,Args(6)+3],saoptions);
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    if (DisplayedWindows.FinalOptWindow || VFch)
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
        drawnow
    end
    tp1=toc(t1);
    timeLeft=timeLeft-tp1;
    waitbar((allTime-timeLeft)/allTime,wb,sprintf('Progress: %f%% Estimated time left: %f s / %d s \n End of Simulated Anealing optimization...',(allTime-timeLeft)/allTime*100,timeLeft,allTime));
end
toc(t1)
Pk=[Args(1),Args(2),Args(3)];
PCCD=[Args(4),Args(5),Args(6)];
%ka¿da klatka identyfikowana jest unikalnym hashem
evalin('base','sha1hasher=System.Security.Cryptography.SHA1Managed');
%f=reshape(char(Frame),480*640*3,1);
f=typecast(reshape(Frame,480*640*3,1),'uint8');
assignin('base', 'f', f);
sha1= uint8(evalin('base','sha1hasher.ComputeHash(uint8(f))'));
%display as hex:
hc=dec2hex(sha1);
save(sprintf('TD%s.mat',num2str(hc)),'Frame','Pk','PCCD');
if VFch==1
    assignin('base', 'Frame', Frame);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    if exist('pointsr','var')
        Args=ViewFinder(Pk,PCCD,ha,hf,hp,hpb,pointsr,pointsb);
    end
end
Pk=[Args(1),Args(2),Args(3)];
PCCD=[Args(4),Args(5),Args(6)];

close(wb);
toc(t0)
end

