function [ Pk, PCCD ] = BorderRecognition( Frame, initial_point )
%BORDERRECOGNITION Procedura dobierania parametrów ramki do filmu
%   Detailed explanation goes here

if(exist('initial_point','var'))
    if(length(initial_point)~=6)
        initial_point=[0,0,0,0,0,82];
    end
else
    initial_point=[0,0,0,0,0,82];
end

load('BR_settings.mat','BP','Op','VFch','BrightTime','OptTime');
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

global efDr efDg efDb;

handles.S=SetSystem;
efDr  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,r,25);
efDg  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,g,25);
efDb  = effective_aperture(handles.S.D/2,handles.S.tc,handles.S.l1,b,25);

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

if BP==2
    hs=scatter(ha,positionr(:,1),positionr(:,2),'filled','MarkerFaceColor','m');

    hs=scatter(ha,positionb(:,1),positionb(:,2),'filled','MarkerFaceColor','c');
end

t0=tic;
t1=0;

%Position=position;



[X Y]=BorderFunction(0,0,0,0,0,82,r);
hp=plot(ha,X,Y,'-xr');
[X Y]=BorderFunction(0,0,0,0,0,82,b);
hpb=plot(ha,X,Y,'-xb');

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
%display(Br+Bb+Dr+Db);
if(toc(t0)>OptTime)
    stop=true;
end
if(toc(t1)>myMaxTime)
    stop=true;
end
end

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
%display(Br+Bb+Dr+Db);
if(toc(t0)>OptTime)
    stop=true;
end
if(toc(t1)>myMaxTime)
    stop=true;
end
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

function [state, options,optchanged] = mygaoutputfcn(options,state,flag)
optchanged = false;

switch flag
 case 'init'
        disp('Starting the algorithm');
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
    case 'done'
        disp('Performing final task');
end
end

if 0
    options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);

    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);

elseif 0
    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
elseif 0
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','dfp','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','steepdesc','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
elseif BP==2 && Op==1
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(positionr,positionb,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
    
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9, 'DiffMinChange', 1e-2);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(positionr,positionb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
elseif Op~=3 && BP==1
    t1=tic;
    options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    [Args, f,exitflag,output]=fminsearch(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
elseif Op~=3 && BP==3
    t1=tic;
    %options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',1200,'TolFun',1e-9,'TolX',1e-9);
    %[Args, f,exitflag,output]=fminsearch(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,options);
    
    hybridopts = optimset('Display','iter');
    options=optimset('Diagnostics','on','Display','iter');
    [Args, f,exitflag,output]=simulannealbnd(@(x)BrightnesScalarization(Frame,a1,a2,x),initial_point,[-0.6,-0.6,-0.6,-1.5,-1.5,initial_point(6)-3],[0.6,0.6,0.6,1.5,1.5,initial_point(6)+9],saoptimset(options,'HybridFcn',{@patternsearch,hybridopts},'PlotFcns',@saplotfun));
    
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)
end
myMaxTime=OptTime;

if Op~=3
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
if (BP==1 || BP==3) && Op~=3
    [pointsr, pointsb]=FindBorderPoints(Frame, [Pk,PCCD]);
elseif BP==2
    pointsr(:,1)=positionr(:,1);
    pointsr(:,2)=positionr(:,2);
    pointsb(:,1)=positionb(:,1);
    pointsb(:,2)=positionb(:,2);
end
if Op~=3
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

t1=tic;
if Op==1

    options = optimset('Display','iter','OutputFcn',@myoutfun,'Diagnostics','on','MaxFunEvals',1200,'HessUpdate','bfgs','TolFun',1e-9,'TolX',1e-9,'TypicalX',[1e-1,1,1,1,1,1e-1]);
    [Args, f,exitflag,output]=fminunc(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    toc(t1)

    t1=tic;

    options = optimset('Display','iter','OutputFcn',@myoutfun,'MaxIter',2400,'MaxFunEvals',4800,'TolFun',1e-9,'TolX',1e-9);
    [Args, f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],options);
elseif Op==2

    [Args, f,exitflag,output]=ga(@(x)MeanSquaredDistance(pointsr,pointsb,x),6,[1,0,0,0,0,0;0,1,0,0,0,0;0,0,1,0,0,0;0,0,0,1,0,0;0,0,0,0,1,0;0,0,0,0,0,1],[1,1,1,1.5,1.5,Args(6)+3],[],[],[-1,-1,-1,-1.5,-1.5,Args(6)-3],[1,1,1,1.5,1.5,Args(6)+3],[],gaoptimset('Display','iter','OutputFcn',@mygaoutputfcn,'TimeLimit',OptTime));

elseif Op==3
    if exist('myNeuralNetworkFunction','file')
        Args = myNeuralNetworkFunction(reshape(Frame,480*640*3,1));
        x=Args;
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
        delete(hp);
        hp=plot(ha,X,Y,'-xr');
        [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
        delete(hpb);
        hpb=plot(ha,X,Y,'-xb');
        set(hf,'name',sprintf('%f %f %f %f %f %f',x(1),x(2),x(3),x(4),x(5),x(6)))
        drawnow
    else
        display('Brak sieci neuronowej!');
    end
elseif Op==4
    options = optimset('Display','iter','OutputFcn',@myoutfun2,'MaxIter',2400,'MaxFunEvals',4800,'TolFun',1e-20,'TolX',1e-20);
    [Args([4,6]), f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,[Args(1),Args(2),Args(3),x(1),Args(5),x(2)]),[Args(4),Args(6)],options);
    
    options = optimset('Display','iter','OutputFcn',@myoutfun3,'MaxIter',4800,'MaxFunEvals',9600,'TolFun',1e-20,'TolX',1e-20);
    [Args([5,6]), f,exitflag,output]=fminsearch(@(x)MeanSquaredDistance(pointsr,pointsb,[Args(1),Args(2),Args(3),Args(4),x(1),x(2)]),[Args(5),Args(6)],options);
    
elseif Op==5
    [Args, f,exitflag,output]=lsqnonlin(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],[],[],optimset('Algorithm','trust-region-reflective','Diagnostics','on','Display','iter','OutputFcn',@myoutfun));
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    hp=plot(ha,X,Y,'-xr');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
    delete(hpb);
    hpb=plot(ha,X,Y,'-xb');
    set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
    drawnow
elseif Op==6
    Args = mySteepestDescent( Pk,PCCD,ha,hf,hp,hpb,t1,OptTime,pointsr,pointsb )
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    hp=plot(ha,X,Y,'-xr');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
    delete(hpb);
    hpb=plot(ha,X,Y,'-xb');
    set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
    drawnow
elseif Op==7
    hybridopts = optimset('Display','iter');
    options=optimset('Diagnostics','on','Display','iter');
    [Args, f,exitflag,output]=simulannealbnd(@(x)MeanSquaredDistance(pointsr,pointsb,x),[Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)],[-0.6,-0.6,-0.6,-1.5,-1.5,Args(6)-3],[0.6,0.6,0.6,1.5,1.5,Args(6)+3],saoptimset(options,'HybridFcn',{@patternsearch,hybridopts},'PlotFcns',@saplotfun));
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
    delete(hp);
    hp=plot(ha,X,Y,'-xr');
    [X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
    delete(hpb);
    hpb=plot(ha,X,Y,'-xb');
    set(hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
    drawnow
end
toc(t1)
if VFch==1
    Pk=[Args(1),Args(2),Args(3)];
    PCCD=[Args(4),Args(5),Args(6)];
    if exist('pointsr','var')
        Args=ViewFinder(Pk,PCCD,ha,hf,hp,hpb,pointsr,pointsb);
    end
end
Pk=[Args(1),Args(2),Args(3)];
PCCD=[Args(4),Args(5),Args(6)];


toc(t0)
end

