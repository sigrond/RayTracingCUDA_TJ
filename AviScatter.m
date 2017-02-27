function varargout = AviScatter(varargin)

% Edit the above text to modify the response to help AviScatter

% Last Modified by GUIDE v2.5 18-Feb-2011 09:36:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @AviScatter_OpeningFcn, ...
    'gui_OutputFcn',  @AviScatter_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before AviScatter is made visible.
function AviScatter_OpeningFcn(hObject, eventdata, handles, varargin)


% Choose default command line output for AviScatter
handles.output = hObject;
handles.MaskIpp = 0; %
handles.MaskIps = 0; %
handles.key = 0;  %
handles.path = 0;
handles.segment = [1 1]; % zakres klatek do obliczenia
% ___________________________________________________________________
% poczatkowe ustawienia czek boxsow oraz slajdera
set(handles.chbIpp,'Value',0);
set(handles.chbIps,'Value',0);
set(handles.slFrame,'Max',10,'Value',1,'Min',1);
%
%wyœwietlamy obraz
Frame = FrameReader(hObject,handles);
handles.hFrame = imshow(Frame);

% Update handles structure
guidata(hObject, handles);
% ---------------------------------------------------------------
%


% UIWAIT makes AviScatter wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = AviScatter_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;
%
% ========================================================================
% My Functions
function Frame = FrameReader(hObject,handles)
%
if length( handles.path )<= 1  % jak niema wprowadzonej sciezki to wydaje
    % czarna klatke albo moon.tif
    try
        Frame = imread('moon.tif');
    catch
        Frame = zeros(480,640);
    end
    %
else
    % W tym miejscu wprowadzone zmiany aviread na Aviread12_matlab
    temp = Aviread12_matlab( handles.path,handles.nom );
    if handles.key == 0  % klatka nie wyprostowana
        Frame = double(reshape(temp(:,:,:,1:2),480,640,2));%% double( sum(temp.cdata,3) );
    else                 % kaltka wyprostowana
        Fr = sum(double(temp.cdata),3);
        shx = round( 320.5 - handles.params(1) );
        shy = round( 240.5 - handles.params(2) );
        sFr = circshift( Fr,[shy shx] );
        Frame = przeli_od( sFr,handles.T1,handles.T2 );
    end
end
%
% ------------------------------------------------------------------------
function [Phi theta sh Ind ] = ThetaWorker(hObject, handles)
% ustawimy os optyczna
shx = round( 320.5 - get( handles.slPr,'value') );%handles.params(1) );
shy = round( 240.5 - handles.params_R(2) );
% przesuwamy maske bo przy prostowaniu przesuwa sie zero katow
handles.MaskIpp = circshift( handles.MaskIpp,[shy shx] );


Thpp  = reshape ( ( handles.TF_frameR(:,:,1)' .* handles.MaskIpp ),1,480*640 );
Phi.pp = reshape ( handles.TF_frameR(:,:,2)',1,480*640 );

ind   = find( Thpp );
Ind.p = ind;
nzTp = Thpp( ind );
span_R = 2*round( length( ind )/1200 ) + 1;
ind_reduced_R = 1:span_R:length(ind);
theta.mTp = nzTp( ind_reduced_R );
theta.nzTp = nzTp;
sh.shx = shx;
sh.shyR = shy;
sh.span_R = span_R;
% ______________________________________________________________
% druga polaryzacia
%  to samo dla k¹tów

shy = round( 240.5 - handles.params_G(2) );
sh.shyG = shy;

handles.MaskIps = circshift( handles.MaskIps,[shy shx] );

Thss  = reshape ( ( handles.TF_frameG(:,:,1)' .* handles.MaskIps),1,480*640 );
Phi.ss = reshape ( handles.TF_frameG(:,:,2)',1,480*640 );
%     Wyrzucamy wszystkie zerowe elementy
ind   = find( Thss );
Ind.s = ind;
%     odejmujemy t³o oraz uwzgledniamy zaleznoœæ od k¹ta phi
span_G = 2*round(length( ind )/1200)+1;
%   wektor odpowiednich k¹tów theta
nzTs = Thss( ind );
theta.nzTs = nzTs;
sh.span_G = span_G;
% wybieramy co span_R'ty punkt
ind_reduced_G = 1:span_G:length( ind );
theta.mTs =nzTs(ind_reduced_G);


% ------------------------------------------------------------------------
%
function [Ip Is ] = FrameWorker(hObject, handles, Frame, sh, Phi, theta,Ind,ivs,ivp);
%
FrR = double( Frame(:,:,1) ); %double( Frame.cdata(:,:,1) );
sFrR = circshift( FrR,[ sh.shyR sh.shx ] );
FrameR = sFrR;

if ~isscalar( handles.MaskIpp )
    %  wyliczamy poziom tla
st =   FrameR .* handles.IppTlo;
    Ipptlo = sum( st(:) ) /  sum( handles.IppTlo(:) );
    %  Wybieramy obszar Ipp z klatki filmu i przekszta³czmy go w wektor
    FrR    = reshape (  FrameR, 1,480*640 );
    %  to samo dla k¹tów
    %     Wyrzucamy wszystkie zerowe elementy, ind jest wektorem indeksów

    % przed uœrednianiem odejmujemy t³o oraz uwzgledniamy zaleznoœæ od k¹ta phi
    nzFrR = ( FrR( Ind.p ) - Ipptlo )  ./ ( cos( Phi.pp( Ind.p ) ).^2 );
    %   wektor odpowiednich k¹tów theta

    % iloœæ punktów do uœrednienia; zamiast 1200 nale¿a³oby wstawiæ 2*size(...)

    % wybieramy co span_R'ty punkt

%     ind_reduced_R = 1:sh.span_R:length(Ind.p);
    
    

 Ip = zeros(1,length( theta.mTp ) );
 if ivp( 1 ) == 1
     Ip( 1 ) =  nzFrR( 1 );
 else
     Ip( 1 ) = median( nzFrR(1:ivp( 1 ) - 1 ) );
 end
 
for ii = 2 : length( ivp ) - 2
     Ip( ii ) = median( nzFrR(ivp(ii)+ 1:ivp(ii+1)-1));
    
end

 
%     Ip_smoo = smooth( theta.nzTp, nzFrR, sh.span_R,'rloess');
%     Ip = Ip_smoo( ind_reduced_R );

else
    Ip = 0;
end
%
% druga polaryzacia
%
FrG = double(Frame(:,:,2));% double(Frame.cdata(:,:,2));
sFrG = circshift( FrG,[sh.shyG sh.shx ] );
FrameG = sFrG;

if ~isscalar( handles.MaskIps )

    %  wyliczamy poziom tla
    stg = FrameG .* handles.IssTlo;
    Isstlo = sum( stg( : ) ) /   sum ( handles.IssTlo( : ) ) ;
    %  Wybieramy obszar Ipp z klatki filmu i przekszta³czmy go w wektor
    FrG    = reshape (  FrameG, 1, 480*640 );
    %  to samo dla k¹tów
    %     Wyrzucamy wszystkie zerowe elementy
    %     odejmujemy t³o oraz uwzgledniamy zaleznoœæ od k¹ta phi
    nzFrG = ( FrG( Ind.s ) - Isstlo )  ./ ( cos( Phi.ss( Ind.s ) ).^2 );
    %   wektor odpowiednich k¹tów theta
    %
    % iloœæ punktów do uœrednienia; zamiast 1200 nale¿a³oby wstawiæ 2*size(...)

    % wybieramy co span_R'ty punkt
%     ind_reduced_G = 1:sh.span_G:length(Ind.s);
    %
     
 Is = zeros(1,length( theta.mTs ) );
 if ivp( 1 ) == 1
     Is( 1 ) =  nzFrG( 1 );
 else
     Is( 1 ) = median( nzFrG(1:ivp( 1 ) - 1 ) );
 end
 

for ii = 2 : length( ivs ) - 2
     Is(ii) = median( nzFrG(ivs(ii)+ 1:ivs(ii+1)-1));
    
end

%     Is_smoo = smooth(theta.nzTs,nzFrG,sh.span_G,'rloess');
%     Is = Is_smoo(ind_reduced_G);

else
    Is = 0;
end


% ------------------------------------------------------------------------

% --- Executes on slider movement.
function slFrame_Callback(hObject, eventdata, handles)
%  slFrame_Callback - pozwalia przewijaæ i przegl¹daæ klatki filmu
%
handles.nom = round( get( handles.slFrame,'Value' ) );
%
% wyœwietla numer bierz¹cej klatki filmu
set( handles.edCarent,'String', num2str( handles.nom ) );
%
% pobiramy odpowiedni frame i rysujemy go
Frame = FrameReader( hObject, handles );
handles.hFrame = imshow( sum(Frame,3),[0,handles.mv] );
%
% przechowujemy bierz¹cy frame w schowku
handles.Fr = Frame;
%
try
    %    jerzeli ustawiona apertura to rysuje niebeskie ku³ko :-))
    [ x,y ] = krug( handles.params(1),handles.params(2),handles.params(3) );
    figure( handles.figure1 );
    handles.hFrame = imshow( Frame,[ 0,max( max( handles.Fr ) ) ] );
    hold on;
    handles.hLKorekt = plot( x,y );
    handles.hmKorekt = plot( handles.params(1),handles.params(2),'color','green','marker','+','markersize',25);
    hold off;
catch
end;
guidata( hObject, handles );

% --- Executes during object creation, after setting all properties.
function slFrame_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in pbBegin.
function pbBegin_Callback(hObject, eventdata, handles)
%
% pbBegin_Callback - ustawia pocz¹tkow¹ klatke filmu z której rozpoczynamy
% przetworzanie danych
nom = round( get( handles.slFrame,'Value' ) );
% wyœwietla numer pocz¹tkowej klatki filmu
set( handles.edBegin,'string',num2str( nom ) );
%  zapisujemy do schowka
handles.segment(1) = nom;
guidata( hObject, handles );

function edBegin_Callback(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
%
function edBegin_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pbEnd.
function pbEnd_Callback(hObject, eventdata, handles)
%
% pbEnd_Callback - ustawia koñcow¹ klatke filmu
nom = round( get( handles.slFrame,'Value' ) );
% wyœwietla numer pocz¹tkowej klatki filmu
set(handles.edEnd,'string',num2str(nom));
%  zapisujemy do schowka
handles.segment(2) = nom;
guidata( hObject, handles );

function edEnd_Callback(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
%
function edEnd_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%
function edCarent_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edCarent_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pbLoad.
function pbLoad_Callback(hObject, eventdata, handles)
%
% pbLoad_Callback - wczytuje film *.avi
[f,dir] = uigetfile( {'*.avi';'*.*'},'Load files','g:\','MultiSelect','on' );
% wyœwietlamy na panelu nazwe filmu
if isstr( f )
    set( handles.uipanel2,'title',f );
    handles.avi_title = f;
    path = [dir f];
elseif iscell( f )
    s = sprintf('%s in %3.1f parts',f{1},size(f,2) );
    set( handles.uipanel2,'title',s );
    handles.avi_title = f(1);
  
    path = [dir f{1}];
elseif f == 0
    set( handles.uipanel2,'title','No files...' );
    return
end
inf = aviinfo(path);
if inf.NumFrames < 250
min_step = floor(inf.NumFrames / 10);
set( handles.edFrameStep, 'string', num2str( min_step ) );
else
    min_step = 250;
    set( handles.edFrameStep, 'string', num2str( min_step ) );
end
    
% ==================== ???? =============================
% if f > 0
% %     jezeli film wczyta³ siê bez b³êdów ustawiamy pocz¹tkowy wartoœci dla
% %     wszystkich pól
% if ~get(handles.RbSeries,'Value') % jeœli seria to nie wygaszamy
%     set(handles.chbIpp,'Value',0,'Enable','off');
%     set(handles.chbIps,'Value',0,'Enable','off');
%     set(handles.pbChangexyr,'Enable','off');
%     set(handles.pbGenTab,'Enable','off');
%     set(handles.pbsetaperture2red,'Enable','off');
%     set(handles.pbsetaperture2green,'Enable','off');
%     set(handles.pbCalc,'Enable','off');
% end
set(handles.pbSummF,'Enable','on');
% set(handles.pbAperture,'Enable','on');
set(handles.pbContrast,'Enable','on');
set(handles.pbView,'Enable','on');
set(handles.pbEnd,'Enable','on');
set(handles.pbBegin,'Enable','on');

% ==================== ???? =============================
%
ainf = aviinfo ( path );
%     ustawiamy zakres sówaka
set( handles.slFrame,'Max',ainf.NumFrames,'Value',1,'sliderstep',[ 1/ainf.NumFrames,10/ainf.NumFrames],'Enable','on');
%     wyœwitlamy numer koñcowej klatki filmu
set( handles.edEnd,'String',num2str( ainf.NumFrames ) );

handles.path = path;
handles.f = f;
handles.dir = dir;
handles.nom  = 1;
%     zakres dla obróbki ustawiamu na ca³y film
handles.segment = [1 ainf.NumFrames];
guidata(hObject, handles);
%  pobiramy i wyœwietlamy pierwszy obraz z filmu
Frame = FrameReader( hObject,handles );
handles.mv = max( max( Frame(:) ) );
%
handles.hFrame = imshow( sum(Frame,3),[0,handles.mv] );
handles.Fr = Frame;
guidata( hObject, handles );
% end

% --- Executes on button press in pbCalc.
function pbCalc_Callback(hObject, eventdata, handles)
% tic
set( handles.upanel, 'Title', '');
%- Sejvovanie parametrow
try
MP.params_R = handles.params_R;
MP.params_G = handles.params_G;
MP.xIpp = handles.xIpp;
MP.yIpp = handles.yIpp;
MP.xIss = handles.xIss;
MP.yIss = handles.yIss;
MP.xIpptlo =handles.xIpptlo;
MP.yIpptlo =handles.yIpptlo;
MP.xIsstlo =handles.xIsstlo;
MP.yIsstlo =handles.yIsstlo;
MP.lens_center = handles.lens_center;

assignin('base','MP',MP);
catch
  warndlg('Nie wszystkie parametry zadane!!'); return  
end
if isstr(handles.f)

fn = [handles.dir handles.f(1:end-3) ];
else
        file = handles.f{1};
    fn = [handles.dir file(1:end-3) ];
end
    
save([fn 'mat'],'MP');

%
[ Phi theta sh Ind] = ThetaWorker(hObject, handles);
% \\ TO DO poprawic katy 
p = diff( theta.nzTp );
[it ivp ]= find( p );  
theta.mTp = theta.nzTp( ivp );
% -------------------------
p = diff( theta.nzTs );
[it ivs ]= find( p );  
theta.mTs = pi - theta.nzTs( ivs );
% -------------------------
   
%  Wyznaczamy rozmiar wektora Ipp iraz Iss 
fstep = str2num( get(handles.edFrameStep2,'string') );
if isstr( handles.f )
    path = [ handles.dir handles.f ];
      inf = aviinfo( path );
    Nom = 1;
     count = length( 1 : fstep : inf.NumFrames );
else
    Nom = size( handles.f, 2 );
    path = [ handles.dir handles.f{ 1 } ];
    inf = aviinfo( path );
    path = [ handles.dir handles.f{ end } ];
    inf2 = aviinfo( path );
    count = ( Nom - 1 ) * length( 1 : fstep : inf.NumFrames ) + length( 1 : fstep : inf2.NumFrames );
    
end
     
    
%
temp = Aviread12_matlab( path, 1 );


[ app bss ] = FrameWorker(hObject, handles, reshape(temp,480,640,3), sh, Phi,theta,Ind,ivs,ivp);%
Ipp = zeros( count ,length( app ) );
Iss = zeros( count ,length( bss ) );

nom = 1;
wb = waitbar(0,'');

for nf = 1 : Nom
    pack;
    try
       waitbar(nf/Nom,wb,...
            ['Processing segment number ' num2str(nf) ' from ' num2str(Nom) ]  );
if isstr( handles.f )
        path = [handles.dir handles.f];
else
        path = [handles.dir handles.f{nf}];
end
    inf =   aviinfo(path);
    for ii = 1 : fstep : inf.NumFrames
    
        temp = Aviread12_matlab( path, ii );

        [Ipp(nom,:) Iss(nom,:)] = FrameWorker(hObject, handles, reshape(temp,480,640,3), sh, Phi,theta,Ind,ivs,ivp);%

        nom = nom + 1;
      
    end
    catch
    s = sprintf(' Frame %3.1f from part %3.1f do not read',ii,nf)
end
end

close(wb);

assignin('base','Ipp',Ipp);
assignin('base','Iss',Iss);
assignin('base','theta',theta);

% --- Executes on button press in chbIpp.
function chbIpp_Callback(hObject, eventdata, handles)
% chbIpp_Callback - ustawiamy maskê dla Ipp
if get( handles.chbIpp,'Value' )

    set( handles.upanel,'Title','Set mask for Ipp');

   Sfr = handles.sFr;

    try
        %          jerzeli ko³o jest zadane wycinamy z frejmu okr¹g³e okienko

        [ x,y ] = krug( handles.params_R(1),handles.params_R(2),handles.params_R(3) );
        bwKrug  = roipoly( Sfr, x, y );
    catch
        bwKrug = ones( size ( handles.sFr ) );
    end
    %         robimy maske z krêgu oraz reszty przez nas wybranej
    
    % maska dla filmu 02022009v5file2_1.00.avi, rozpraszanie
    %xIpp=[81.01613      99.30184      125.2558      138.2327      144.7212      151.7995       158.288      163.0069      164.7765      164.7765      173.6244      180.1129       188.371      204.8871      213.1452      217.2742      215.5046      240.8687      272.7212      269.7719      253.2558      253.2558      264.4631      272.7212      281.5691      287.4677       286.288      280.9793      278.6198      273.3111      272.7212      275.6705      282.7488      282.7488      280.3894      309.8825      319.3203      325.2189      328.7581       358.841      363.5599      370.0484       375.947      381.8456       388.924      396.0023      406.6198      420.7765      424.9055      431.9839       444.371      450.8594      456.7581      462.0668      515.1544      513.9747      508.6659      507.4862      510.4355      513.9747      519.2834      522.2327      529.9009        534.03      536.9793      544.6475      549.3664      547.0069      545.2373      552.3157      559.9839      561.7535      564.1129      573.5507      585.9378      581.8088      581.8088      581.8088      578.2696      571.7811      562.3433      559.9839      559.9839      565.2926      567.6521      569.4217      574.1406      577.0899      571.7811      571.7811      574.7304      578.8594      585.9378      591.2465      590.0668       589.477      582.3986      584.1682      585.9378      591.2465      595.9654      599.5046      605.4032      609.5323      609.5323      607.7627      598.3249      591.2465      591.2465      601.8641      609.5323      617.2005      623.0991      630.1774      634.3065      635.4862      635.4862      630.1774      625.4585      622.5092      616.0207      616.0207      617.7903      623.6889      630.1774      634.8963      634.8963      634.8963      634.8963      634.8963      586.5276      579.4493      577.6797      568.2419      551.7258      526.3618      498.0484      443.1912      407.2097      407.2097      396.5922      397.7719      403.0806      396.0023      391.2834      391.8733      391.8733      391.2834      389.5138      378.8963      377.1267      370.0484      369.4585       371.818      370.0484      366.5092      356.4816      354.1221      353.5323      363.5599      371.2281      374.7673      374.1774      374.1774      374.1774       375.947      382.4355      381.8456      385.9747      388.3341      392.4631      392.4631      390.6935       375.947      323.4493      302.8041      283.9286      276.8502      271.5415      263.8733      259.1544      232.6106      182.4724      156.5184      151.7995      151.7995      155.3387      155.9286      157.1083      155.9286      149.4401      149.4401      152.9793      152.9793      151.2097      148.8502      145.3111      137.6429       124.076        106.97      96.35253      95.76267      95.76267      97.53226      97.53226      108.7396      112.8687      125.2558       137.053      138.8226      138.8226      128.2051      101.0714      93.99309      82.78571      75.11751      66.26959      56.24194      58.01152      57.42166      57.42166      57.42166      59.78111      62.73041      65.08986       72.1682      74.52765       76.8871      75.11751      71.57834      65.08986      62.14055      62.14055      65.67972      67.44931      70.39862      74.52765       76.8871      79.24654      75.70737      72.75806      81.01613];
    %yIpp=[147.1148       158.321      157.6984      154.5856      164.5467      168.9047      163.3016      161.4339      151.4728      147.1148      145.8696      143.3794      143.3794      158.9436      157.6984      157.6984      145.8696      148.3599      152.7179      178.2432      178.8658      187.5817      196.2977      203.7685      203.7685      196.2977       190.072      188.8268       190.072      192.5623      183.8463      172.0175      167.6595      160.1887      151.4728       153.963      154.5856      155.8307      155.8307      152.0953      153.3405      159.5661      159.5661      160.8113      160.8113       153.963      154.5856      158.9436      157.6984      147.1148      151.4728      153.3405      152.7179      149.6051      149.6051      157.0759      168.9047      177.6206      180.1109      180.1109      170.1498      161.4339      161.4339      157.0759      150.2276      150.2276      150.2276      152.7179      160.8113      165.7918      160.8113      153.3405      150.2276      150.2276      151.4728      160.8113      168.9047      180.1109      184.4689      180.1109      172.6401      177.6206      185.0914      186.9591      187.5817      188.2043       185.714      189.4494      191.9397      199.4105      211.8619      215.5973      215.5973       213.107      203.7685      195.0525      188.2043      180.1109      175.1304      175.1304      175.1304      173.8852      176.9981      173.2626      166.4144      163.3016      163.3016      160.8113      153.3405      155.8307      153.3405      154.5856      155.8307      175.1304      183.2237      191.9397      198.7879      197.5428      195.6751      186.3366      186.9591      192.5623      198.7879      199.4105      201.9008      203.7685      209.3716         240.5      259.7996      265.4027      261.6673      267.2704      262.2899       263.535      266.6479      261.0447      262.2899      264.7802      269.1381      257.3093      251.0837      245.4805      238.0097       231.784      233.0292      229.2938      219.9553      216.8424      214.9747      215.5973      206.2588      193.1848      185.0914      178.2432      172.6401      165.1693      165.1693      169.5272      176.9981      186.9591      203.7685      215.5973      224.9358      232.4066      242.3677      253.5739      252.3288      238.6323      238.6323      246.1031      247.9708      252.3288      257.3093      259.7996       263.535       263.535       263.535      252.3288      250.4611      252.3288      262.2899      266.6479      266.0253      266.0253      264.1576       259.177      260.4222      250.4611      243.6128      241.1226      238.0097      233.0292      224.9358      218.7101      214.3521      205.6362      199.4105      193.1848      191.9397      196.9202      205.6362      214.9747      223.0681       231.784      238.0097      247.3482      247.3482      248.5934      247.9708      256.0642      262.9125       263.535      266.6479      266.0253      266.0253      268.5156      273.4961      274.1187       272.251      261.0447      247.3482      239.2549      246.7257      254.1965      257.9319      259.7996      254.8191      245.4805      237.3872      228.6712      219.9553      216.2198      203.7685      186.3366       181.356      180.1109      180.7335      180.1109      170.1498      168.2821      167.6595      147.1148];

    %BW jest zerojedynkow¹ mask¹ rozmiaru Sfr,
    % xIpp, yIpp s¹ wspó³rzêdnymi wierzcho³ków maski
    [BW handles.xIpp handles.yIpp] = roipoly( Sfr );
    handles.MaskIpp = BW .* bwKrug;
    
    set(handles.upanel,'Title','Set background mask for Ipp');

    %   analogicznie maska dla t³a
    %   xpptlo=[49.75346      1.974654      2.564516      2.564516      291.0069      584.7581      640.2051      639.0253      639.6152      639.6152       636.076      632.5369      622.5092      556.4447      495.6889      411.3387      364.1498      356.4816      334.0668      288.0576      244.4078      213.1452      199.5783      145.9009      131.7442      106.3802      81.60599      70.98848      60.96083      53.29263      50.34332      50.93318      49.75346];
    %   ypptlo=[281.5895      282.2121      93.57393       2.05642      3.924125      3.301556      3.924125      85.48054      163.3016      185.0914       167.037      149.6051      125.9475      122.2121      127.1926      129.0603      129.6829      122.2121      124.7023        126.57      111.0058       107.893        126.57      132.7957      138.3988      139.0214      132.7957      132.7957      165.7918      207.5039      243.6128       263.535      281.5895];

     [handles.IppTlo handles.xIpptlo handles.yIpptlo] = roipoly( Sfr );

    set(handles.upanel,'Title',' ');

else

    handles.MaskIpp = 0;

    handles.IppTlo  = 0;

    handles.hFrame  = imshow( handles.sFr, [ min( handles.sFr( : ) )  max( handles.sFr( : ) )  ] );

end

guidata(hObject, handles);


% --- Executes on button press in chbIps.
function chbIps_Callback(hObject, eventdata, handles)

if get(handles.chbIps,'Value')
    set( handles.upanel,'Title','Set mask for Iss');
    Sfr =handles.sFg;
    try
        [ x, y ] = krug( handles.params_G(1),handles.params_G(2),handles.params_G(3) );
        bwKrug = roipoly(  Sfr, x, y );
    catch
        bwKrug = ones( size( handles.sFg ) );
    end

    %         xss=[64.5      183.6521      178.9332      178.9332      178.9332      172.4447      163.0069      160.6475      161.2373      161.2373      160.0576      159.4677      155.9286      151.2097       154.159      156.5184      161.2373      164.7765       171.265      174.8041      173.6244      169.4954      168.3157      173.6244      176.5737      183.0622      183.6521      183.0622      187.7811       196.629      201.3479      203.7074      206.6567      211.9654      215.5046      217.2742       226.712      242.0484      267.4124      282.7488      269.7719      259.7442       252.076      249.7166      240.8687      233.7903       230.841      230.2512      223.1728      221.4032      221.4032      214.9147      207.2465      207.8364      208.4263       209.606      203.1175      194.2696      187.7811      183.0622      183.6521      197.2189      202.5276      207.2465      207.2465      206.6567      203.7074      197.2189      198.9885      206.0668      210.1959       213.735      218.4539      218.4539      219.0438      220.2235      228.4816        234.97        234.97        234.97        234.97      244.9977      249.7166      257.9747       265.053      272.7212      281.5691      286.8779      289.2373      296.9055      298.0853       299.265      301.6244      310.4724      312.2419      308.1129      306.3433      302.2143      301.6244      297.4954      287.4677      283.9286      268.5922      258.5645      253.8456      246.1774      244.9977      250.3065       252.076      249.7166      240.2788      239.6889      235.5599      231.4309      232.0207      238.5092      242.6382      245.5876      254.4355       265.053      266.2327      269.7719      281.5691      289.2373      306.9332      317.5507      339.3756      340.5553      325.2189      319.3203       324.629       333.477      339.3756      339.3756      338.7857      338.1959      339.3756      353.5323      360.6106      361.7903      362.3802      363.5599      364.7396      369.4585      380.6659      379.4862      376.5369      376.5369      374.1774      370.6382      366.5092      364.7396      364.7396      367.0991      372.9977      378.8963      387.7442      386.5645      380.6659      375.3571      367.0991      360.6106      354.1221      351.1728      345.8641      344.6843      344.6843      351.7627      353.5323      360.6106      360.0207      355.8917      356.4816      367.6889      375.3571      386.5645      385.9747      383.0253      383.0253      384.7949      390.6935      394.8226      394.8226      394.8226      413.6982      419.5968      430.2143      440.8318       461.477      460.2972      453.2189      446.7304       444.371      443.7811      443.1912      450.2696      452.0392      457.9378      463.2465      467.3756      471.5046      476.2235      477.4032      532.2604      571.7811      601.2742      625.4585       636.076      631.3571      623.6889      615.4309      604.8134      595.9654      587.7074      578.2696      577.0899      577.0899      582.9885      591.8364      581.2189      575.9101      567.0622      565.8825      565.8825      565.8825      569.4217      565.8825      553.4954      531.0806      531.0806      536.9793      544.6475      552.3157      555.8548       555.265      546.4171        534.03        534.03      530.4908      527.5415      527.5415       525.182      509.8456      513.9747      507.4862      499.2281      492.1498      496.8687      510.4355      515.7442      523.4124      515.1544      508.6659      500.4078      487.4309      477.9931      474.4539      464.4263      456.1682      456.1682      464.4263      469.1452      481.5323      491.5599      500.4078      493.9194      483.3018      464.4263      465.0161      458.5276      431.9839      424.9055      423.7258      416.0576      391.2834      384.7949      383.6152      375.3571      365.3295      361.7903      352.3525       341.735      332.8871      324.0392      318.7304      308.7028      297.4954      286.8779        278.03      251.4862      253.8456      247.3571      238.5092      233.7903      206.0668      190.7304      171.8548      164.1866       141.182      142.9516      151.7995      155.9286       158.288       158.288      152.3894      147.0806      137.6429      131.1544      122.3065      120.5369      128.2051      135.8733      137.6429      138.8226      134.6935      131.7442      122.8963      115.2281      105.7903      96.35253      86.91475      81.01613          64.5];
    %         yss=[314.5856      305.2471      310.2276      318.9436      326.4144       327.037      330.1498      334.5078      343.8463      348.8268      355.0525      360.6556      371.2393      374.3521      379.3327       381.823      381.2004      374.9747      369.9942      361.9008      355.6751      352.5623      348.2043      346.9591      343.2237       341.356      332.6401      327.6595      327.6595      326.4144      320.1887      316.4533      315.2082      315.2082      308.9825      306.4922      306.4922      305.8696      301.5117      301.5117      307.7374      314.5856      320.1887      325.7918      330.1498      332.0175      333.8852      339.4883      346.9591      356.2977      368.1265      371.8619      373.7296      383.6907      386.8035      389.2938      390.5389      391.1615      396.7646      405.4805      415.4416      424.1576       423.535      418.5545      412.9514      412.3288      410.4611      409.8385      403.6128         400.5      399.2549      399.2549      399.2549      403.6128       409.216      411.0837      415.4416      413.5739      404.2354      397.3872      394.2743      394.8969      394.8969      396.7646      398.6323         400.5      401.7451      401.7451      401.7451      401.7451      414.8191      424.7802      431.6284      431.6284      427.2704      411.0837      401.7451      390.5389      384.9358      384.9358      388.6712      389.9163      386.1809      383.6907      381.2004      381.2004      379.3327      371.8619      363.1459      360.6556      361.9008      366.2588      369.9942      369.3716      366.2588      351.3171      346.9591      338.8658      333.2626      328.2821       322.679      317.0759      310.2276      307.1148      332.0175      336.9981      336.9981       341.356       341.356      342.6012      348.2043      358.1654      364.3911       373.107      379.9553      385.5584       391.784      396.7646      397.3872      401.7451      411.7062      421.6673      431.0058      437.8541      439.7218      431.0058      424.1576      416.0642      412.9514      412.9514      411.0837      398.6323      393.6518      388.0486      383.0681      378.0875      369.9942      365.0136      363.1459      363.1459       373.107      373.7296      376.2198      379.3327      381.2004      380.5778       373.107       368.749      361.9008      352.5623      344.4689       341.356      335.7529      331.3949      327.6595      325.1693      327.6595      331.3949       341.356      343.8463      345.0914       341.356      334.5078      329.5272       322.679      308.9825      305.2471      305.2471      300.2665      308.9825       322.679      328.2821      336.3755      343.2237      348.2043      348.2043      346.9591      339.4883      328.9047      323.9241      316.4533      311.4728      302.7568      307.7374      312.0953      310.8502      310.8502      310.2276      333.8852      365.0136       377.465      394.8969      406.7257      413.5739       419.177      416.0642      402.3677      394.2743      381.2004       381.823      388.0486         400.5      407.3482      420.4222      429.1381      444.7023      454.6634      455.9086      457.7763      443.4572      440.9669      440.9669      440.9669      428.5156      421.6673      421.6673       427.893      434.1187      438.4767      445.3249      453.4183       450.928      442.2121      424.7802      408.5934      394.2743      382.4455      379.9553       377.465      379.9553      365.6362      363.1459      363.7685      366.8813      375.5973      369.9942      363.7685      356.2977      360.0331      365.0136      375.5973      383.0681      395.5195      412.3288       427.893      439.0992      440.3444      434.1187      422.9125      422.9125      424.1576      426.0253      434.7412      435.3638      432.8735      430.3833      439.0992      441.5895      441.5895       432.251      431.6284      430.3833      430.3833      433.4961      439.0992      440.9669      440.3444      440.3444      437.2315      435.3638      441.5895      445.3249      445.3249      444.0798      450.3054       450.928      449.6829      447.1926      439.7218      434.1187      429.7607      424.7802      416.6868      411.7062      406.1031      406.1031      398.6323      394.2743      394.2743      399.8774      414.8191       423.535      429.1381      432.8735      434.1187      424.1576      419.7996      422.9125      411.0837      395.5195       377.465      358.1654      314.5856];
    %
    [ BW handles.xIss handles.yIss ]= roipoly( Sfr );
    handles.MaskIps = BW .* bwKrug;
    
    set(handles.upanel,'Title','Set background mask for Iss');
   
    %
    % xsstlo=[6.103687      35.00691      41.49539      39.13594      39.13594      39.72581      52.70276      55.65207      65.67972      78.65668      107.5599      143.5415      194.8594      218.4539      227.8917      257.9747      328.7581      347.0438      376.5369        406.03         448.5      483.8917      519.2834      531.0806      534.6198      544.6475      549.3664      562.3433      580.0392      583.5783      581.2189      577.6797       580.629      588.2972      600.6843       610.712      621.3295      630.1774      633.1267      635.4862      640.2051      639.6152      638.4355      639.6152       593.606      552.3157      421.9562      229.0714      104.6106      40.90553      7.873272      6.693548      6.693548      6.103687];
    % ysstlo=[305.2471      304.6245      304.6245      320.1887      332.0175      336.3755      336.3755      335.7529      352.5623       396.142      435.3638      454.0409      462.1342      460.2665      457.7763      456.5311      463.3794       459.644      465.2471      469.6051      471.4728      472.0953      476.4533      476.4533      468.9825      462.1342      463.3794      463.3794       450.928      444.7023      440.9669      437.2315      431.0058      429.7607      421.6673       404.858      388.6712       368.749      356.2977      336.3755      333.8852      393.0292      444.7023      476.4533      475.2082      477.0759       478.321      475.2082      474.5856      477.0759      477.0759      427.2704      381.2004      305.2471];
    [ handles.IssTlo handles.xIsstlo handles.yIsstlo ] = roipoly( Sfr );
    set( handles.upanel, 'Title', ' ' );

else
    handles.MaskIps = 0;
    handles.IssTlo = 0 ;
    handles.hFrame = imshow( handles.sFg,[ min( handles.sFg(:) )  max( handles.sFg(:) ) ] );

end
set(handles.pbCalc,'Enable','on');
guidata(hObject, handles);



% --- Executes on button press in chbProstuj.
function chbProstuj_Callback(hObject, eventdata, handles)
% chbProstuj_Callback - zmienia tryb otczytu klatki
% wyprostowana czy nie wyprostowana
if get( handles.chbProstuj,'Value' )
    handles.key = 1;
    Frame = FrameReader(hObject,handles);
    handles.hFrame = imshow( Frame, [ 0,handles.mv ] );
    handles.Fr = Frame;
else
    handles.key = 0;
    Frame = FrameReader(hObject,handles);
    handles.hFrame = imshow(Frame,[0,handles.mv]);
    handles.Fr = Frame;

end
try
    %    jerzeli ustawiona apertura to rysuje niebeskie ku³ko :-))

    [ x,y ] = krug( handles.params(1),handles.params(2),handles.params(3) );
    figure( handles.figure1 );
    handles.hFrame = imshow( Frame,[ 0,max( max( handles.Fr ) ) ] );
    hold on;
    handles.hLKorekt = plot( x,y );
    handles.hmKorekt = plot( handles.params(1),handles.params(2),'color','green','marker','+','markersize',25);
    hold off;
catch
end;
guidata(hObject, handles);





% --- Executes on button press in pbSummF.
function pbSummF_Callback(hObject, eventdata, handles)
% pbSummF_Callback - sumuje klatki z wybranego zakresu
%
% //  TO DO avtomatyczna ocena pamieci
%
handles.nom = 1;
% count = handles.segment(1) : handles.segment(2);
% sF = zeros( size( handles.Fr(:,:,1) ) );

% deklaracja macierzy
sFr = zeros( size( handles.Fr(:,:,1) ) );
sFg = sFr;
if iscell( handles.f )
    wb = waitbar(0,'');
    Nom = size( handles.f, 2 );

    for ii = 1 : Nom
        waitbar(ii/Nom,wb,['Processing segment number ' num2str(ii) ' from ' num2str(Nom)]);
        path = [handles.dir handles.f{ii}];
        inf = aviinfo( path );
        count_step = str2num( get(handles.edFrameStep,'string') );%floor( inf.NumFrames / min( 10, inf.NumFrames ) );
        count_min = 1 : count_step :  inf.NumFrames; % TO DO

        if length(count_min) > 150
            h = msgbox('Too many frames !','Error','warn');
            return
        end

        temp = Aviread12_matlab( path, count_min );%% W tym miejscu wprowadzone zmiany aviread na Aviread12_matlab
%         sF = sF + reshape(sum(sum(temp,4),1),480,640);
        sFr = sFr + squeeze( sum(temp(:,:,:,1),1)); % squeeze wyrzuca niewykorzystane wymiary macierzy
        sFg = sFg + squeeze( sum(temp(:,:,:,2),1));
    end
    close(wb);
    
elseif isstr( handles.f )
    path = [ handles.dir handles.f ];
    inf = aviinfo( path );
    count_step = str2num( get( handles.edFrameStep,'string') );
    count_min = 1 : count_step :  inf.NumFrames; % TO DO
    if length(count_min) > 150
        h = msgbox('Too many frames !','Error','warn');
        return
    end
    h = msgbox('Busy','Calculating','warn');
    temp = Aviread12_matlab( path, count_min );
    close( h );
%     sF = sF + reshape(sum(sum(temp,4),1),480,640);
    sFr = sFr + squeeze( sum(temp(:,:,:,1),1));
    sFg = sFg + squeeze( sum(temp(:,:,:,2),1));
end

handles.hFrame = imshow( sFr + sFg , [0,max( max( sFr+sFg ) ) ] ); % wyskalowane do przyzwoitej jasnoœci OK
handles.sFr = sFr;
handles.sFg = sFg;
%
try
    %    je¿eli apertura ustawiona to rysuje niebeskie kó³ko :-))

    [ x,y ] = krug( handles.params(1),handles.params(2),handles.params(3) );
    figure( handles.figure1 );
    handles.hFrame = imshow( Frame,[ 0,max( max( handles.Fr ) ) ] );
    hold on;
    handles.hLKorekt = plot( x,y );
    handles.hmKorekt = plot( handles.params(1),handles.params(2),...
                             'color','green','marker','+','markersize',25);
    hold off;
catch
end;

set(handles.pbsetaperture2red,'Enable','on');
set(handles.pbsetaperture2green,'Enable','on');
guidata( hObject, handles );

% assignin('base','sF',sF);


% --- Executes on button press in pbAperture.
function pbAperture_Callback(hObject, eventdata, handles)

params   = find_Aperture ( handles.Fr );
%
[x,y] = krug(params(1),params(2),params(3));
%
figure( handles.figure1 );
handles.hFrame = imshow(handles.Fr,[0,max(max(handles.Fr))]);
hold on;
handles.hLKorekt = plot( x,y );
handles.hmKorekt = plot(params(1),params(2),'color','green','marker','+','markersize',25);
hold off;

handles.params = params;
guidata( hObject, handles );
set(handles.pbChangexyr,'Enable','on');
set(handles.pbGenTab,'Enable','on');
set(handles.pbsetaperture2red,'Enable','on');
set(handles.pbsetaperture2green,'Enable','on');


% --- Executes on button press in pbChangexyr.
function pbChangexyr_Callback(hObject, eventdata, handles)
[h,nx,ny,nr] = changeXYR(handles);
params(1) = nx;
params(2) = ny;
params(3) = nr;
handles.params = params;
guidata( hObject, handles );
[x,y] = krug(params(1),params(2),params(3));

figure( handles.figure1 );
handles.hFrame = imshow( uint8( 255 * handles.Fr / max( handles.Fr(:) ) ) );
hold on;
handles.hLKorekt = plot(x,y);
handles.hmKorekt = plot(params(1),params(2),'color','green','marker','+','markersize',25);
hold off;
guidata( hObject, handles );


% --- Executes on button press in pbsetaperture2red.
function pbsetaperture2red_Callback(hObject, eventdata, handles)
% hObject    handle to pbsetaperture2red (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
params   = find_Aperture ( handles.sFr );
%
[x,y] = krug(params(1),params(2),params(3));
%
figure( handles.figure1 );
handles.hFrame = imshow(handles.sFr,[0,max(max(handles.sFr))]);
hold on;
handles.hLKorekt = plot( x,y );
handles.hmKorekt = plot(params(1),params(2),'color','green','marker','+','markersize',25);
hold off;



set(handles.pbChangexyr,'Enable','on');
set(handles.pbGenTab,'Enable','on');
set(handles.pbGetParamsRed,'Enable','on');


handles.Fr = handles.sFr;
handles.params = params;
guidata( hObject, handles );


% --- Executes on button press in pbsetaperture2green.
function pbsetaperture2green_Callback(hObject, eventdata, handles)
% hObject    handle to pbsetaperture2green (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
params   = find_Aperture ( handles.sFg );
%
[x,y] = krug(params(1),params(2),params(3));
%
figure( handles.figure1 );
handles.hFrame = imshow(handles.sFg,[0,max(max(handles.sFg))]);
hold on;
handles.hLKorekt = plot( x,y );
handles.hmKorekt = plot(params(1),params(2),'color','green','marker','+','markersize',25);
hold off;



set(handles.pbChangexyr,'Enable','on');
set(handles.pbGenTab,'Enable','on');
set(handles.pbGetParamsGreen,'Enable','on');

handles.Fr = handles.sFg;
handles.params = params;
guidata( hObject, handles );


% --- Executes on button press in pbView.
function pbContrast_Callback(hObject, eventdata, handles)
imcontrast(handles.figure1);
% --- Executes on button press in pbView.

function pbView_Callback(hObject, eventdata, handles)

[x,y] = krug(handles.params(1),handles.params(2),handles.params(3));

figure(handles.figure1);
imtool(handles.Fr,[0,max(max(handles.Fr))]);
try
    hold on;
    handles.hLKorekt = plot(x,y);
    handles.hmKorekt = plot(handles.params(1),handles.params(2),'color','green','marker','+','markersize',25);
    hold off;
catch
end


% --- Executes on button press in pbGenTab.
function [ handles ] =  pbGenTab_Callback(hObject, eventdata, handles)

    handles.GenTabErr = 0;
try
    handles.params;

    r_R =  (handles.params_R(3));% promieñ w pikselach,
    r_G =  (handles.params_G(3));
    % lista œrednic diafragmy w mm
    stDif{1} = '8.63';
    stDif{2} = '9.82';
    stDif{3} = '10';
    stDif{4} = '10.16';
    stDif{5} = '10.2';
    [s,v] = listdlg('PromptString','Choose diafragm:','SelectionMode','single','ListString',stDif,'InitialValue',5);
    handles.Diafragma = str2num(stDif{s})*1e-3; % przeliczenie na metry
    handles.alpha_0_max = atan( handles.Diafragma/( 2*16.86e-3 ) );
    hccd_max_R = r_R*9.36e-6; % przeliczenie na metry
    hccd_max_G = r_G*9.36e-6;
    out.FileName = handles.avi_title;
    out.hccd_max_R = hccd_max_R;
    out.hccd_max_G = hccd_max_G;
    out.Diafragma = handles.Diafragma;
    assignin('base','setup',out);


    lambda(1) = 654.25;
    lambda(2) = 532.07;
    handles.lambda = lambda;
    guidata(hObject, handles);
    koefR = gentab_angle(handles.Diafragma,hccd_max_R,lambda(1));
    koefG = gentab_angle(handles.Diafragma,hccd_max_G,lambda(2));
    handles.koefR = koefR;
    handles.koefG = koefG;

    % [T1,T2] = Konstruowanie_odfitt21(r,koef);
    % handles.T1 = T1;
    % handles.T2 = T2;
    handles.koef = koefR;
    handles.params = handles.params_R;
    TF_frameR = kat_maker( handles,lambda(1) );

    handles.koef = koefG;
    handles.params = handles.params_G;
    TF_frameG = kat_maker( handles,lambda(2) );

    handles.TF_frameR = TF_frameR;
    handles.TF_frameG = TF_frameG;

    guidata(hObject, handles);
    set(handles.chbIpp,'Enable','on');
    set(handles.chbIps,'Enable','on');
    % set(handles. chbProstuj,'Enable','on');
catch
    hed = warndlg('Cos jest nie dobrze');
    uiwait(hed);
    handles.GenTabErr = 1;
end

% --- Executes on slider movement.
function slPr_Callback(hObject, eventdata, handles)
nom = round( get( handles.slPr,'Value' ) );
imshow( sum(handles.Fr,3),[0,mean(max(sum(handles.Fr,3))) ]);
hold on;
plot([nom nom],[1 480],'r');
hold off;
assignin('base','lens_center',nom);
handles.lens_center = nom;
guidata( hObject, handles );

%
% pobiramy odpowiedni frame i rysujemy go

%
% przechowujemy bierz¹cy frame w schowku



% --- Executes during object creation, after setting all properties.
function slPr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slPr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on button press in RbSeries.
function RbSeries_Callback(hObject, eventdata, handles)
% hObject    handle to RbSeries (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of RbSeries





function edFrameStep_Callback(hObject, eventdata, handles)
% hObject    handle to edFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edFrameStep as text
%        str2double(get(hObject,'String')) returns contents of edFrameStep as a double


% --- Executes during object creation, after setting all properties.
function edFrameStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function edFrameStep2_Callback(hObject, eventdata, handles)
% hObject    handle to edFrameStep2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edFrameStep2 as text
%        str2double(get(hObject,'String')) returns contents of edFrameStep2 as a double


% --- Executes during object creation, after setting all properties.
function edFrameStep2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edFrameStep2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in pbShowMascs.
function [ handles ] = pbShowMascs_Callback(hObject, eventdata, handles)
   Sfr = uint8( 255 * ( handles.sFr -  min( handles.sFr(:) ) ) / max( handles.sFr(:) ) );
   
        [ x,y ] = krug( handles.params_R(1), handles.params_R(2), handles.params_R(3) );
        [bwKrug]  = roipoly( Sfr, x, y );
    [ xg,yg ] = krug( handles.params_G(1), handles.params_G(2), handles.params_G(3) );
        

        imshow( Sfr );
        hold on;
       hl = plot(x,y,'r');
       hlg = plot(xg,yg,'g');
        hold off;
        [BW xi yi ]= roipoly(Sfr, x, y );
%  Ustawiamy maske
set( handles.upanel, 'Title', 'Set mask for Ipp');
        hp = impoly(gca,[handles.xIpp,handles.yIpp]);
        pos = wait( hp );
        delete( hp );
        [BW handles.xIpp handles.yIpp] = roipoly( Sfr,pos(:,1),pos(:,2) );
        handles.MaskIpp = BW .* bwKrug;
%  ustawiamy tlo    
set( handles.upanel, 'Title', 'Set background for Ipp');
hp = impoly(gca,[handles.xIpptlo handles.yIpptlo]);
        pos = wait( hp );
         delete( hp );
        [handles.IppTlo handles.xIpptlo handles.yIpptlo] = roipoly( Sfr,pos(:,1),pos(:,2) );
 set(handles.chbIpp,'Enable','on');
 set(handles.chbIpp,'Value',1);  

 Sfr = uint8( 255 * ( handles.sFg -  min( handles.sFg(:) ) ) / max( handles.sFg(:) ) );
 [ x,y ] = krug( handles.params_R(1), handles.params_R(2), handles.params_R(3) );
        [bwKrug]  = roipoly( Sfr, x, y );
    [ xg,yg ] = krug( handles.params_G(1), handles.params_G(2), handles.params_G(3) );
        

        imshow( Sfr );
        hold on;
       hl = plot(x,y,'r');
       hlg = plot(xg,yg,'g');
        hold off;
        [BW xi yi ]= roipoly(Sfr, x, y );
%  Ustawiamy maske
set( handles.upanel, 'Title', 'Set mask for Iss');
        hp = impoly(gca,[handles.xIss,handles.yIss]);
        pos = wait( hp );
        delete( hp );
        [BW handles.xIss handles.yIss] = roipoly( Sfr,pos(:,1),pos(:,2) );
        handles.MaskIps = BW .* bwKrug;
%  ustawiamy tlo       
set( handles.upanel, 'Title', 'Set background for Iss');
hp = impoly(gca,[handles.xIsstlo handles.yIsstlo]);
        pos = wait( hp );
         delete( hp );
        [handles.IssTlo handles.xIsstlo handles.yIsstlo] = roipoly( Sfr,pos(:,1),pos(:,2) );
        delete(  hl );
        delete(hlg);
set(handles.chbIps,'Enable','on'); 
set(handles.chbIps,'Value',1); 
set( handles.upanel, 'Title', '');
guidata( hObject, handles );
               


% --- Executes on button press in pbSaveParam.
function pbSaveParam_Callback(hObject, eventdata, handles)
% hObject    handle to pbSaveParam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
MP.params_R = handles.params_R;
MP.params_G = handles.params_G;
MP.xIpp =handles.xIpp;
MP.yIpp =handles.yIpp;
MP.xIss = handles.xIss;
MP.yIss = handles.yIss;
MP.xIpptlo =handles.xIpptlo;
MP.yIpptlo =handles.yIpptlo;
MP.xIsstlo =handles.xIsstlo;
MP.yIsstlo =handles.yIsstlo;
MP.lens_center = handles.lens_center;

% MP.params = handles.params;
assignin('base','MP',MP);



% --- Executes on button press in pbLoadParam.
function pbLoadParam_Callback(hObject, eventdata, handles)
% hObject    handle to pbLoadParam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
MP = evalin('base', 'MP');
handles.params = MP.params_R;
handles.Fr = handles.sFr;
handles.hLKorekt = plot([0 0],[0 , 0]);
handles.hmKorekt = plot([0 0],[0 , 0]);
% pbsetaperture2red_Callback(hObject, eventdata, handles);
set( handles.upanel, 'Title', 'Set RED field-of-view');
pbChangexyr_Callback(hObject, eventdata, handles);
handles.params_R = handles.params;

handles.params = MP.params_G;
handles.Fr = handles.sFg;
set( handles.upanel, 'Title', 'Set GREEN field-of-view');
pbChangexyr_Callback(hObject, eventdata, handles);
handles.params_G = handles.params;

% wspó³rzêdne wierzcho³ków masek
handles.xIpp = MP.xIpp;
handles.yIpp = MP.yIpp;
handles.xIss = MP.xIss;
handles.yIss = MP.yIss;

handles.xIpptlo = MP.xIpptlo;
handles.yIpptlo = MP.yIpptlo;
handles.xIsstlo = MP.xIsstlo;
handles.yIsstlo = MP.yIsstlo;

handles.lens_center = MP.lens_center;
% guidata( hObject, handles );

handles.GenTabErr = 1;
while  handles.GenTabErr
[ handles ] =  pbGenTab_Callback(hObject, eventdata, handles);
end
[ handles ] = pbShowMascs_Callback(hObject, eventdata, handles);

set( handles.upanel, 'Title', 'Set lens center');
handles.lens_center = round( get( handles.slPr,'Value' ) );

assignin('base','circleR',handles.params_R);
assignin('base','circleG',handles.params_G);
assignin('base','lens_center',handles.lens_center);
set(handles.pbChangexyr,'Enable','on');
set(handles.pbGenTab,'Enable','on');
set(handles.pbsetaperture2red,'Enable','on');
set(handles.pbsetaperture2green,'Enable','on');
set(handles.chbIpp,'Enable','on');
set(handles.chbIps,'Enable','on');
set(handles.pbCalc,'Enable','on');
guidata( hObject, handles );
% warndlg('Remember Gen Tab !!! ','!! Warning !!');


% --- Executes on button press in pbGetParamsRed.
function pbGetParamsRed_Callback(hObject, eventdata, handles)
% hObject    handle to pbGetParamsRed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.params_R = handles.params;
guidata( hObject, handles );
assignin('base','circleR',handles.params_R);

% --- Executes on button press in pbGetParamsGreen.
function pbGetParamsGreen_Callback(hObject, eventdata, handles)
% hObject    handle to pbGetParamsGreen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.params_G = handles.params;
guidata( hObject, handles );
assignin('base','circleG',handles.params_G);
