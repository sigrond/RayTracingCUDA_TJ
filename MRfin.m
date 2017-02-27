function varargout = MRfin(varargin)
% MRFIN M-file for MRfin.fig
%      MRFIN, by itself, creates a new MRFIN or raises the existing
%      singleton*.
%
%      H = MRFIN returns the handle to a new MRFIN or the handle to
%      the existing singleton*.
%
%      MRFIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MRFIN.M with the given input arguments.
%
%      MRFIN('Property','Value',...) creates a new MRFIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MRfin_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MRfin_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MRfin

% Last Modified by GUIDE v2.5 11-Dec-2013 12:29:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MRfin_OpeningFcn, ...
    'gui_OutputFcn',  @MRfin_OutputFcn, ...
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
%==============My fynctions==============================
function rad = deg2rad( deg )
rad = pi*deg/180;
%-----
function handles = Old_data(handles)
% Reading data from "base" workspase
    handles.Ipp = evalin('base','Ipp');
    handles.Iss = evalin('base','Iss');
    handles.theta = evalin('base','theta');
    handles.setup = evalin('base','setup');
    
    set( handles.edFrame_End,'string', num2str( size( handles.Ipp,1 ) ) );
    set( handles.edFrame_Step,'string', '100' );
% Parameters of  the laser beams
    handles.Wr.wavelength = 654.25;
    handles.Wr.theta = 0;
    handles.Wr.polarization = 0;

    handles.Wg.wavelength = 532.07;
    handles.Wg.polarization = 1;
    handles.Wg.theta = pi;

handles.Tp = atan( tan( handles.theta.mTp - pi / 2 ) *...
                   str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
                   deg2rad( str2num( get( handles.edShift_R,'string' ) ) );
handles.Ts = atan( tan( handles.theta.mTs - pi / 2 ) *...
                   str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
                   deg2rad( str2num( get( handles.edShift_G,'string' ) ) );
handles.rrp = ( running_radius( abs( handles.Tp - pi/2 ),...
                handles.setup.hccd_max_R, handles.setup.Diafragma, handles.Wr.wavelength ) ) .^ 2;
handles.rrs = ( running_radius(abs(handles.Ts-pi/2),...
                handles.setup.hccd_max_G, handles.setup.Diafragma, handles.Wg.wavelength ) ).^2;

handles.mr = Calculate_m(23,handles.Wr.wavelength,'EG');
handles.mg = Calculate_m(23,handles.Wg.wavelength,'EG');
handles.r = 1e3:20:15e3;
handles.ind = 1:100:size( handles.Ipp,1 );
set(handles.te_m_red,'string',['m_r = ' num2str( handles.mr )]);
set(handles.te_m_green,'string',['m_g = ' num2str( handles.mg )]);
set(handles.uipanel1,'title',handles.setup.FileName);
%-----
%==============My fynctions==============================

% --- Executes just before MRfin is made visible.
function MRfin_OpeningFcn(hObject, eventdata, handles, varargin)

handles.C=1;   % Kontola tego czy ma wykonywac sie kod C, C=1 TAK, C=0 NIE

handles.output = hObject;

W = evalin('base','who');

if ( ismember('Ipp', W)*...
     ismember('Iss', W)*...
     ismember('setup', W)*...
     ismember('theta', W) ) == 1 % The old data is detected

try
    handles.Ipp = evalin('base','Ipp');
    handles.Iss = evalin('base','Iss');
    handles.theta = evalin('base','theta');
    set( handles.edFrame_End,'string', num2str( size( handles.Ipp,1 ) ) );
    set( handles.edFrame_Step,'string', '100' );
catch
    s = sprintf('Ipp Iss not found \nin Base Workspace');
    he = warndlg( s );
    uiwait( he );
end

% if ~handles.C
%     if (matlabpool('size')~=4 & matlabpool('size')>0)
%         matlabpool close;
%         matlabpool 4;
%     end
%     if(matlabpool('size')==0)
%         matlabpool 4;
%     end
% end

try
    handles.setup = evalin('base','setup');
catch
    s = sprintf('Setup not found \nin Base Workspace')
    he = warndlg( s );
    uiwait( he );
end
handles.Wr.wavelength = 654.25;
handles.Wr.theta = 0;
handles.Wr.polarization = 0;

handles.Wg.wavelength = 532.07;
handles.Wg.polarization = 1;
handles.Wg.theta = pi;





    handles.Tp = atan( tan( handles.theta.mTp - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_R,'string' ) ) );
    handles.Ts = atan( tan( handles.theta.mTs - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_G,'string' ) ) );
    handles.rrp = ( running_radius( abs( handles.Tp - pi/2 ),...
        handles.setup.hccd_max_R, handles.setup.Diafragma, handles.Wr.wavelength ) ) .^ 2;
    handles.rrs = ( running_radius(abs(handles.Ts-pi/2),...
        handles.setup.hccd_max_G, handles.setup.Diafragma, handles.Wg.wavelength ) ).^2;

handles.mr = Calculate_m(25,handles.Wr.wavelength,'EG');
handles.mg = Calculate_m(25,handles.Wg.wavelength,'EG');


handles.mr2 = Calculate_m(25,handles.Wr.wavelength,'EG');
handles.mg2 = Calculate_m(25,handles.Wg.wavelength,'EG');

handles.r = 1e3:20:15e3;
handles.ind = 1:100:size( handles.Ipp,1 );
set(handles.te_m_red,'string',['m_r = ' num2str( handles.mr )]);
set(handles.te_m_green,'string',['m_g = ' num2str( handles.mg )]);

set(handles.te_m_red2,'string',['m_r = ' num2str( handles.mr2 )]);
set(handles.te_m_green2,'string',['m_g = ' num2str( handles.mg2 )]);

set(handles.uipanel1,'title',handles.setup.FileName);


    handles = Old_data(handles);
elseif ( ismember('I_R',W) ||...
         ismember('I_G',W) ||...
         ismember('I_B',W) ) % The new data is detected
     
     if ismember('IppConv',W)
         handles.Ipp = evalin('base','IppConv');
         handles.Iss = evalin('base','IssConv');
         out = evalin('base','out');
     else
         out = New_Old_Data_Converter;
     end
    
    
   
   handles.theta = out.theta;
   handles.Tp = out.theta.mTp;
   handles.Ts = out.theta.mTs;
   % Parameters of  the laser beams
   handles.Wr = out.Wr;
   handles.Wg = out.Wg;
   handles.mr = Calculate_m(23,handles.Wr.wavelength,'EG');
   handles.mg = Calculate_m(23,handles.Wg.wavelength,'EG');
   
   
   
   handles.setup.FileName  = 'FN';
   handles.setup.hccd_max_R = 0.0026;
   handles.setup.hccd_max_G = 0.0027;
   handles.setup.Diafragma = 0.0098;
   handles.rrp = ( running_radius( abs( handles.Tp - pi/2 ),...
                handles.setup.hccd_max_R, handles.setup.Diafragma, handles.Wr.wavelength ) ) .^ 2;
   handles.rrs = ( running_radius(abs(handles.Ts-pi/2),...
                handles.setup.hccd_max_G, handles.setup.Diafragma, handles.Wg.wavelength ) ).^2;
            %% temp 
            indNan = find(isnan(handles.rrp));
            handles.rrp(indNan) = (handles.rrp(indNan-1)+handles.rrp(indNan+1))/2;
            indNan = find(isnan(handles.rrs));
            handles.rrs(indNan) = (handles.rrs(indNan-1)+handles.rrs(indNan+1))/2;
            %% end temp
   
if ~ismember('IppConv',W)
    wb = waitbar(0,'Calculating...');
   handles.Ipp = zeros(size(out.Ipp));
   handles.Iss = zeros(size(out.Iss));

   for in = 1 : size(out.Ipp,1)         
       waitbar(in/size(out.Ipp,1),wb);
       handles.Ipp(in,:) = out.Ipp(in,:);%./handles.rrp;
       handles.Iss(in,:) = out.Iss(in,:);%./handles.rrs;
   end
   close(wb);
   assignin('base','IppConv',handles.Ipp);
   assignin('base','IssConv',handles.Iss);
   assignin('base','theta',handles.theta );
   assignin('base','setup',handles.setup );
   assignin('base','out',out );
end

   set(handles.te_m_red,'string',['m_r = ' num2str( handles.mr )]);
   set(handles.te_m_green,'string',['m_g = ' num2str( handles.mg )]);
   set( handles.edFrame_End,'string', num2str( size( handles.Ipp,1 ) ) );
   set( handles.edFrame_Step,'string', '100' );
   handles.r = 1e3:20:15e3;
   handles.ind = 1:100:size( handles.Ipp,1 );
%    assignin('base','setup',setup)

else
    s = sprintf('There is no appropriate data in the "base" workspace!');
    he = warndlg( s );
    uiwait( he );
    
end

if handles.C
    save_workspace;         % KOD C
end

% Update handles structure
guidata(hObject, handles);


function varargout = MRfin_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in pmRefInd.
function pmRefInd_Callback(hObject, eventdata, handles)
% Calculate refractive index
S = get( handles.pmRefInd,'String' );
Vel = get( handles.pmRefInd,'Value' );
handles.mr = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );
set( handles.te_m_red,'string',['m_r = ' num2str(handles.mr)]);
set( handles.te_m_green,'string',['m_g = ' num2str(handles.mg)]);
% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function pmRefInd_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edRmin_Callback(hObject, eventdata, handles)
handles.r = str2num( get( handles.edRmin,'string' ) ):...
    str2num( get( handles.edRstep,'string' ) ):...
    str2num( get( handles.edRmax,'string' ) );
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edRmin_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edRmax_Callback(hObject, eventdata, handles)
handles.r = str2num( get( handles.edRmin,'string' ) ):...
    str2num( get( handles.edRstep,'string' ) ):...
    str2num( get( handles.edRmax,'string' ) );
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.

function edRmax_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edRstep_Callback(hObject, eventdata, handles)
handles.r = str2num( get( handles.edRmin,'string' ) ):...
    str2num( get( handles.edRstep,'string' ) ):...
    str2num( get( handles.edRmax,'string' ) );
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edRstep_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edScale_Callback(hObject, eventdata, handles)

    handles.Tp = atan( tan( handles.theta.mTp - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_R,'string' ) ) );
    handles.Ts = atan( tan( handles.theta.mTs - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_G,'string' ) ) );

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edScale_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edShift_R_Callback(hObject, eventdata, handles)

    handles.Tp = atan( tan( handles.theta.mTp - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_R,'string' ) ) );

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edShift_R_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edShift_G_Callback(hObject, eventdata, handles)

    handles.Ts = atan( tan( handles.theta.mTs - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_G,'string' ) ) );

guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of edShift_G as text
%        str2double(get(hObject,'String')) returns contents of edShift_G as a double


% --- Executes during object creation, after setting all properties.
function edShift_G_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edShift_m_Callback(hObject, eventdata, handles)
S = get( handles.pmRefInd,'String' );
Vel = get( handles.pmRefInd,'Value' );
handles.mr = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );

set( handles.te_m_red,'string',sprintf('m_r = %2.4f ',  handles.mr ));

set( handles.te_m_green,'string',sprintf('m_g = %2.4f ',  handles.mg  ) );
guidata(hObject, handles);


function edShift_m_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edShift_mred_Callback(hObject, eventdata, handles)
S = get( handles.pmRefInd,'String' );
Vel = get( handles.pmRefInd,'Value' );
handles.mr = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
set( handles.te_m_red,'string',sprintf('m_r = %2.4f ',  handles.mr ));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edShift_mred_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edFrame_Step_Callback(hObject, eventdata, handles)
handles.ind =  str2num( get( handles.edFrame_begin,'string' ) ):...
    str2num( get( handles.edFrame_Step,'string' ) ):...
    str2num( get( handles.edFrame_End,'string' ) );
if handles.C
    save_workspace;         % KOD C
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edFrame_Step_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edFrame_begin_Callback(hObject, eventdata, handles)
handles.ind =  str2num( get( handles.edFrame_begin,'string' ) ):...
    str2num( get( handles.edFrame_Step,'string' ) ):...
    str2num( get( handles.edFrame_End,'string' ) );
if handles.C
    save_workspace;         % KOD C
end
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edFrame_begin_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edFrame_End_Callback(hObject, eventdata, handles)
handles.ind =  str2num( get( handles.edFrame_begin,'string' ) ):...
    str2num( get( handles.edFrame_Step,'string' ) ):...
    str2num( get( handles.edFrame_End,'string' ) );
if handles.C
    save_workspace;         % KOD C
end
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function edFrame_End_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbCalc.
function pbCalc_Callback(hObject, eventdata, handles)
% ======= Calc ==========================
handles.C = 1;
if handles.C
    
    externalM = get(handles.ch2refInd,'Value');
    
    save_setup; %KOD DO C
    pause(0.1);
    TT=tic;
    noRunningRadius=0;
    [status,result] = system(sprintf('client.exe %d',noRunningRadius)); %KOD DO C
    if size(strfind(result,'error'),1)~=0
        [status,result] = system(sprintf('START mrfin_gtx680.exe'))
        [status,result] = system(sprintf('client.exe %d',noRunningRadius))
    end
    czas=toc(TT)
end

if ~handles.C
    TTTT=tic;
    ppp=tic;
    
    Ittp = GeneratePattern( handles.r, handles.mr,handles.Tp, handles.Wr );
    Itts = GeneratePattern( handles.r, handles.mg, handles.Ts, handles.Wg );
    
    %m_red = maxGarnett(handles.r);                                                  % !!!!!!!!!!!
    %Ittp = GeneratePattern( handles.r,m_red,handles.Tp,handles.Wr );                % !!!!!!!!!!!

    %m_green = maxGarnett(handles.r);                                                % !!!!!!!!!!!
    %Itts = GeneratePattern( handles.r, m_green, handles.Ts, handles.Wg );           % !!!!!!!!!!!
    
    CzasGenerate=toc(ppp)
    
    %-------------------------------------------------------------------------
    Ittp = Ittp ./ meshgrid( handles.rrp, 1 : size( Ittp, 1 ) ); % unormowane przez running radius
    Itts = Itts ./ meshgrid( handles.rrs, 1 : size( Itts, 1 ) );
    %[uv] = memory;
    %step = floor( uv.MaxPossibleArrayBytes / ( size( Ittp, 1 ) * 64 ) );
    step=500;
    
    if size( handles.Ipp( handles.ind,: ),1 ) > step
        N = floor( size( handles.Ipp( handles.ind,: ),1 ) / step );
    else
        N = 1;
        step = size( handles.Ipp( handles.ind,: ),1 );
    end
    
    res.r = zeros( 1, size( handles.Ipp( handles.ind,: ),1 ) );
    res.rr = res.r;
    res.rg = res.r;
    
    
    CCC=tic;
    for p = 0 : N - 1
        vek = ( ( p * step ) + 1 ) : ( ( p + 1 )* step );
        
        
        
        [ errp, ~] = ReferenceDistance( handles.Ipp( handles.ind( vek ), : ), Ittp );
        [~, irmp] = min( errp, [], 2 );
       
        [ errs, ~] = ReferenceDistance( handles.Iss(handles.ind( vek ),:), Itts );
        [~, irms] = min(errs, [], 2);
        
        %-------------------------------------------------------------------------
        err = ( errp.' ./ meshgrid( mean( errp, 2 ), 1 : size( errp, 2 ) ) ).* ...
            ( errs.' ./ meshgrid( mean( errs, 2 ), 1 : size( errs, 2 ) ) );
        
        [v irm] = min( err.', [], 2 ); % v to vektor warto�ci, irm to vektor indeks�w
        res.rr( vek ) = handles.r( irmp );
        res.rg( vek ) = handles.r( irms );
        res.r( vek ) = handles.r( irm  );
    end
    CzasDystansu=toc(CCC)
    
    
    if N*step < size( handles.Ipp( handles.ind,: ),1 )
        vek = N*step:size( handles.Ipp( handles.ind,: ),1 );
        [ errp, scalep ] = ReferenceDistance( handles.Ipp( handles.ind( vek ), : ), Ittp );
        [vp irmp] = min( errp, [], 2 ); % to jest podgl�d min dla pierwszej polaryzacji
        clear('scalep','vp');
        
        [ errs, scales ] = ReferenceDistance( handles.Iss(handles.ind( vek ),:), Itts );
        [vp irms] = min(errs, [], 2);
        clear('scales','vp');
        
        %-------------------------------------------------------------------------
        err = ( errp.' ./ meshgrid( median( errp, 2 ), 1 : size( errp, 2 ) ) ).* ...
            ( errs.' ./ meshgrid( median( errs, 2 ), 1 : size( errs, 2 ) ) );
        
        [v irm] = min( err.', [], 2 ); % v to vektor warto�ci, irm to vektor indeks�w
        res.rr( vek ) = handles.r( irmp );
        res.rg( vek ) = handles.r( irms );
        res.r( vek ) = handles.r( irm  );
        res.mr = handles.mr;
        res.mg = handles.mg;
        
        
    end
    
end
if handles.C
   import_results;
end

% ====== Create time vector ===============

fps = str2num( get( handles.edFps,'string' ) );
frame_step = str2num( get( handles.edFrame_Step,'string' ) );
dt = frame_step / fps;
rNum = length(handles.ind);
time = [0:dt:(rNum-1)*dt]';

handles.time = time;
guidata(hObject,handles);

% ======= Draw ==========================
hf =  figure;

handles.current_figure = hf;
guidata(hObject, handles);

axes;
if get(handles.cbRed,'value')
    plot( time, res.rr ,'r.', 'MarkerSize', 6 );
    grid on;
end
%=================== fitt preview ====================================
if 1==0
    for ii = 1:size(handles.Ipp(handles.ind,:),1);
        plot( time, handles.Tp,handles.Ipp( ii,: ) );%,theta.mTs(nom),Iss(ii,nom));grid on;
        hold on; grid on;
        plot(time, handles.Tp, Ittp( irmp( ii ), : )* scalep( ii,irmp( ii ) ) ,'r');
        hold off;
        pause( 0.1 );
    end
end
%=================== fitt preview ====================================
if get(handles.cbGreen,'value')
    figure( hf );
    hold on;
    plot( time, res.rg, 'g.', 'MarkerSize', 6);
    grid on;
end
s = sprintf(['Scale = %s \n'...
    'Shift R = %s \n'...
    'Shift G = %s \n'...
    'Shift m = %s\n'...
    'mr = %2.4f \n'...
    'mg = %2.4f \n'...
    'r(1) = %2.1f \n'...
    'r(end) = %2.1f'],...
    get( handles.edScale,'string' ),...
    get( handles.edShift_R,'string' ),...
    get( handles.edShift_G,'string' ),...
    get( handles.edShift_m, 'string' ),...
    handles.mr, handles.mg,...
    res.rg(1) , res.rg(end) );
       % handles.r( irms(1) ),handles.r( irms(end) ) );
legend(s);
%=================== fitt preview ====================================
if 1==0
    for ii = 1:size(handles.Iss(handles.ind,:),1);
        plot( time, handles.Ts,handles.Iss( ii,: ) );%,theta.mTs(nom),Iss(ii,nom));grid on;
        hold on; grid on;
        plot(time, handles.Ts, Itts( irms( ii ), : )* scales( ii,irms( ii ) ) ,'r');
        hold off;
        pause( 0.5 );
    end
end
%=================== fitt preview ====================================
if get(handles.cbBlue,'value')
    figure( hf );
    figure(hf);
    plot( time, res.r,'.b' );
    grid on;
    hold off;
end
assignin('base','results',res);
if ~handles.C
    czas=toc(TTTT)
end
% --- Executes on button press in pbLoadParam.
function pbLoadParam_Callback(hObject, eventdata, handles)

P = evalin('base', 'parameters');
set( handles.edShift_mred,'string',P.Angle_Lim );
set( handles.edShift_m,'string',P.Shift_m );
set( handles.edShift_G,'string',P.Shift_G );
set( handles.edShift_R,'string',P.Shift_R );
set( handles.edScale,'string',P.Scale );
set( handles.edRstep,'string',P.Rstep );
set( handles.edRmax,'string',P.Rmax );
set( handles.edRmin,'string',P.Rmin );
set( handles.edFrame_Step,'string',P.Frame_Step );
set( handles.edFrame_End,'string',P.Frame_End );
set( handles.edFrame_begin,'string',P.Frame_begin );
set( handles.pmRefInd,'Value', P.Vel );
set(handles.uipanel1,'title',handles.setup.FileName);


    handles.Tp = atan( tan( handles.theta.mTp - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_R,'string' ) ) );
    handles.Ts = atan( tan( handles.theta.mTs - pi / 2 ) *...
        str2num( get( handles.edScale,'string' ) ) ) + pi/2 +...
        deg2rad( str2num( get( handles.edShift_G,'string' ) ) );
S = get( handles.pmRefInd,'String' );
Vel = get( handles.pmRefInd,'Value' );

handles.mr = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );


set( handles.te_m_red,'string',sprintf('m_r = %2.4f ',  handles.mr ));

set( handles.te_m_green,'string',sprintf('m_g = %2.4f ',  handles.mg  ) );

handles.r = str2num( get( handles.edRmin,'string' ) ):...
    str2num( get( handles.edRstep,'string' ) ):...
    str2num( get( handles.edRmax,'string' ) );
handles.ind =  str2num( get( handles.edFrame_begin,'string' ) ):...
    str2num( get( handles.edFrame_Step,'string' ) ):...
    str2num( get( handles.edFrame_End,'string' ) );
handles.rrp = ( running_radius( abs( handles.Tp - pi/2 ),...
    handles.setup.hccd_max_R, handles.setup.Diafragma, handles.Wr.wavelength ) ) .^ 2;
handles.rrs = ( running_radius(abs(handles.Ts-pi/2),...
    handles.setup.hccd_max_G, handles.setup.Diafragma, handles.Wg.wavelength ) ) .^ 2;
if handles.C
    save_workspace;         % KOD C
end
guidata(hObject, handles);

% --- Executes on button press in pbSaveParam.
function pbSaveParam_Callback(hObject, eventdata, handles)

P.Angle_Lim =  get( handles.edShift_mred,'string') ;
P.Shift_m =  get(handles.edShift_m,'string') ;
P.Shift_G =  get(handles.edShift_G,'string') ;
P.Shift_R =  get(handles.edShift_R,'string') ;
P.Scale =  get(handles.edScale,'string') ;
P.Rstep =  get(handles.edRstep,'string');
P.Rmax = get(handles.edRmax,'string') ;
P.Rmin =  get(handles.edRmin,'string') ;
P.Frame_Step =  get(handles.edFrame_Step,'string') ;
P.Frame_End =  get(handles.edFrame_End,'string') ;
P.Frame_begin =  get(handles.edFrame_begin,'string') ;
P.Vel = get( handles.pmRefInd,'Value' );
assignin('base', 'parameters',P);


% --- Executes on button press in pbView.
function pbView_Callback(hObject, eventdata, handles)
figure;
[AX,H1,H2] = plotyy(handles.Tp,handles.Ipp(1,:),handles.Ts,handles.Iss(1,:));
grid on;
title('First frame.');
set(H1,'color','r');
set(H2,'color','g');
% set(AX(1),'color','r');
% set(AX(2),'color','g');
figure;
[AX,H1,H2] = plotyy(handles.Tp,handles.Ipp(end,:),handles.Ts,handles.Iss(end,:));
grid on;
title('Last frame.');
set(H1,'color','r');
set(H2,'color','g');

% --- Executes on button press in cbRed.
function cbRed_Callback(hObject, eventdata, handles)
% hObject    handle to cbRed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbRed


% --- Executes on button press in cbBlue.
function cbBlue_Callback(hObject, eventdata, handles)
% hObject    handle to cbBlue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbBlue


% --- Executes on button press in cbGreen.
function cbGreen_Callback(hObject, eventdata, handles)
% hObject    handle to cbGreen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbGreen


% --- Executes during object creation, after setting all properties.
function uipanel1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uipanel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)

fit_model(hObject, handles);

% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edFps_Callback(hObject, eventdata, handles)
% hObject    handle to edFps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edFps as text
%        str2double(get(hObject,'String')) returns contents of edFps as a double


% --- Executes during object creation, after setting all properties.
function edFps_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edFps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ch2refInd.
function ch2refInd_Callback(hObject, eventdata, handles)

if(get(handles.ch2refInd,'Value')==1)
    set(handles.pmRefInd2,'Enable','on');
    set(handles.te_m_red2,'Enable','on');
    set(handles.te_m_green2,'Enable','on');
else
    set(handles.pmRefInd2,'Enable','off');
    set(handles.te_m_red2,'Enable','off');
    set(handles.te_m_green2,'Enable','off');
end
    

% hObject    handle to ch2refInd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ch2refInd


% --- Executes on selection change in pmRefInd2.
function pmRefInd2_Callback(hObject, eventdata, handles)

% Calculate refractive index
S = get( handles.pmRefInd2,'String' );
Vel = get( handles.pmRefInd2,'Value' );
handles.mr2 = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg2 = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );
set( handles.te_m_red2,'string',['m_r = ' num2str(handles.mr2)]);
set( handles.te_m_green2,'string',['m_g = ' num2str(handles.mg2)]);
% Update handles structure
guidata(hObject, handles);

% hObject    handle to pmRefInd2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pmRefInd2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pmRefInd2


% --- Executes during object creation, after setting all properties.
function pmRefInd2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pmRefInd2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)

[ref_ind_r, fraction] = maxGarnett( hObject, handles, 'red' );
[ref_ind_g, fraction] = maxGarnett( hObject, handles, 'green' );

handles.fraction = fraction;
guidata(hObject,handles);

figure()
plot(handles.time,handles.fraction);

% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edSmoothFactor_Callback(hObject, eventdata, handles)
% hObject    handle to edSmoothFactor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edSmoothFactor as text
%        str2double(get(hObject,'String')) returns contents of edSmoothFactor as a double


% --- Executes during object creation, after setting all properties.
function edSmoothFactor_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edSmoothFactor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)

[ref_ind_r, fraction] = maxGarnett( hObject, handles, 'red' );
[ref_ind_g, fraction] = maxGarnett( hObject, handles, 'green' );

time = handles.time;
results = evalin('base', 'results');
rExp = handles.rMaxGarnett;

p = polyfit(time,rExp,25);
ft = polyval(p,time);

ref_ind_r_real = interp1(handles.r',ref_ind_r,ft);
ref_ind_g_real = interp1(handles.r',ref_ind_g,ft);

handles.ref_ind_r_real = ref_ind_r_real;
handles.ref_ind_g_real = ref_ind_g_real;
guidata(hObject,handles);

figure()
plot(handles.time,handles.ref_ind_r_real,'r');
hold on
plot(handles.time,handles.ref_ind_g_real,'g');
hold off

% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4


% --- Executes on button press in radiobutton5.
function radiobutton5_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton5


% --- Executes on button press in radiobutton6.
function radiobutton6_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton6


% --- Executes on button press in chRed.
function chRed_Callback(hObject, eventdata, handles)

val = get(handles.chRed,'Value');
if(val == 1)
    set(handles.chGreen,'Value', 0);
    set(handles.chBlue,'Value', 0);
else
    set(handles.chGreen,'Value', 1);
    set(handles.chBlue,'Value', 0);
end

% hObject    handle to chRed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chRed


% --- Executes on button press in chGreen.
function chGreen_Callback(hObject, eventdata, handles)

val = get(handles.chGreen,'Value');
if(val == 1)
    set(handles.chRed,'Value', 0);
    set(handles.chBlue,'Value', 0);
else
    set(handles.chRed,'Value', 0);
    set(handles.chBlue,'Value', 1);
end
% hObject    handle to chGreen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chGreen


% --- Executes on button press in chBlue.
function chBlue_Callback(hObject, eventdata, handles)

val = get(handles.chBlue,'Value');
if(val == 1)
    set(handles.chGreen,'Value', 0);
    set(handles.chRed,'Value', 0);
else
    set(handles.chRed,'Value', 1);
    set(handles.chBlue,'Value', 0);
end

% hObject    handle to chBlue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chBlue
