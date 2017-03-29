function varargout = SetSystem(varargin)
%  This function creates initial structure for objective and CCD
%  parameters.
%  All lengths are in [ mm ]
%  The origin of the coordinate system is in the center of the trap.

% SETSYSTEM MATLAB code for SetSystem.fig
%      SETSYSTEM, by itself, creates a new SETSYSTEM or raises the existing
%      singleton*.
%
%      H = SETSYSTEM returns the handle to a new SETSYSTEM or the handle to
%      the existing singleton*.
%
%      SETSYSTEM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SETSYSTEM.M with the given input arguments.
%
%      SETSYSTEM('Property','Value',...) creates a new SETSYSTEM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SetSystem_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SetSystem_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SetSystem

% Last Modified by GUIDE v2.5 28-Mar-2017 15:19:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SetSystem_OpeningFcn, ...
                   'gui_OutputFcn',  @SetSystem_OutputFcn, ...
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


% --- Executes just before SetSystem is made visible.
function SetSystem_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SetSystem (see VARARGIN)

% Choose default command line output for SetSystem
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SetSystem wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SetSystem_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
if isempty(handles)
    return
end
switch get(handles.pmu_SetSystem,'Value');
    
    case 1
       % Parameters of lens:
handles.S.D    = 12;            % lens diametr. Actually defined by the inner diameter of the objective tube 
                        % nominal according to drawing is 12
                        % big chaber objectives: 11.88
                        % whole lens from Mr. Wódka workshop - www.poioptyka.com.pl:
                        % 12.4 - 0.2 is the edge cut --> 0.14142 in diam. and thickness
                        % Setting 15 is sometimes convenient for calculations
handles.S.R(1)    = 10.3; % radius of lens curvature first lens
handles.S.R(2)    = 10.3; % radius of lens curvature second lens
handles.S.tc   = 1.975+0.2;  % thickness of the lenses barrel - this parameter -
                 % is uniquely determined by the radius of curvature,
                 % diameter of the lens and thickness of the whole lens along
                 % optical axes & 1.8 by GD
                 % + shift caused by reduced inner diameter of the tube
handles.S.g    = 4.01;  % thickness of the whole lens along optical axes
% Distances:
handles.S.ld   = 15;   % Distance between center of the trap and  first diaphragm
handles.S.R_dis_Ring = 29.68/2; % outer radius of electrode distancing ring
handles.S.R_midl_El  = 29.64/2;  % outer radius of midle electrode of trap

handles.S.l1   = 17.33;   %  Distance between center of the trap and  first lens
%handles.S.ll   = 37.4;   % Distance between lenses apex-apex + optical length in polarizer - small chamber
handles.S.ll   = 37.4+1.3;   % Distance between lenses apex-apex + optical length in polarizer - big chamber
handles.S.ld2  = 58.3 + handles.S.l1; % 58.3 - distence from first lens to second diafragm  %76.6; % Distance to second diaphragm
% Wavelength of incident ray
handles.S.lambda = [];  %lambda can have the structure of RGB lambda(1:3) = [R,G,B] but so far it does not;
handles.S.N  =   20;    % number of points per border side
% Effective lens aperture (diameter) % 11.8 suggested by GD
% S.efD  = effective_aperture(S.D/2,S.tc,S.l1,S.lambda,25); % it is calculated in PikeReader for each color
% First Diaphragm
handles.S.dW   = 9;    % width of diaphragm
handles.S.dH   = 4.27;    % height of diaphragm
% Second diaphragm
% handles.S.RDph  = 1;   % Radius of aperture - small chamber
handles.S.RDph  = 10; %0.5;   % Radius of aperture - big chamber 10 - aperture removed
handles.S.W2    = 1;   % thickness of the diaphragm wall
% CCD parameters
handles.S.lCCD = 82.8; % Distance to CCD detector
handles.S.CCDPH = 480; % width of CCD [ Pix ]
handles.S.CCDPW = 640; % height of CCD [Pix ]
handles.S.PixSize = 7.4e-3; % Pixel size[ mm ] Pike
handles.S.CCDH = handles.S.CCDPH * handles.S.PixSize;  % height of CCD
handles.S.CCDW = handles.S.CCDPW * handles.S.PixSize;  % width  of CCD
% Base droplet position 
handles.S.Pk   = [0,0,0]; % Position of droplet relativ to the origin of coordinate system
        
    case 2
       % Parameters of lens:
handles.S.D    = 12;            % lens diametr. Actually defined by the inner diameter of the objective tube 
                        % nominal according to drawing is 12
                        % big chaber objectives: 11.88
                        % whole lens from Mr. Wódka workshop - www.poioptyka.com.pl:
                        % 12.4 - 0.2 is the edge cut --> 0.14142 in diam. and thickness
                        % Setting 15 is sometimes convenient for calculations
handles.S.R(1)    = 10.3; % radius of lens curvature first lens
handles.S.R(2)    = 10.3; % radius of lens curvature second lens
handles.S.tc   = 1.975+0.2;  % thickness of the lenses barrel - this parameter -
                 % is uniquely determined by the radius of curvature,
                 % diameter of the lens and thickness of the whole lens along
                 % optical axes & 1.8 by GD
                 % + shift caused by reduced inner diameter of the tube
handles.S.g    = 4.01;  % thickness of the whole lens along optical axes
% Distances:
handles.S.ld   = 15;   % Distance between center of the trap and  first diaphragm
handles.S.R_dis_Ring = 29.68/2; % outer radius of electrode distancing ring
handles.S.R_midl_El  = 29.64/2;  % outer radius of midle electrode of trap

handles.S.l1   = 17.33;   %  Distance between center of the trap and  first lens
handles.S.ll   = 37.4;   % Distance between lenses apex-apex + optical length in polarizer - small chamber
% handles.S.ll   = 37.4+1.3;   % Distance between lenses apex-apex + optical length in polarizer - big chamber
handles.S.ld2  = 58.3 + handles.S.l1; % 58.3 - distence from first lens to second diafragm  %76.6; % Distance to second diaphragm
% Wavelength of incident ray
handles.S.lambda = [];  %lambda can have the structure of RGB lambda(1:3) = [R,G,B] but so far it does not;
handles.S.N  =   20;    % number of points per border side
% Effective lens aperture (diameter) % 11.8 suggested by GD
% S.efD  = effective_aperture(S.D/2,S.tc,S.l1,S.lambda,25); % it is calculated in PikeReader for each color
% First Diaphragm
handles.S.dW   = 9;    % width of diaphragm
handles.S.dH   = 4.27;    % height of diaphragm
% Second diaphragm
% S.RDph  = 1;   % Radius of aperture - small chamber
handles.S.RDph  = 10; %0.5;   % Radius of aperture - big chamber 10 - aperture removed
handles.S.W2    = 1;   % thickness of the diaphragm wall
% CCD parameters
handles.S.lCCD = 82.8; % Distance to CCD detector
handles.S.CCDPH = 480; % width of CCD [ Pix ]
handles.S.CCDPW = 640; % height of CCD [Pix ]
handles.S.PixSize = 7.4e-3; % Pixel size[ mm ] Pike
handles.S.CCDH = handles.S.CCDPH * handles.S.PixSize;  % height of CCD
handles.S.CCDW = handles.S.CCDPW * handles.S.PixSize;  % width  of CCD
% Base droplet position 
handles.S.Pk   = [0,0,0]; % Position of droplet relativ to the origin of coordinate system
end
 guidata(hObject, handles);
varargout{1} = handles.output;
varargout{2} = handles.S;
uiresume(hObject)


% --- Executes on selection change in pmu_SetSystem.
function pmu_SetSystem_Callback(hObject, eventdata, handles)
% hObject    handle to pmu_SetSystem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pmu_SetSystem contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pmu_SetSystem

% switch get(handles.pmu_SetSystem,'Value');
%     
%     case 1
%        % Parameters of lens:
% handles.S.D    = 12;            % lens diametr. Actually defined by the inner diameter of the objective tube 
%                         % nominal according to drawing is 12
%                         % big chaber objectives: 11.88
%                         % whole lens from Mr. Wódka workshop - www.poioptyka.com.pl:
%                         % 12.4 - 0.2 is the edge cut --> 0.14142 in diam. and thickness
%                         % Setting 15 is sometimes convenient for calculations
% handles.S.R(1)    = 10.3; % radius of lens curvature first lens
% handles.S.R(2)    = 10.3; % radius of lens curvature second lens
% handles.S.tc   = 1.975+0.2;  % thickness of the lenses barrel - this parameter -
%                  % is uniquely determined by the radius of curvature,
%                  % diameter of the lens and thickness of the whole lens along
%                  % optical axes & 1.8 by GD
%                  % + shift caused by reduced inner diameter of the tube
% handles.S.g    = 4.01;  % thickness of the whole lens along optical axes
% % Distances:
% handles.S.ld   = 15;   % Distance between center of the trap and  first diaphragm
% handles.S.R_dis_Ring = 29.68/2; % outer radius of electrode distancing ring
% handles.S.R_midl_El  = 29.64/2;  % outer radius of midle electrode of trap
% 
% handles.S.l1   = 17.33;   %  Distance between center of the trap and  first lens
% %handles.S.ll   = 37.4;   % Distance between lenses apex-apex + optical length in polarizer - small chamber
% handles.S.ll   = 37.4+1.3;   % Distance between lenses apex-apex + optical length in polarizer - big chamber
% handles.S.ld2  = 58.3 + handles.S.l1; % 58.3 - distence from first lens to second diafragm  %76.6; % Distance to second diaphragm
% % Wavelength of incident ray
% handles.S.lambda = [];  %lambda can have the structure of RGB lambda(1:3) = [R,G,B] but so far it does not;
% handles.S.N  =   20;    % number of points per border side
% % Effective lens aperture (diameter) % 11.8 suggested by GD
% % S.efD  = effective_aperture(S.D/2,S.tc,S.l1,S.lambda,25); % it is calculated in PikeReader for each color
% % First Diaphragm
% handles.S.dW   = 9;    % width of diaphragm
% handles.S.dH   = 4.27;    % height of diaphragm
% % Second diaphragm
% % handles.S.RDph  = 1;   % Radius of aperture - small chamber
% handles.S.RDph  = 10; %0.5;   % Radius of aperture - big chamber 10 - aperture removed
% handles.S.W2    = 1;   % thickness of the diaphragm wall
% % CCD parameters
% handles.S.lCCD = 82.8; % Distance to CCD detector
% handles.S.CCDPH = 480; % width of CCD [ Pix ]
% handles.S.CCDPW = 640; % height of CCD [Pix ]
% handles.S.PixSize = 7.4e-3; % Pixel size[ mm ] Pike
% handles.S.CCDH = handles.S.CCDPH * handles.S.PixSize;  % height of CCD
% handles.S.CCDW = handles.S.CCDPW * handles.S.PixSize;  % width  of CCD
% % Base droplet position 
% handles.S.Pk   = [0,0,0]; % Position of droplet relativ to the origin of coordinate system
%         
%     case 2
%        % Parameters of lens:
% handles.S.D    = 12;            % lens diametr. Actually defined by the inner diameter of the objective tube 
%                         % nominal according to drawing is 12
%                         % big chaber objectives: 11.88
%                         % whole lens from Mr. Wódka workshop - www.poioptyka.com.pl:
%                         % 12.4 - 0.2 is the edge cut --> 0.14142 in diam. and thickness
%                         % Setting 15 is sometimes convenient for calculations
% handles.S.R(1)    = 10.3; % radius of lens curvature first lens
% handles.S.R(2)    = 10.3; % radius of lens curvature second lens
% handles.S.tc   = 1.975+0.2;  % thickness of the lenses barrel - this parameter -
%                  % is uniquely determined by the radius of curvature,
%                  % diameter of the lens and thickness of the whole lens along
%                  % optical axes & 1.8 by GD
%                  % + shift caused by reduced inner diameter of the tube
% handles.S.g    = 4.01;  % thickness of the whole lens along optical axes
% % Distances:
% handles.S.ld   = 15;   % Distance between center of the trap and  first diaphragm
% handles.S.R_dis_Ring = 29.68/2; % outer radius of electrode distancing ring
% handles.S.R_midl_El  = 29.64/2;  % outer radius of midle electrode of trap
% 
% handles.S.l1   = 17.33;   %  Distance between center of the trap and  first lens
% handles.S.ll   = 37.4;   % Distance between lenses apex-apex + optical length in polarizer - small chamber
% % handles.S.ll   = 37.4+1.3;   % Distance between lenses apex-apex + optical length in polarizer - big chamber
% handles.S.ld2  = 58.3 + handles.S.l1; % 58.3 - distence from first lens to second diafragm  %76.6; % Distance to second diaphragm
% % Wavelength of incident ray
% handles.S.lambda = [];  %lambda can have the structure of RGB lambda(1:3) = [R,G,B] but so far it does not;
% handles.S.N  =   20;    % number of points per border side
% % Effective lens aperture (diameter) % 11.8 suggested by GD
% % S.efD  = effective_aperture(S.D/2,S.tc,S.l1,S.lambda,25); % it is calculated in PikeReader for each color
% % First Diaphragm
% handles.S.dW   = 9;    % width of diaphragm
% handles.S.dH   = 4.27;    % height of diaphragm
% % Second diaphragm
% % S.RDph  = 1;   % Radius of aperture - small chamber
% handles.S.RDph  = 10; %0.5;   % Radius of aperture - big chamber 10 - aperture removed
% handles.S.W2    = 1;   % thickness of the diaphragm wall
% % CCD parameters
% handles.S.lCCD = 82.8; % Distance to CCD detector
% handles.S.CCDPH = 480; % width of CCD [ Pix ]
% handles.S.CCDPW = 640; % height of CCD [Pix ]
% handles.S.PixSize = 7.4e-3; % Pixel size[ mm ] Pike
% handles.S.CCDH = handles.S.CCDPH * handles.S.PixSize;  % height of CCD
% handles.S.CCDW = handles.S.CCDPW * handles.S.PixSize;  % width  of CCD
% % Base droplet position 
% handles.S.Pk   = [0,0,0]; % Position of droplet relativ to the origin of coordinate system
% end
%  guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function pmu_SetSystem_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pmu_SetSystem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
 delete(hObject);
