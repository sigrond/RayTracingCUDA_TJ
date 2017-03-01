function varargout = BR_settings(varargin)
% BR_SETTINGS MATLAB code for BR_settings.fig
%      BR_SETTINGS, by itself, creates a new BR_SETTINGS or raises the existing
%      singleton*.
%
%      H = BR_SETTINGS returns the handle to a new BR_SETTINGS or the handle to
%      the existing singleton*.
%
%      BR_SETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BR_SETTINGS.M with the given input arguments.
%
%      BR_SETTINGS('Property','Value',...) creates a new BR_SETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BR_settings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BR_settings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BR_settings

% Last Modified by GUIDE v2.5 01-Mar-2017 13:58:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BR_settings_OpeningFcn, ...
                   'gui_OutputFcn',  @BR_settings_OutputFcn, ...
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


% --- Executes just before BR_settings is made visible.
function BR_settings_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BR_settings (see VARARGIN)

% Choose default command line output for BR_settings
handles.output = hObject;

handles.BP=1;
handles.Op=1;
handles.VFch=0;
load('BR_settings.mat');
if ~exist('BP','var');
    BP=1;
end
if ~exist('Op','var');
    Op=1;
end
if ~exist('VFch','var');
    VFch=0;
end
if ~exist('BrightTime','var');
    BrightTime=10;
end
handles.BrightTime=BrightTime;
if ~exist('OptTime','var');
    OptTime=300;
end
handles.OptTime=OptTime;
handles.BP=BP;
switch BP
    case 1
        set(handles.radiobutton1,'Value',1);
    case 2
        set(handles.radiobutton2,'Value',1);
end
handles.Op=Op;
switch Op
    case 1
        set(handles.radiobutton3,'Value',1);
    case 2
        set(handles.radiobutton4,'Value',1);
    case 3
        set(handles.radiobutton5,'Value',1);
    case 4
        set(handles.radiobutton6,'Value',1);
    case 5
        set(handles.radiobutton7,'Value',1);
    case 6
        set(handles.radiobutton_SD,'Value',1);
    case 7
        set(handles.radiobutton_SA,'Value',1);
end
handles.VFch=VFch;
set(handles.viewfinderchb,'Value',VFch);
if ~exist('BPoints','var');
    BPoints=12;
end
handles.BPoints=BPoints;
set(handles.edit1,'String',BPoints);
set(handles.edit_BrightTime,'String',BrightTime);
set(handles.edit_OptTime,'String',OptTime);
        

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BR_settings wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BR_settings_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pbsave.
function pbsave_Callback(hObject, eventdata, handles)
% hObject    handle to pbsave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
BP=handles.BP;
Op=handles.Op;
BPoints=handles.BPoints;
VFch=handles.VFch;
BrightTime=handles.BrightTime;
OptTime=handles.OptTime;
save('BR_settings.mat','BP','Op','BPoints','VFch','BrightTime','OptTime');


% --- Executes on button press in pbload.
function pbload_Callback(hObject, eventdata, handles)
% hObject    handle to pbload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('BR_settings.mat');
handles.BP=BP;
handles.Op=Op;
handles.VFch=VFch;
switch BP
    case 1
        set(handles.radiobutton1,'Value',1);
    case 2
        set(handles.radiobutton2,'Value',1);
end
switch Op
    case 1
        set(handles.radiobutton3,'Value',1);
    case 2
        set(handles.radiobutton4,'Value',1);
    case 3
        set(handles.radiobutton5,'Value',1);
    case 4
        set(handles.radiobutton6,'Value',1);
    case 5
        set(handles.radiobutton7,'Value',1);
    case 6
        set(handles.radiobutton_SD,'Value',1);
    case 7
        set(handles.radiobutton_SA,'Value',1);
end
handles.VFch=VFch;
set(handles.viewfinderchb,'Value',VFch);
handles.BPoints=BPoints;
set(handles.edit1,'String',BPoints);
handles.BrightTime=BrightTime;
set(handles.edit_BrightTime,'String',BrightTime);
handles.OptTime=OptTime;
set(handles.edit_OptTime,'String',OptTime);

% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1

handles.BP=1;
guidata(hObject, handles);


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2

handles.BP=2;
guidata(hObject, handles);


% --- Executes on button press in radiobutton3.
function radiobutton3_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton3
handles.Op=1;
guidata(hObject, handles);


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4
handles.Op=2;
guidata(hObject, handles);


% --- Executes on button press in radiobutton5.
function radiobutton5_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton5
handles.Op=3;
guidata(hObject, handles);


% --- Executes on button press in radiobutton6.
function radiobutton6_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton6
handles.Op=4;
guidata(hObject, handles);


% --- Executes on button press in radiobutton7.
function radiobutton7_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton7
handles.Op=5;
guidata(hObject, handles);



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
handles.BPoints=str2double(get(hObject,'String'));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in viewfinderchb.
function viewfinderchb_Callback(hObject, eventdata, handles)
% hObject    handle to viewfinderchb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of viewfinderchb
handles.VFch=get(hObject,'Value');
guidata(hObject, handles);


% --- Executes on button press in pbqstart.
function pbqstart_Callback(hObject, eventdata, handles)
% hObject    handle to pbqstart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
quickstart


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
uiresume();
delete(hObject);



function edit_OptTime_Callback(hObject, eventdata, handles)
% hObject    handle to edit_OptTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_OptTime as text
%        str2double(get(hObject,'String')) returns contents of edit_OptTime as a double
handles.OptTime=str2double(get(hObject,'String'));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit_OptTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_OptTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_BrightTime_Callback(hObject, eventdata, handles)
% hObject    handle to edit_BrightTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_BrightTime as text
%        str2double(get(hObject,'String')) returns contents of edit_BrightTime as a double
handles.BrightTime=str2double(get(hObject,'String'));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit_BrightTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_BrightTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton_SD.
function radiobutton_SD_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_SD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_SD
handles.Op=6;
guidata(hObject, handles);


% --- Executes on button press in radiobutton_SA.
function radiobutton_SA_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_SA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_SA
handles.Op=7;
guidata(hObject, handles);


