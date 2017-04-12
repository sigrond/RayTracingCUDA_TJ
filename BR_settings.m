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

% Last Modified by GUIDE v2.5 12-Apr-2017 12:40:46

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

function Dict=make_dictionary(name, Dict, n, v)
%Dict=struct();
Dict=setfield(Dict,n,v);
save(sprintf('Lang_%s.mat',name),'Dict');

function handles=LoadLang(hObject, handles)
handles.lang_list={'PL','ENG'};
filename=sprintf('Lang_%s.mat',handles.Lang);
try
    load(filename);
    handles.Dict=Dict;
catch
    disp(sprintf('Dictionary not found!\n Creating empty!\n'));
    Dict=struct();
    save(sprintf('%s',filename),'Dict');
    return;
end
for i=1:numel(handles.named_object_list)
    if isfield(handles.Dict,handles.named_object_list(i).code_name) && ~strcmp('',handles.Dict.(handles.named_object_list(i).code_name))
        handles.named_object_list(i).name=getfield(handles.Dict,handles.named_object_list(i).code_name);
        if isfield(handles.named_object_list(i).object,'Title')
            set(handles.named_object_list(i).object,'Title',handles.named_object_list(i).name);
        elseif isfield(handles.named_object_list(i).object,'Label')
            set(handles.named_object_list(i).object,'Label',handles.named_object_list(i).name);
        elseif isfield(handles.named_object_list(i).object,'String')
            set(handles.named_object_list(i).object,'String',handles.named_object_list(i).name);
        else
            try
                setfield(handles.named_object_list(i).object,'Title',handles.named_object_list(i).name);
            catch
            
                try
                    setfield(handles.named_object_list(i).object,'Label',handles.named_object_list(i).name);
                catch
                    try
                        setfield(handles.named_object_list(i).object,'String',handles.named_object_list(i).name);
                    catch

                        disp(handles.named_object_list(i).code_name);
                        disp(sprintf('no Title or Label!\n'));
                    end
                end
            end
        end
    else
        handles.named_object_list(i).code_name
        v=''
        handles.Dict=make_dictionary(handles.Lang,handles.Dict,handles.named_object_list(i).code_name,v);
    end
end
guidata(hObject, handles);

function mystr = iffchb(myvalue)
    if myvalue
        mystr='on';
    else
        mystr='off';
    end

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
try
load('BR_settings.mat');
catch
    disp(sprintf('Open BR_settings.mat exception!\n'));
end
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
if ~exist('OptTime','var');
    OptTime=300;
end
if ~exist('BPoints','var');
    BPoints=12;
end
if ~exist('Lang','var');
    Lang='PL';
end
if ~exist('SPointsR','var');
    SPointsR=[3:2:10 15:8:40 41:2:50 51:8:80];
end
if ~exist('SPointsB','var');
    SPointsB=[3:2:10 15:8:40 41:2:50 51:8:80];
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
if ~exist('ManualPointCorrection','var');
    ManualPointCorrection=0;
end
handles.ManualPointCorrection=ManualPointCorrection;
handles.DisplayedWindows=DisplayedWindows;
handles.FitFresnel=FitFresnel;
handles.SPointsR=SPointsR;
handles.SPointsB=SPointsB;
handles.Lang=Lang;
handles=LoadLang(hObject, handles);
handles.BrightTime=BrightTime;
handles.OptTime=OptTime;
handles.BP=BP;
switch BP
    case 1
        set(handles.radiobutton1,'Value',1);
    case 2
        set(handles.radiobutton2,'Value',1);
    case 3
        set(handles.radiobutton_BR_sel_sim_ane,'Value',1);
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
set(handles.checkbox_FitFresnel,'Value',FitFresnel);

handles.BPoints=BPoints;
set(handles.edit1,'String',BPoints);
set(handles.edit_BrightTime,'String',BrightTime);
set(handles.edit_OptTime,'String',OptTime);
set(handles.edit4,'String',num2str(SPointsR));
set(handles.edit5,'String',num2str(SPointsB));

if ~isfield(handles,'named_object_list')
    handles.named_object_list=struct('object',{},'code_name',{},'name',{});
end

if strcmp(handles.Lang,'PL')
    set(handles.PL,'checked','on');
else
    set(handles.PL,'checked','off');
end
if strcmp(handles.Lang,'ENG')
    set(handles.ENG,'checked','on');
else
    set(handles.ENG,'checked','off');
end


set(handles.BrightnesWindow,'checked',iffchb(DisplayedWindows.BrightnesWindow));
set(handles.SPointsWindow,'checked',iffchb(DisplayedWindows.SPointsWindow));
set(handles.OptimInfo,'checked',iffchb(DisplayedWindows.OptimInfo));
set(handles.SimAnealingWindow,'checked',iffchb(DisplayedWindows.SimAnealingWindow));
set(handles.FresnelFitPlots,'checked',iffchb(DisplayedWindows.FresnelFitPlots));
set(handles.FinalOptWindow,'checked',iffchb(DisplayedWindows.FinalOptWindow));

set(handles.checkboxManualPointCorrection,'Value',ManualPointCorrection);

%handles.named_object_list

%[f,p]=matlab.codetools.requiredFilesAndProducts('BorderRecognition.m');
p=ver;
for i=1:size(p,2)
    tmpOpt(i)=strcmp(p(i).Name,'Optimization Toolbox');
    tmpSym(i)=strcmp(p(i).Name,'Symbolic Math Toolbox');
    tmpIma(i)=strcmp(p(i).Name,'Image Processing Toolbox');
    tmpGlo(i)=strcmp(p(i).Name,'Global Optimization Toolbox');
    tmpCur(i)=strcmp(p(i).Name,'Curve Fitting Toolbox');
    tmpSim(i)=strcmp(p(i).Name,'Simulink Control Design');
    tmpSta(i)=strcmp(p(i).Name,'Statistics and Machine Learning Toolbox');
    tmpCom(i)=strcmp(p(i).Name,'Computer Vision System Toolbox');
end
if ~any(tmpOpt)
    set(handles.text_Optimization_Toolbox,'ForegroundColor','red');
    set(handles.radiobutton3,'Enable','off');
    handles.FitFresnel=0;
    set(handles.checkbox_FitFresnel,'value',0);
    set(handles.checkbox_FitFresnel,'Enable','off');
    set(handles.OptimInfo,'Checked','off');
    handles.DisplayedWindows.OptimInfo=0;
    set(handles.OptimInfo,'Enable','off');
end
if ~any(tmpSym)
    set(handles.text_Symbolic_Math_Toolbox,'ForegroundColor','red');
    %bêdzie brakowaæ ca³ki fresnela
    handles.FitFresnel=0;
    set(handles.checkbox_FitFresnel,'value',0);
    set(handles.checkbox_FitFresnel,'Enable','off');
end
if ~any(tmpIma)
    set(handles.text_Image_Processing_Toolbox,'ForegroundColor','red');
    set(handles.radiobutton2,'Enable','off');
    handles.ManualPointCorrection=0;
    set(handles.checkboxManualPointCorrection,'checked','off');
    set(handles.checkboxManualPointCorrection,'Enable','off');
    set(handles.BrightnesWindow,'Checked','off');
    handles.DisplayedWindows.BrightnesWindow=0;
    set(handles.BrightnesWindow,'Enable','off');
    set(handles.SPointsWindow,'Checked','off');
    handles.DisplayedWindows.SPointsWindow=0;
    set(handles.SPointsWindow,'Enable','off');
    set(handles.FinalOptWindow,'Checked','off');
    handles.DisplayedWindows.FinalOptWindow=0;
    set(handles.FinalOptWindow,'Enable','off');
end
if ~any(tmpGlo)
    set(handles.text_Global_Optimization_Toolbox,'ForegroundColor','red');
    set(handles.radiobutton4,'Enable','off');
    set(handles.radiobutton_BR_sel_sim_ane,'Enable','off');
    set(handles.radiobutton_SA,'Enable','off');
end
if ~any(tmpCur)
    set(handles.text_Curve_Fitting_Toolbox,'ForegroundColor','red');
    %handles.FitFresnel=0;
    %set(handles.checkbox_FitFresnel,'value',0);
    %set(handles.checkbox_FitFresnel,'Enable','off');
end
if ~any(tmpSim)
    set(handles.text_Simulink_Control_Design,'ForegroundColor','red');
    %handles.FitFresnel=0;
    %set(handles.checkbox_FitFresnel,'value',0);
    %set(handles.checkbox_FitFresnel,'Enable','off');
end
if ~any(tmpSta)
    set(handles.text_Statistics_and_Machine_Learning_Toolbox,'ForegroundColor','red');
    %handles.FitFresnel=0;
    %set(handles.checkbox_FitFresnel,'value',0);
    %set(handles.checkbox_FitFresnel,'Enable','off');
end
if ~any(tmpCom)
    set(handles.text_Computer_Vision_System_Toolbox,'ForegroundColor','red');
    %handles.FitFresnel=0;
    %set(handles.checkbox_FitFresnel,'value',0);
    %set(handles.checkbox_FitFresnel,'Enable','off');
end

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
Lang=handles.Lang;
%SPoints=handles.SPoints;
SPointsR=handles.SPointsR;
SPointsB=handles.SPointsB;
FitFresnel=handles.FitFresnel;
DisplayedWindows=handles.DisplayedWindows;
ManualPointCorrection=handles.ManualPointCorrection;
save('BR_settings.mat','BP','Op','BPoints','VFch','BrightTime','OptTime','Lang','SPointsR','SPointsB','FitFresnel','DisplayedWindows','ManualPointCorrection');


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
    case 3
        set(handles.radiobutton_BR_sel_sim_ane,'Value',1);
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
handles.SPointsR=SPointsR;
set(handles.edit4,'String',num2str(SPointsR));
handles.SPointsB=SPointsB;
set(handles.edit5,'String',num2str(SPointsB));

handles.DisplayedWindows=DisplayedWindows;
set(handles.BrightnesWindow,'checked',iffchb(DisplayedWindows.BrightnesWindow));
set(handles.SPointsWindow,'checked',iffchb(DisplayedWindows.SPointsWindow));
set(handles.OptimInfo,'checked',iffchb(DisplayedWindows.OptimInfo));
set(handles.SimAnealingWindow,'checked',iffchb(DisplayedWindows.SimAnealingWindow));
set(handles.FresnelFitPlots,'checked',iffchb(DisplayedWindows.FresnelFitPlots));
set(handles.FinalOptWindow,'checked',iffchb(DisplayedWindows.FinalOptWindow));

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


% --- Executes during object creation, after setting all properties.
function uibuttongroup1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uibuttongroup1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','uibuttongroup1','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if ~isfield(handles,'named_object_list')
    handles.named_object_list=struct('object',{},'code_name',{},'name',{});
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function lang_Callback(hObject, eventdata, handles)
% hObject    handle to lang (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function PL_Callback(hObject, eventdata, handles)
% hObject    handle to PL (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Lang='PL';
handles=LoadLang(hObject, handles);
for i=1:numel(handles.lang_handles_list)
    set(handles.lang_handles_list(i).object,'checked','off');
end
set(handles.PL,'checked','on');
guidata(hObject, handles);



% --------------------------------------------------------------------
function ENG_Callback(hObject, eventdata, handles)
% hObject    handle to ENG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Lang='ENG';
handles=LoadLang(hObject, handles);
for i=1:numel(handles.lang_handles_list)
    set(handles.lang_handles_list(i).object,'checked','off');
end
set(handles.ENG,'checked','on');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function lang_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lang (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.lang_handles_list=struct('object',{},'name',{});
if ~isfield(handles,'named_object_list')
    handles.named_object_list=struct('object',{},'code_name',{},'name',{});
end
handles.named_object_list(end+1)=struct('object',hObject,'code_name','lang','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function PL_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PL (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.lang_handles_list(end+1)=struct('object',hObject,'name','PL');

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function ENG_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ENG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.lang_handles_list(end+1)=struct('object',hObject,'name','ENG');

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function uibuttongroup2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uibuttongroup2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','uibuttongroup2','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function text2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','text2','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton1','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton2','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function text3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','text3','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function text4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','text4','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton3','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton4','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton5','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton6','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function radiobutton_SD_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radiobutton_SD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','radiobutton_SD','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function viewfinderchb_CreateFcn(hObject, eventdata, handles)
% hObject    handle to viewfinderchb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','viewfinderchb','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function pbsave_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pbsave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','pbsave','name','');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function pbload_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pbload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.named_object_list(end+1)=struct('object',hObject,'code_name','pbload','name','');
guidata(hObject, handles);



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double
handles.SPointsR=str2num(get(hObject,'String'));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double
handles.SPointsB=str2num(get(hObject,'String'));
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton_BR_sel_sim_ane.
function radiobutton_BR_sel_sim_ane_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton_BR_sel_sim_ane (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton_BR_sel_sim_ane
handles.BP=3;
guidata(hObject, handles);


% --- Executes on button press in pushbuttonResetRed.
function pushbuttonResetRed_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonResetRed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
SPointsR=[3:2:10 15:8:40 41:2:50 51:8:80];
set(handles.edit4,'String',num2str(SPointsR));
handles.SPointsR=SPointsR;
guidata(hObject, handles);



% --- Executes on button press in pushbuttonResetBlue.
function pushbuttonResetBlue_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonResetBlue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
SPointsB=[3:2:10 15:8:40 41:2:50 51:8:80];
set(handles.edit5,'String',num2str(SPointsB));
handles.SPointsB=SPointsB;
guidata(hObject, handles);


% --- Executes on button press in checkbox_FitFresnel.
function checkbox_FitFresnel_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_FitFresnel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_FitFresnel
FitFresnel=get(hObject,'Value');
handles.FitFresnel=FitFresnel;
guidata(hObject, handles);


% --------------------------------------------------------------------
function Windows_Callback(hObject, eventdata, handles)
% hObject    handle to Windows (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function BrightnesWindow_Callback(hObject, eventdata, handles)
% hObject    handle to BrightnesWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.BrightnesWindow.Checked, 'on')
    set(handles.BrightnesWindow,'Checked','off');
    handles.DisplayedWindows.BrightnesWindow=0;
else
    set(handles.BrightnesWindow,'Checked','on');
    handles.DisplayedWindows.BrightnesWindow=1;
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function SPointsWindow_Callback(hObject, eventdata, handles)
% hObject    handle to SPointsWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.SPointsWindow.Checked, 'on')
    set(handles.SPointsWindow,'Checked','off');
    handles.DisplayedWindows.SPointsWindow=0;
else
    set(handles.SPointsWindow,'Checked','on');
    handles.DisplayedWindows.SPointsWindow=1;
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function OptimInfo_Callback(hObject, eventdata, handles)
% hObject    handle to OptimInfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.OptimInfo.Checked, 'on')
    set(handles.OptimInfo,'Checked','off');
    handles.DisplayedWindows.OptimInfo=0;
else
    set(handles.OptimInfo,'Checked','on');
    handles.DisplayedWindows.OptimInfo=1;
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function SimAnealingWindow_Callback(hObject, eventdata, handles)
% hObject    handle to SimAnealingWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.SimAnealingWindow.Checked, 'on')
    set(handles.SimAnealingWindow,'Checked','off');
    handles.DisplayedWindows.SimAnealingWindow=0;
else
    set(handles.SimAnealingWindow,'Checked','on');
    handles.DisplayedWindows.SimAnealingWindow=1;
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function FresnelFitPlots_Callback(hObject, eventdata, handles)
% hObject    handle to FresnelFitPlots (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.FresnelFitPlots.Checked, 'on')
    set(handles.FresnelFitPlots,'Checked','off');
    handles.DisplayedWindows.FresnelFitPlots=0;
else
    set(handles.FresnelFitPlots,'Checked','on');
    handles.DisplayedWindows.FresnelFitPlots=1;
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function FinalOptWindow_Callback(hObject, eventdata, handles)
% hObject    handle to FinalOptWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp(handles.FinalOptWindow.Checked, 'on')
    set(handles.FinalOptWindow,'Checked','off');
    handles.DisplayedWindows.FinalOptWindow=0;
else
    set(handles.FinalOptWindow,'Checked','on');
    handles.DisplayedWindows.FinalOptWindow=1;
end
guidata(hObject, handles);



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in checkboxManualPointCorrection.
function checkboxManualPointCorrection_Callback(hObject, eventdata, handles)
% hObject    handle to checkboxManualPointCorrection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkboxManualPointCorrection
handles.ManualPointCorrection=get(hObject,'Value');
guidata(hObject, handles);


% --- Executes on button press in pushbutton_Continue.
function pushbutton_Continue_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Continue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
