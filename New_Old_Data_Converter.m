function varargout = New_Old_Data_Converter(varargin)
% NEW_OLD_DATA_CONVERTER M-file for New_Old_Data_Converter.fig
%      NEW_OLD_DATA_CONVERTER, by itself, creates a new NEW_OLD_DATA_CONVERTER or raises the existing
%      singleton*.
%
%      H = NEW_OLD_DATA_CONVERTER returns the handle to a new NEW_OLD_DATA_CONVERTER or the handle to
%      the existing singleton*.
%
%      NEW_OLD_DATA_CONVERTER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NEW_OLD_DATA_CONVERTER.M with the given input arguments.
%
%      NEW_OLD_DATA_CONVERTER('Property','Value',...) creates a new NEW_OLD_DATA_CONVERTER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before New_Old_Data_Converter_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to New_Old_Data_Converter_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help New_Old_Data_Converter

% Last Modified by GUIDE v2.5 02-Sep-2016 14:51:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @New_Old_Data_Converter_OpeningFcn, ...
                   'gui_OutputFcn',  @New_Old_Data_Converter_OutputFcn, ...
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


% --- Executes just before New_Old_Data_Converter is made visible.
function New_Old_Data_Converter_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to New_Old_Data_Converter (see VARARGIN)

% Choose default command line output for New_Old_Data_Converter
handles.output = hObject;
%--- Reading the data from base workspace ---
W = evalin('base','who');
handles.Ipp = [];
if ismember('Save',W)
    Save = evalin('base','Save');
else
    Save=struct;
end
if ismember('I_R',W)
%     I_R = evalin('base','I_R'); % Intensity
    set( handles.ed_I_R, 'string', Save.edR); % wavelength
else
    set( handles.ed_I_R, 'string', ''); % wavelength
    set( handles.ed_I_R,'enable','off');
    set( handles.rbSetAsIppR ,'enable','off');
    set( handles.rbSetAsIssR ,'enable','off')
end
if ismember('I_G',W)
%     I_G = evalin('base','I_G');
    set( handles.ed_I_G, 'string', Save.edG);
else
    set( handles.ed_I_G, 'string', '');
    set( handles.ed_I_G, 'enable','off');
    set( handles.rbSetAsIppG ,'enable','off');
    set( handles.rbSetAsIssG ,'enable','off')
end
if ismember('I_B',W)
%     I_B = evalin('base','I_B');
    set( handles.ed_I_B, 'string', Save.edB);
else
    set( handles.ed_I_B, 'string', '');
    set( handles.ed_I_B, 'enable','off');
    set( handles.rbSetAsIppB ,'enable','off');
    set( handles.rbSetAsIssB ,'enable','off')
end
handles.theta = [];
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes New_Old_Data_Converter wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = New_Old_Data_Converter_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
if ~isempty(handles.Ipp)
    out.Ipp = handles.Ipp;
    out.Iss = handles.Iss;
    out.theta = handles.theta;
    out.Wr = handles.Wr;
    out.Wg = handles.Wg;
    varargout{1} = out;
else
    varargout{1} = [];
end

delete(handles.figure1);



function ed_I_R_Callback(hObject, eventdata, handles)
% hObject    handle to ed_I_R (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_I_R as text
%        str2double(get(hObject,'String')) returns contents of ed_I_R as a double


% --- Executes during object creation, after setting all properties.
function ed_I_R_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_I_R (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_I_G_Callback(hObject, eventdata, handles)
% hObject    handle to ed_I_G (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_I_G as text
%        str2double(get(hObject,'String')) returns contents of ed_I_G as a double


% --- Executes during object creation, after setting all properties.
function ed_I_G_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_I_G (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_I_B_Callback(hObject, eventdata, handles)
% hObject    handle to ed_I_B (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_I_B as text
%        str2double(get(hObject,'String')) returns contents of ed_I_B as a double


% --- Executes during object creation, after setting all properties.
function ed_I_B_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_I_B (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rbSetAsIppR.
function rbSetAsIppR_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIppR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

 if get(hObject,'value')
   % set  radiobuttons value  
     set(handles.edIpp,'string',get(handles.ed_I_R,'string'));
     set(handles.rbSetAsIppG,'value',0);
     set(handles.rbSetAsIppB,'value',0);
   % reading data from "base" workspase  
     handles.Ipp = evalin('base','I_R');
    
   % setting wave structure
     handles.Wr.wavelength = str2double( get(handles.ed_I_R,'string') );
     handles.Wr.theta = 0;
     handles.Wr.polarization = 0;
     
     if get(handles.rbInvAngl_R,'value')
%          handles.theta.mTp = handles.theta.mTp(end:-1:1);
           handles.theta.mTp = -evalin('base','ThetaR')*pi/180 + pi/2;
     else
          handles.theta.mTp = evalin('base','ThetaR')*pi/180 + pi/2;
     end
 else
     set(handles.edIpp,'string','');
     handles.theta.mTp = [];
     handles.Ipp = [];
 end
 guidata(hObject, handles);

 
     
% Hint: get(hObject,'Value') returns toggle state of rbSetAsIppR


% --- Executes on button press in rbSetAsIssR.
function rbSetAsIssR_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIssR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject,'value')
     set(handles.edIss,'string',get(handles.ed_I_R,'string'));
     set(handles.rbSetAsIssG,'value',0);
     set(handles.rbSetAsIssB,'value',0)
     
     handles.Iss = evalin('base','I_R');
     % setting wave structure
     handles.Wg.wavelength = str2double( get(handles.ed_I_R,'string') );
     handles.Wg.theta = 0;
     handles.Wg.polarization = 1;
     
     if get(handles.rbInvAngl_R,'value')
%           handles.theta.mTs = handles.theta.mTs(end:-1:1);
          handles.theta.mTs = -evalin('base','ThetaR')*pi/180 + pi/2;
     else
         handles.theta.mTs = evalin('base','ThetaR')*pi/180 + pi/2;
     end
 else
     set(handles.edIss,'string','');
      handles.theta.mTs = [];
      handles.Iss = [];
end
 guidata(hObject, handles);

% Hint: get(hObject,'Value') returns toggle state of rbSetAsIssR


% --- Executes on button press in rbSetAsIppG.
function rbSetAsIppG_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIppG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject,'value')
     set(handles.edIpp,'string',get(handles.ed_I_G,'string'));
     set(handles.rbSetAsIppR,'value',0);
     set(handles.rbSetAsIppB,'value',0);
     
     handles.Ipp = evalin('base','I_G');
     % setting wave structure
     handles.Wr.wavelength = str2double( get(handles.ed_I_G,'string') );
     handles.Wr.theta = 0;
     handles.Wr.polarization = 0;
     
     if get(handles.rbInvAngl_G,'value')
         handles.theta.mTp = -evalin('base','ThetaG')*pi/180 + pi/2;
     else
         handles.theta.mTp = evalin('base','ThetaG')*pi/180 + pi/2;
     end
 else
     set(handles.edIpp,'string','');
     handles.Ipp = [];
      handles.theta.mTp = [];
end
 guidata(hObject, handles);

% Hint: get(hObject,'Value') returns toggle state of rbSetAsIppG


% --- Executes on button press in rbSetAsIssG.
function rbSetAsIssG_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIssG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject,'value')
     set(handles.edIss,'string',get(handles.ed_I_G,'string'));
     set(handles.rbSetAsIssR,'value',0);
     set(handles.rbSetAsIssB,'value',0);
     handles.Iss = evalin('base','I_G');
     
     % setting wave structure
     handles.Wg.wavelength = str2double( get(handles.ed_I_G,'string') );
     handles.Wg.theta = 0;
     handles.Wg.polarization = 1;
     
     if get(handles.rbInvAngl_G,'value')
         handles.theta.mTs = -evalin('base','ThetaG')*pi/180 + pi/2;
     else
         handles.theta.mTs = evalin('base','ThetaG')*pi/180 + pi/2;
     end
 else
     set(handles.edIss,'string','');
     handles.theta.mTs =[];
     handles.Iss = [];
end
 guidata(hObject, handles);

% Hint: get(hObject,'Value') returns toggle state of rbSetAsIssG


% --- Executes on button press in rbSetAsIppB.
function rbSetAsIppB_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIppB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject,'value')
     set(handles.edIpp,'string',get(handles.ed_I_B,'string'));
     set(handles.rbSetAsIppG,'value',0);
     set(handles.rbSetAsIppR,'value',0);
      handles.Ipp = evalin('base','I_B');
     
     % setting wave structure
     handles.Wr.wavelength = str2double( get(handles.ed_I_B,'string') );
     handles.Wr.theta = 0;
     handles.Wr.polarization = 0;
     if get(handles.rbInvAngl_B,'value')
         handles.theta.mTp = -evalin('base','ThetaB')*pi/180 + pi/2;
     else
         handles.theta.mTp = evalin('base','ThetaB')*pi/180 + pi/2;
     end
     
 else
     set(handles.edIpp,'string','');
     handles.theta.mTp = [];
     handles.Ipp = [];
end
 guidata(hObject, handles);

% Hint: get(hObject,'Value') returns toggle state of rbSetAsIppB


% --- Executes on button press in rbSetAsIssB.
function rbSetAsIssB_Callback(hObject, eventdata, handles)
% hObject    handle to rbSetAsIssB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject,'value')
     set(handles.edIss,'string',get(handles.ed_I_B,'string'));
     set(handles.rbSetAsIssG,'value',0);
     set(handles.rbSetAsIssR,'value',0);
      handles.Iss = evalin('base','I_B');
      handles.theta.mTs = evalin('base','ThetaB')*pi/180 + pi/2;
      % setting wave structure
     handles.Wg.wavelength = str2double( get(handles.ed_I_B,'string') );
     handles.Wg.theta = 0;
     handles.Wg.polarization = 1;
     if get(handles.rbInvAngl_B,'value')
         handles.theta.mTs = -evalin('base','ThetaB')*pi/180 + pi/2;
     else
          handles.theta.mTs = evalin('base','ThetaB')*pi/180 + pi/2;
        
     end
 else
     set(handles.edIss,'string','');
     handles.theta.mTs = [];
      handles.Iss = [];
end
 guidata(hObject, handles);

% Hint: get(hObject,'Value') returns toggle state of rbSetAsIssB



function edIpp_Callback(hObject, eventdata, handles)
% hObject    handle to edIpp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edIpp as text
%        str2double(get(hObject,'String')) returns contents of edIpp as a double


% --- Executes during object creation, after setting all properties.
function edIpp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edIpp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edIss_Callback(hObject, eventdata, handles)
% hObject    handle to edIss (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edIss as text
%        str2double(get(hObject,'String')) returns contents of edIss as a double


% --- Executes during object creation, after setting all properties.
function edIss_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edIss (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbConvert.
function pbConvert_Callback(hObject, eventdata, handles)
% hObject    handle to pbConvert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --- Executes on button press in rbInvAngl_B.
function rbInvAngl_B_Callback(hObject, eventdata, handles)
% hObject    handle to rbInvAngl_B (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbInvAngl_B


% --- Executes on button press in rbInvAngl_G.
function rbInvAngl_G_Callback(hObject, eventdata, handles)
% hObject    handle to rbInvAngl_G (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbInvAngl_G


% --- Executes on button press in rbInvAngl_R.
function rbInvAngl_R_Callback(hObject, eventdata, handles)
% hObject    handle to rbInvAngl_R (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of rbInvAngl_R


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if isequal(get(hObject,'waitstatus'),'waiting')
    uiresume(hObject);
else
delete(hObject);
end


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_2_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
