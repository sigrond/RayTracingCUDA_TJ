function varargout = ViewFinder(varargin)
% VIEWFINDER MATLAB code for ViewFinder.fig
%      VIEWFINDER, by itself, creates a new VIEWFINDER or raises the existing
%      singleton*.
%
%      H = VIEWFINDER returns the handle to a new VIEWFINDER or the handle to
%      the existing singleton*.
%
%      VIEWFINDER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VIEWFINDER.M with the given input arguments.
%
%      VIEWFINDER('Property','Value',...) creates a new VIEWFINDER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ViewFinder_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ViewFinder_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ViewFinder

% Last Modified by GUIDE v2.5 11-Feb-2017 11:46:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ViewFinder_OpeningFcn, ...
                   'gui_OutputFcn',  @ViewFinder_OutputFcn, ...
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

function handles=ReDraw(hObject,handles)
r=658;
g=532;
b=458;
Args=[handles.Pk, handles.PCCD];
[X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),r);
delete(handles.hp);
handles.hp=plot(handles.ha,X,Y,'-xr');
[X Y]=BorderFunction(Args(1),Args(2),Args(3),Args(4),Args(5),Args(6),b);
delete(handles.hpb);
handles.hpb=plot(handles.ha,X,Y,'-xb');
set(handles.hf,'name',sprintf('%f %f %f %f %f %f',Args(1),Args(2),Args(3),Args(4),Args(5),Args(6)))
drawnow
oldFcelu=handles.Fcelu;
handles.Fcelu=MeanSquaredDistance(handles.pointsr,handles.pointsb,Args);
set(handles.text7,'String',sprintf('F-celu: %s',num2str(handles.Fcelu)));
Fdif=oldFcelu-handles.Fcelu;
    set(handles.text8,'String',sprintf('Poprawa: %s',num2str(Fdif)));
if handles.Fcelu>oldFcelu
    set(handles.text8,'ForeGroundColor','r');
else
    set(handles.text8,'ForeGroundColor','g');
end
myButtons=[handles.FPkx handles.FPky handles.FPkz handles.FPCx handles.FPCy handles.FPCz];
g=zeros(6,2);
g1=zeros(6,1);
dx=0.1;
for i=1:6
    tmpA=Args;
    tmpA(i)=Args(i)+dx;
    g(i,1)=MeanSquaredDistance(handles.pointsr,handles.pointsb,Args)-MeanSquaredDistance(handles.pointsr,handles.pointsb,tmpA);
    tmpA(i)=Args(i)-dx;
    g(i,2)=MeanSquaredDistance(handles.pointsr,handles.pointsb,Args)-MeanSquaredDistance(handles.pointsr,handles.pointsb,tmpA);
    if(g(i,1)>g(i,2) && g(i,1)>0)
        g1(i)=g(i,1);
        set(myButtons(i),'String','+0.1');
    elseif g(i,2)>g(i,1) && g(i,2)>0
        g1(i)=g(i,2);
        set(myButtons(i),'String','-0.1');
    else
        g1(i)=0;
        set(myButtons(i),'String','+0.0');
    end
end
quiver3(handles.axes2,0,0,0,g1(1),g1(2),g1(3));
quiver3(handles.axes1,0,0,0,g1(4),g1(5),g1(6));
[M,I]=max(g(:));
[I_row, I_col] = ind2sub(size(g),I);
set(handles.FPkx,'ForeGroundColor','k');
set(handles.FPky,'ForeGroundColor','k');
set(handles.FPkz,'ForeGroundColor','k');
set(handles.FPCx,'ForeGroundColor','k');
set(handles.FPCy,'ForeGroundColor','k');
set(handles.FPCz,'ForeGroundColor','k');

for i=1:6
    if(g(i,1)>g(i,2) && g(i,1)>0)
        g1(i)=g(i,1);
    elseif g(i,2)>g(i,1) && g(i,2)>0
        g1(i)=g(i,2);
    else
        g1(i)=0;
    end
end
switch I_row
    case 1
        if I_col==1
            set(handles.FPkx,'String','+0.1');
            %quiver3(handles.axes2,0,0,0,dx,0,0);
        else
            set(handles.FPkx,'String','-0.1');
            %quiver3(handles.axes2,0,0,0,-dx,0,0);
        end
        quiver3(handles.axes1,0,0,0,0,0,0);
        %set(handles.FPkx,'ForeGroundColor','g');
    case 2
        if I_col==1
            set(handles.FPky,'String','+0.1');
            %quiver3(handles.axes2,0,0,0,0,dx,0);
        else
            set(handles.FPky,'String','-0.1');
            %quiver3(handles.axes2,0,0,0,0,-dx,0);
        end
        %quiver3(handles.axes1,0,0,0,0,0,0);
        set(handles.FPky,'ForeGroundColor','g');
    case 3
        if I_col==1
            set(handles.FPkz,'String','+0.1');
            %quiver3(handles.axes2,0,0,0,0,0,dx);
        else
            set(handles.FPkz,'String','-0.1');
            %quiver3(handles.axes2,0,0,0,0,0,-dx);
        end
        %quiver3(handles.axes1,0,0,0,0,0,0);
        set(handles.FPkz,'ForeGroundColor','g');
    case 4
        if I_col==1
            set(handles.FPCx,'String','+0.1');
            %quiver3(handles.axes1,0,0,0,dx,0,0);
        else
            set(handles.FPCx,'String','-0.1');
            %quiver3(handles.axes1,0,0,0,-dx,0,0);
        end
        %quiver3(handles.axes2,0,0,0,0,0,0);
        set(handles.FPCx,'ForeGroundColor','g');
    case 5
        if I_col==1
            set(handles.FPCy,'String','+0.1');
            %quiver3(handles.axes1,0,0,0,0,dx,0);
        else
            set(handles.FPCy,'String','-0.1');
            %quiver3(handles.axes1,0,0,0,0,-dx,0);
        end
        %quiver3(handles.axes2,0,0,0,0,0,0);
        set(handles.FPCy,'ForeGroundColor','g');
    case 6
        if I_col==1
            set(handles.FPCz,'String','+0.1');
            %quiver3(handles.axes1,0,0,0,0,0,dx);
        else
            set(handles.FPCz,'String','-0.1');
            %quiver3(handles.axes1,0,0,0,0,0,-dx);
        end
        %quiver3(handles.axes2,0,0,0,0,0,0);
        set(handles.FPCz,'ForeGroundColor','g');
end
set(handles.edit1,'String',num2str(handles.Pk,4));
set(handles.edit2,'String',num2str(handles.PCCD,4));
%guidata(hObject, handles);


% --- Executes just before ViewFinder is made visible.
function ViewFinder_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ViewFinder (see VARARGIN)


%varargin
handles.Pk=varargin{1};
handles.PCCD=varargin{2};
handles.ha=varargin{3};
handles.hf=varargin{4};
handles.hp=varargin{5};
handles.hpb=varargin{6};
handles.pointsr=varargin{7};
handles.pointsb=varargin{8};

% Choose default command line output for ViewFinder
handles.output = [handles.Pk, handles.PCCD];

set(handles.edit1,'String',num2str(handles.Pk,4));
set(handles.edit2,'String',num2str(handles.PCCD,4));

Args=[handles.Pk, handles.PCCD];
handles.Fcelu=MeanSquaredDistance(handles.pointsr,handles.pointsb,Args);
set(handles.text7,'String',sprintf('F-celu: %s',num2str(handles.Fcelu)));

handles=ReDraw(hObject,handles);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ViewFinder wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ViewFinder_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.output = [handles.Pk, handles.PCCD];
% Get default command line output from handles structure
varargout{1} = handles.output;
% The figure can be deleted now
delete(handles.figure1);



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
handles.Pk=str2num(get(hObject,'String'));
handles=ReDraw(hObject,handles);
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



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double
handles.PCCD=str2num(get(hObject,'String'));
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_Apply.
function pushbutton_Apply_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Apply (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles=ReDraw(hObject,handles);
handles.output = [handles.Pk, handles.PCCD];
guidata(hObject, handles);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in FPkx.
function FPkx_Callback(hObject, eventdata, handles)
% hObject    handle to FPkx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Pk(1)=handles.Pk(1)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes on button press in FPky.
function FPky_Callback(hObject, eventdata, handles)
% hObject    handle to FPky (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Pk(2)=handles.Pk(2)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes on button press in FPkz.
function FPkz_Callback(hObject, eventdata, handles)
% hObject    handle to FPkz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.Pk(3)=handles.Pk(3)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes on button press in FPCx.
function FPCx_Callback(hObject, eventdata, handles)
% hObject    handle to FPCx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.PCCD(1)=handles.PCCD(1)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes on button press in FPCy.
function FPCy_Callback(hObject, eventdata, handles)
% hObject    handle to FPCy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.PCCD(2)=handles.PCCD(2)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes on button press in FPCz.
function FPCz_Callback(hObject, eventdata, handles)
% hObject    handle to FPCz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.PCCD(3)=handles.PCCD(3)+str2num(get(hObject,'String'));
set(hObject,'ForeGroundColor','b');
handles=ReDraw(hObject,handles);
guidata(hObject, handles);


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end
