function varargout = changeXYR(varargin)

%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help changeXYR

% Last Modified by GUIDE v2.5 08-Jan-2008 08:45:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @changeXYR_OpeningFcn, ...
                   'gui_OutputFcn',  @changeXYR_OutputFcn, ...
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

function plotkrug( handles )

r = floor(handles.nr);
x = -r:r;
         krug = [ ( handles.nyo-sqrt ( r^2-x.^2 ) ),( handles.nyo+sqrt ( r^2-x.^2 ) ) ];
         x = [ x -x ];
         set ( handles.ed.hLKorekt,'XData',x + handles.nxo,'YData',krug,'color','m' );
         set ( handles.ed.hmKorekt,'XData',handles.nxo,'YData',handles.nyo,...
              'MarkerFaceColor','r', 'MarkerEdgeColor',...
              'r','Marker','+', 'MarkerSize',15 );
  
% --- Executes just before changeXYR is made visible.
function changeXYR_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% 
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to changeXYR (see VARARGIN)

% Choose default command line output for changeXYR
handles.output = hObject;

if length(varargin) == 0

else
    enterDat = varargin{1};
    handles.ed=enterDat;
    
    handles.nxo = enterDat.params(1);
    handles.nyo = enterDat.params(2);
    handles.nr  = enterDat.params(3); 
    
    guidata(hObject, handles);
    
    set(handles.edxo,'String',num2str( floor ( enterDat.params(1) ) ) );
    
    set(handles.edyo,'String',num2str( floor ( enterDat.params(2) ) ) );
    
    set(handles.edr,'String',num2str( floor ( enterDat.params(3) ) ) );
%     
    set(handles.slxo,'Min',1,...
        'Max',480,'SliderStep',[1/480,1/48]',...
        'Value',floor ( enterDat.params(1) ) );
        
    set(handles.slyo,'Min',1,...
        'Max',640,'SliderStep',[1/640,1/64]',...
        'Value',floor ( enterDat.params(2) ) ); 
    
    set(handles.slr,'Min',1,...
         'Max',handles.nr*2,'SliderStep',[1/handles.nr,1/handles.nr]',...
         'Value',floor ( enterDat.params(3) ) ); 
end
  
% UIWAIT makes changeXYR wait for user response (see UIRESUME)
uiwait;%( handles.figure1 );

% guidata( hObject, handles );
% --- Outputs from this function are returned to the command line.
function varargout = changeXYR_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
varargout{2} = handles.nxo;
varargout{3} = handles.nyo;
varargout{4} = handles.nr;
close;


% --- Executes on slider movement.
function slxo_Callback(hObject, eventdata, handles)
% hObject    handle to slxo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.nxo = get ( handles.slxo,'Value' );
plotkrug(handles);
set ( handles.edxo,'String',num2str ( floor ( get( handles.slxo,'Value') ) ) );
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function slxo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slxo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edxo_Callback(hObject, eventdata, handles)

handles.nxo = str2num ( get ( handles.edxo,'String' ) );
set ( handles.slxo,'Value',handles.nxo );
plotkrug(handles);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edxo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edxo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal( get ( hObject,'BackgroundColor' ), get ( 0,'defaultUicontrolBackgroundColor' ) )
    set(hObject,'BackgroundColor','white');
end



% --- Executes on slider movement.
function slyo_Callback(hObject, eventdata, handles)

handles.nyo = get ( handles.slyo,'Value' );
plotkrug(handles);
set ( handles.edyo,'String',num2str ( floor ( get( handles.slyo,'Value') ) ) ); 
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function slyo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slyo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edyo_Callback(hObject, eventdata, handles)

handles.nyo = str2num ( get ( handles.edyo,'String' ) );
set ( handles.slyo,'Value',handles.nyo );
plotkrug(handles);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edyo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edyo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on slider movement.
function slr_Callback(hObject, eventdata, handles)

handles.nr = get ( handles.slr,'Value' );
plotkrug(handles);
set ( handles.edr,'String',num2str (floor ( get( handles.slr,'Value') ) ) ); 
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function slr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edr_Callback(hObject, eventdata, handles)

handles.nr = str2num ( get ( handles.edr,'String' ) );
set ( handles.slr,'Value',handles.nr );
plotkrug(handles);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbOk.
function pbOk_Callback(hObject, eventdata, handles)
% hObject    handle to pbOk (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% UIRESUME;
% 
% reDraw circle
% 

 r = floor(handles.nr);   
     x = -r:r;
     krug = [ ( handles.nyo-sqrt ( r^2-x.^2 ) ),( handles.nyo+sqrt ( r^2-x.^2 ) ) ];
     x = [ x -x ];
     
set ( handles.ed.hLine,'XData',x+handles.nxo,'YData',krug,'color','b' );
set ( handles.ed.hCentr,'XData',handles.nxo,'YData',handles.nyo,...
    'MarkerFaceColor','b', 'MarkerEdgeColor',...
    'g','Marker','+', 'MarkerSize',15 );

set ( handles.ed.hLKorekt,'XData',[0 0],'YData',[0 0],'color','m' );
set ( handles.ed.hmKorekt,'XData',[0 0],'YData',[0 0],...
    'MarkerFaceColor','r', 'MarkerEdgeColor',...
    'r','Marker','+', 'MarkerSize',1 );
close( handles.hImtool);
  UIRESUME;

% --- Executes on button press in pbClose.
function pbClose_Callback(hObject, eventdata, handles)
% hObject    handle to pbClose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
     handles.nxo = handles.ed.params(1);
     handles.nyo = handles.ed.params(2);
     handles.nr  = handles.ed.params(3);
   guidata(hObject, handles);
   
    r=floor(handles.nr);   
     x = -r:r;
     krug = [ ( handles.nyo-sqrt ( r^2-x.^2 ) ),( handles.nyo+sqrt ( r^2-x.^2 ) ) ];
     x = [ x -x ];
     
     set ( handles.ed.hLine,'XData',x+handles.nxo,'YData',krug,'color','b' );
     set ( handles.ed.hCentr,'XData',handles.nxo,'YData',handles.nyo,...
         'MarkerFaceColor','b', 'MarkerEdgeColor',...
         'g','Marker','+', 'MarkerSize',15 );
   
     set ( handles.ed.hLKorekt,'XData',[0 0],'YData',[0 0],'color','m' );
     set ( handles.ed.hmKorekt,'XData',[0 0],'YData',[0 0],...
         'MarkerFaceColor','r', 'MarkerEdgeColor',...
         'r','Marker','+', 'MarkerSize',1 );
     close( handles.hImtool);
UIRESUME;




% --- Executes on button press in pbOpen.
function pbOpen_Callback(hObject, eventdata, handles)
[x,y] = krug(handles.ed.params(1),handles.ed.params(2),handles.ed.params(3));    

handles.hImtool = imtool(handles.ed.Fr,[0,max(max(handles.ed.Fr))]);
hold on;
handles.ed.hLKorekt = plot(x,y);
handles.ed.hmKorekt = plot(handles.ed.params(1),handles.ed.params(2),'color','green','marker','+','markersize',25);
handles.ed.hLine = plot(x,y);
handles.ed.hCentr = plot(handles.ed.params(1),handles.ed.params(2),'color','green','marker','+','markersize',25);
hold off;
guidata(hObject, handles);
plotkrug( handles ); 