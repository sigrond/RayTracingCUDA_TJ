function fit_model(hObject, handles)

results = evalin('base', 'results');
fps = str2num( get( handles.edFps,'string' ) );
frame_step = str2num( get( handles.edFrame_Step,'string' ) );

chRed = get(handles.chRed,'Value');
chGreen = get(handles.chGreen,'Value');
chBlue = get(handles.chBlue,'Value');

if(chRed == 1)
    rExp = results.rr;
end

if(chGreen == 1)
    rExp = results.rg;
end

if(chBlue == 1)
    rExp = results.r;
end

handles.rMaxGarnett = rExp;
guidata(hObject, handles);

y_data = handles.rMaxGarnett;

dt = frame_step / fps;

time = handles.time;


%time = [1:1:length(y_data)]';     %= evalin('base', 'time');
%time = evalin('base', 'time');

figure(handles.current_figure);

[x,y]=getpts;

fo_ = fitoptions('method','NonlinearLeastSquares','Lower',...
    [-Inf 159.999999 -Inf],'Upper',[Inf 160.000001 Inf]);

indx = find(time > x(1) & time < x(2));
x_pts = time(indx);
y_pts = y_data(indx); 

ok_ = isfinite(x_pts) & isfinite(y_pts);
if ~all( ok_ )
    warning( 'GenerateMFile:IgnoringNansAndInfs',...
        'Ignoring NaNs and Infs in data.' );
end
st_ = [y(1) 160 -4000 ];
set(fo_,'Startpoint',st_);
ft_ = fittype('sqrt((a0+b)^2+2*c*t )-b',...
    'dependent',{'y'},'independent',{'t'},...
    'coefficients',{'a0', 'b', 'c'});

% Fit this model using new data
cf_ = fit(x_pts(ok_),y_pts(ok_),ft_,fo_);
% Alternatively uncomment the following lines to use coefficients from the
% original fit. You can use this choice to plot the original fit against new
% data.
%    cv_ = { 3778.3566767006487, 160.00989118266003, -5390.4495928506703};
%    cf_ = cfit(ft_,cv_{:});

% Plot this fit
hold on
h_ = plot(time,cf_(time)); %,'fit',0.95);
set(h_(1),'Color',[0 0 0],...
    'LineStyle','-', 'LineWidth',2,...
    'Marker','none', 'MarkerSize',6);

hold off

handles.model = cf_(time);

[ref_ind_r, fraction] = maxGarnett( hObject, handles, 'red' );
[ref_ind_g, fraction] = maxGarnett( hObject, handles, 'green' );

handles.fraction = fraction;
guidata(hObject,handles);

end