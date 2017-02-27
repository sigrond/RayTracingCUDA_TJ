function [ ref_indx, realFraction ] = maxGarnett( hObject, handles, color )

results = evalin('base', 'results');

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

rModel = handles.model;
rTab = handles.r';
time = handles.time;

S = get( handles.pmRefInd,'String' );
Vel = get( handles.pmRefInd,'Value' );
handles.mr = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );
set( handles.te_m_red,'string',['m_r = ' num2str(handles.mr)]);
set( handles.te_m_green,'string',['m_g = ' num2str(handles.mg)]);

S = get( handles.pmRefInd2,'String' );
Vel = get( handles.pmRefInd2,'Value' );
handles.mr2 = Calculate_m( 25, handles.Wr.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) ) + str2num( get( handles.edShift_mred, 'string' ) );
handles.mg2 = Calculate_m( 25, handles.Wg.wavelength, S{Vel} ) +...
    str2num( get( handles.edShift_m, 'string' ) );
set( handles.te_m_red2,'string',['m_r = ' num2str(handles.mr2)]);
set( handles.te_m_green2,'string',['m_g = ' num2str(handles.mg2)]);

switch(color)
    case 'red'
        %n = 1.4678;    
        n = handles.mr;
        
        %nm = 1.4445;
        nm = handles.mr2;
        
    case 'green'
        %n = 1.4724;        
        n = handles.mg;
        
        %nm = 1.4581;
        nm = handles.mg2;
end

V = 4/3 .* pi .* rExp .^ 3;
V0 = 4/3 .* pi .* rModel .^ 3;
fraction = V0 ./ V;

sw = 0;
for i = 1 : length(fraction)
    if(fraction(i)>0.99)
        sw = 1;
    end
    if(sw==1)
        fraction(i) = 1;
    end
end

smoothFactor = str2num( get( handles.edSmoothFactor,'string' ) );
fraction = smooth(fraction,round(length(fraction)/smoothFactor));
realFraction = fraction;

p = polyfit(time,rExp,25);
ft = polyval(p,time);

rExp = smooth(ft,100);

f = interp1(rExp,fraction,rTab,'nearest');

%f = smooth(f,100);
%f1 = fit(rExp,fraction,'smoothingspline');
%f = f1(rTab);

indx = find(~isnan(f));

f(1:indx(1)) = f(indx(1));
f(indx(end):length(f)) = f(indx(end));


    E = n^2;
    Em = nm^2;

    num = 3 * f * ( E - Em ) / ( E + 2 * Em );
    denom = 1 - f * ( E - Em) / ( E + 2 * Em );

    Eavg = Em * ( 1 + num ./ denom );

    ref_indx = sqrt(Eavg);

 
end
