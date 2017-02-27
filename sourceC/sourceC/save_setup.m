idSetup = fopen('Cdata/setup','w');

reall='float';

fwrite(idSetup, real(handles.mr), reall);
fwrite(idSetup, imag(handles.mr), reall);


fwrite(idSetup, real(handles.mg), reall);
fwrite(idSetup, imag(handles.mg), reall);


fwrite(idSetup, str2num( get( handles.edRmin,'string' ) ), reall);
fwrite(idSetup, str2num( get( handles.edRmax,'string' ) ), reall);
fwrite(idSetup, str2num( get( handles.edRstep,'string' ) ), reall);

fwrite(idSetup, str2num( get(handles.edScale,'string') ), reall);
fwrite(idSetup, str2num( get(handles.edShift_R,'string') ), reall);
fwrite(idSetup, str2num( get(handles.edShift_G,'string') ), reall);

fwrite(idSetup, real(str2num( get(handles.edShift_m,'string') )), reall);
fwrite(idSetup, imag(str2num( get(handles.edShift_m,'string') )), reall);

fwrite(idSetup, real(str2num( get( handles.edShift_mred,'string') )), reall);
fwrite(idSetup, imag(str2num( get( handles.edShift_mred,'string') )), reall);

fwrite(idSetup, str2num( get(handles.edFrame_begin,'string') ), 'int32');
fwrite(idSetup, str2num( get(handles.edFrame_Step,'string') ), 'int32');
fwrite(idSetup, str2num( get(handles.edFrame_End,'string') ), 'int32');

fwrite(idSetup, handles.Wr.wavelength, reall);
fwrite(idSetup, handles.Wg.wavelength, reall);

fclose(idSetup);
