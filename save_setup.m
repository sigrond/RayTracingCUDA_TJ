idSetup = fopen('Cdata/setup','w');

reall='float';

% Tutaj nalezy dopisywac zapis jakiejs zmiennej do pliku idSetup ktora bedzie decydowac czy czytamy z pliku wspolczynnik zalamania
% Nazwe zmiennej mozna sobie wybrac dowolnie, przykladowa funkcja moze wygladac tak:
% fwrite(idSetup, ZMIENNA, 'int32');
% Jezeli ZMIENNA = 1 to serwer mrfin bedzie czytal wspolczynniki z pliku, jezeli ZMIENNA != 1 to nie czyta
% Wspolczynnik zalamania musi byc podany jako 2 tablice liczb zespolonych zmiennoprzecinkowych pojedynczej precyzji (float). 
% Kazdy element tablicy to jedna liczba zespolona, najpierw przechowywana jest czesc rzeczywista.
% Najpierw czytany jest wspolczynnik zalamania dla koloru czerwonego, pozniej dla zielonego
% Dlugosc kazdej z tablic z wspolczynnikiem zalamania musi byc rowna:
% (rSize+3)/4*4; 
% gdzie rSize to dlugosc tablicy promieni (to powyzej to zaokraglenie do najmniejszej wielokrotnosci czworki ktora jest wieksza lub rowna rSize
% program oczekuje ze wspolczynnik zalamania dla koloru czerwonego bedzie w pliku "Cdata/mR", a dla zielonego w pliku "Cdata/mG"

% externalM = 0;                        % Wartość ustawiona przed wywołaniem funkcji  % MW

fwrite(idSetup, externalM, 'int32');

if  externalM ~= 1 
	externalM
	fwrite(idSetup, real(handles.mr), reall);
	fwrite(idSetup, imag(handles.mr), reall);

	fwrite(idSetup, real(handles.mg), reall);
	fwrite(idSetup, imag(handles.mg), reall);
end

if externalM == 1
    
    create_ref_ind_files(hObject, handles);
    
end

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
