disp('Zapisuję pliki. Proszę czekać');
reall='float';

idSetup=fopen('Cdata/setupMatlab','w');
sizeIpp=size(handles.Ipp( handles.ind,: ));
fwrite(idSetup, sizeIpp, 'int32');
sizeIss=size(handles.Iss( handles.ind,: ));
fwrite(idSetup, sizeIss, 'int32');
sizemTp=length(handles.theta.mTp);
sizemTs=length(handles.theta.mTs);
fwrite(idSetup, sizemTp, 'int32');
fwrite(idSetup, sizemTs, 'int32');
fwrite(idSetup, handles.setup.Diafragma, reall);
fwrite(idSetup, handles.setup.hccd_max_G, reall);
fwrite(idSetup, handles.setup.hccd_max_R, reall);
fclose(idSetup);

idIpp=fopen('Cdata/ipp','w');
fwrite(idIpp, handles.Ipp( handles.ind,: )', reall);
fclose(idIpp);

idIss=fopen('Cdata/iss','w');
fwrite(idIss, handles.Iss( handles.ind,: )', reall);
fclose(idIss);

idmTp=fopen('Cdata/mTp','w');
fwrite(idmTp, handles.theta.mTp, reall);
fclose(idmTp);

idmTs=fopen('Cdata/mTs','w');
fwrite(idmTs, handles.theta.mTs, reall);
fclose(idmTs);
