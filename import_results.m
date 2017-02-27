reall='float';
idResults = fopen('Cdata/results', 'r');
mIpp = fread(idResults, 1, 'int32');
res.rr = fread(idResults, mIpp, reall);
res.rg = fread(idResults, mIpp, reall);
res.r  = fread(idResults, mIpp, reall);
fclose(idResults);