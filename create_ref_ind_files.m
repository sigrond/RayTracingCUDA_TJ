function create_ref_ind_files(hObject, handles)

    %rSize = str2num( get( handles.r,'string' ) );
    
[ref_ind_r, fraction] = maxGarnett( hObject, handles, 'red' );
[ref_ind_g, fraction] = maxGarnett( hObject, handles, 'green' );

len = 4-mod(length(ref_ind_r),4);           % tablica zer dla części urojonej
zerosTab = 1E-3.*ones(2*(length(ref_ind_r)+len),1);      %SM : wydluzylem dwukrotnie tablice zer

    % === Red ===================================

    fid = fopen('Cdata/mR','w');

    extTable = ref_ind_r(length(ref_ind_r),1).*ones(len,1);
    data1 = [ref_ind_r(:,1); extTable];     % zwiększenie długości do wielokrotności 4
    
    %extTable = ref_ind_r(length(ref_ind_r),2).*ones(len,1);
    %data2 = [ref_ind_r(:,2); extTable];
    data2 = zerosTab;                       % na razie tylko zera
    
    %data = [data1 data2];
    output = zerosTab;
    output(1:2:end) = data1;          % SM: przeplatam zera z wynikami z data1

    fwrite(fid, output, 'float');

    fclose(fid);

    % === Green =================================
    
    fid = fopen('Cdata/mG','w');

    extTable = ref_ind_r(length(ref_ind_g),1).*ones(len,1);
    data1 = [ref_ind_g(:,1); extTable];     % zwiększenie długości do wielokrotności 4
    
    %extTable = ref_ind_r(length(ref_ind_g),2).*ones(len,1);
    %data2 = [ref_ind_g(:,2); extTable];
    data2 = zerosTab;                       % na razie tylko zera
    
    %data = [data1 data2];
    output = zerosTab;
    output(1:2:end) = data1;

    fwrite(fid, output, 'float');

    fclose(fid);

end
