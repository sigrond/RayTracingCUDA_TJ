function[mov]=aviread12_matlab(Sciezka,zakres)
% Sciezka - œcie¿ka do pliku avi
% zakres - wektor zakresu klatek do przetworzenia
% 30.06.2010. Nale¿y optymalizowaæ wykorzystanie pamiêci!!!

X=Aviread12(Sciezka,zakres);
height=480;
width=640;
paddedWidth = 640*3;  
mov=zeros(length(X),480,640,3,'uint16');
% Truecolor frames 48bit RGB
% mov=zeros(length(X),height,width,3);
 
for i=1:length(X)          
% for i=1:size(X,1)% X jest macierz o rozmiarze 1x921600  
f = X(i).cdata;
    % f = X(i,:);
    % if height<0         
        % Movie frames are stored top down (height is negative).             
    %    f = permute(reshape(f, paddedWidth, abs(height)),[2 1 3]);         
    % else
    %   f = reshape(f, paddedWidth,height);
    %   f = rot90(reshape(f, paddedWidth,height));   
    % end
    % if paddedWidth ~= width            
    %   f = f(:,1:width*3);         
    % end
    % Movie frames are stored top down (height is negative).    
    
    % Wiemy, ¿e height zawsze dodatnia i paddedWidth~=width
    
    f = rot90(reshape(f, paddedWidth,height)); 
    f = f(:,1:width*3);
    RGB(1:abs(height), 1:width,3) = f(:,1:3:end);          
    RGB(:, :, 2) = f(:,2:3:end);          
    RGB(:, :, 1) = f(:,3:3:end);  
    mov(i,:,:,:)=RGB;
    %mov(i).cdata = RGB;               
end