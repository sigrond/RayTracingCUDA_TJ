function RGB = demosaic_mw(Image,pattern)
    [wid, len] = size(Image);
   
    R = zeros(wid+2,len+2);
    G = zeros(wid+2,len+2);
    B = zeros(wid+2,len+2);
      
    I = zeros(wid+2,len+2);
    I(2:wid+1,2:len+1) = Image;
   
    I(1,1) = Image(2,2);
    I(wid+2,len+2) = Image(wid-1,len-1);
    I(wid+2,1) = Image(wid-1,2);
    I(1,len+2) = Image(2,len-1);
   
    I(2:wid+1,1) = Image(1:wid,2);
    I(2:wid+1,len+2) = Image(1:wid,len-1);
    I(1,2:len+1) = Image(2,1:len);
    I(wid+2,2:len+1) = Image(wid-1,1:len);
       
    switch pattern
     
        case 'rggb'
   
            for i = 2 : wid+1
                for j = 2 : len+1
           
                    if(mod(i,2))        % nieparzyste
               
                        if(mod(j,2))        % nieparzyste
                   
                            R(i,j) = ( I(i-1,j-1) + I(i+1,j+1) + I(i-1,j+1) + I(i+1,j-1) ) / 4;
                            G(i,j) = ( I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) ) / 4;
                            B(i,j) = I(i,j);
                    
                        else                % parzyste
                   
                            R(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                            G(i,j) = I(i,j);
                            B(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                       
                        end
               
                    else                % parzyste
               
                        if(mod(j,2))        % nieparzyste       
        
                            R(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                            G(i,j) = I(i,j);
                            B(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                   
                        else                % parzyste
                   
                            R(i,j) = I(i,j);
                            G(i,j) = ( I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) ) / 4;
                            B(i,j) = ( I(i-1,j-1) + I(i+1,j+1) + I(i-1,j+1) + I(i+1,j-1) ) / 4;
                   
                        end
               
                    end
                end
            end
   
        case 'grbg'
   
            for i = 2 : wid+1
                for j = 2 : len+1
           
                    if(mod(i,2))        % nieparzyste
               
                        if(mod(j,2))        % nieparzyste
                   
                            R(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                            G(i,j) = I(i,j);
                            B(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
               
                        else                % parzyste
                   
                            R(i,j) = ( I(i-1,j-1) + I(i+1,j+1) + I(i-1,j+1) + I(i+1,j-1) ) / 4;
                            G(i,j) = ( I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) ) / 4;
                            B(i,j) = I(i,j);
   
                        end
               
                    else                % parzyste
               
                        if(mod(j,2))        % nieparzyste       
                    
                            R(i,j) = I(i,j);
                            G(i,j) = ( I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) ) / 4;
                            B(i,j) = ( I(i-1,j-1) + I(i+1,j+1) + I(i-1,j+1) + I(i+1,j-1) ) / 4;
                   
                        else                % parzyste
                   
                            R(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                            G(i,j) = I(i,j);
                            B(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                   
                        end
               
                    end
       
                end
            end
        
        case 'gbrg'
   
            for i = 3 : 2 : wid+1
                for j = 2 : 2 : len
                   
                      % if(mod(j,2))        % nieparzyste
                   
                            R(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                            G(i,j) = I(i,j);
                            B(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                   
                     %   else                % parzyste
                   
                            R(i,j+1) = I(i,j+1);
                            G(i,j+1) = ( I(i-1,j+1) + I(i+1,j+1) + I(i,j-1+1) + I(i,j+1+1) ) / 4;
                            B(i,j+1) = ( I(i-1,j-1+1) + I(i+1,j+1+1) + I(i-1,j+1+1) + I(i+1,j-1+1) ) / 4;
                    
                    %    end
                   
                end
            end
       
            for i = 2 : 2 : wid+1
                for j = 2 : 2 : len
               
                        R(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                        G(i,j) = I(i,j);
                        B(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                       
                        R(i,j+1) = ( I(i-1,j-1+1) + I(i+1,j+1+1) + I(i-1,j+1+1) + I(i+1,j-1+1) ) / 4;
                        G(i,j+1) = ( I(i-1,j+1) + I(i+1,j+1) + I(i,j-1+1) + I(i,j+1+1) ) / 4;
                        B(i,j+1) = I(i,j+1);
                   
                end
            end
           
        
    case 'bggr'                             % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       
            for i = 3 : 2 : wid+1
                for j = 2 : 2 : len
                   
                        R(i,j) = ( I(i,j-1) + I(i,j+1) ) / 2;
                        G(i,j) = I(i,j);
                        B(i,j) = ( I(i-1,j) + I(i+1,j) ) / 2;
                       
                        R(i,j+1) = I(i,j+1);
                        G(i,j+1) = ( I(i-1,j+1) + I(i+1,j+1) + I(i,j-1+1) + I(i,j+1+1) ) / 4;
                        B(i,j+1) = ( I(i-1,j-1+1) + I(i+1,j+1+1) + I(i-1,j+1+1) + I(i+1,j-1+1) ) / 4;
                   
                end
            end
       
            for i = 2 : 2 : wid+1
                for j = 2 : 2 : len
               
                        R(i,j) = ( I(i-1,j-1) + I(i+1,j+1) + I(i-1,j+1) + I(i+1,j-1) ) / 4;
                        G(i,j) = ( I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) ) / 4;
                        B(i,j) = I(i,j);
                       
                        R(i,j+1) = ( I(i-1,j+1) + I(i+1,j+1) ) / 2;
                        G(i,j+1) = I(i,j+1);
                        B(i,j+1) = ( I(i,j-1+1) + I(i,j+1+1) ) / 2;
                   
                end
            end
    end
    
    RGB = zeros(wid,len,3);
      
    RGB(:,:,1) = R(2:wid+1,2:len+1);
    RGB(:,:,2) = G(2:wid+1,2:len+1);
    RGB(:,:,3) = B(2:wid+1,2:len+1);
 
    %RGB = uint8(RGB);
   
end