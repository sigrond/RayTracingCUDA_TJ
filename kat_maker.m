function TF_frame = kat_maker(handles,lambda)
 % iloœæ kolumn ramki
   rr = (0 :319);
   qq = (0 :239);
  % zakres pixeli efektywnych z zerem prawie w œrodku
  % wektor k¹tów wyznaczanych z apertury diafragmy i unormowanego po³o¿enia
  % na wyprostowanym obrazie
  TF_frame = zeros(640, 480, 3);
    fitt=@(x,k) ((k(1)*x^4+k(2)*x^3+k(3)*x^2+k(4)*x+k(5))/(x^3+k(6)*x^2+k(7)*x+k(8)));
    wb = waitbar(0,'');
  
   
    
  for ii = 1:320
      waitbar(ii/640,wb,[' Licze ...'  ' ' num2str(ii)  ] );
        for jj = 1:240
%             TF_frame(ii,jj,1) = atan( tan( handles.alpha_0_max )*( rr(ii)/handles.params(3) ) * fitt( sqrt ( rr(ii)^2 + qq(jj)^2 )...
%                                      /handles.params(3), handles.koef)/( sqrt ( rr(ii)^2 + qq(jj)^2 )/handles.params(3) ) )+ pi/2;
            TF_frame(ii+319,jj+239,1) = fitt( rr(ii)/(handles.params(3)), handles.koef) + pi/2;
            TF_frame(321-ii,241-jj,1) = pi/2 - fitt( rr(ii)/(handles.params(3)), handles.koef) ;
            
            TF_frame(ii+319,241-jj,1) = fitt( rr(ii)/(handles.params(3)), handles.koef) + pi/2;
            TF_frame(321-ii,jj+239,1) = pi/2 - fitt( rr(ii)/(handles.params(3)), handles.koef)  ;
            
             if isnan(TF_frame(ii,jj,1))
                TF_frame(ii,jj,1) = pi/2;
             end;
%             TF_frame(ii,jj,2) =  atan( tan( handles.alpha_0_max )*( qq(jj)/handles.params(3) )* fitt( sqrt ( rr(ii)^2+qq(jj)^2 )...
%                                       /handles.params(3), handles.koef)/( sqrt ( rr(ii)^2 + qq(jj)^2 )/handles.params(3) ) ) ;
             TF_frame(ii+320,jj+240,2) = fitt( qq(jj)/(handles.params(3)), handles.koef) ;
            TF_frame(321-ii,241-jj,2) = fitt( qq(jj)/(handles.params(3)), handles.koef) ;
            
            TF_frame(ii+320,241-jj,2) = fitt( qq(jj)/(handles.params(3)), handles.koef) ;
            TF_frame(321-ii,jj+240,2) = fitt( qq(jj)/(handles.params(3)), handles.koef) ;
               if isnan(TF_frame(ii,jj,2))
                TF_frame(ii,jj,2) = 0;
               end;
            alpha_0 =  fitt( sqrt ( rr(ii)^2 + qq(jj)^2 )/handles.params(3), handles.koef) ;
        
            temp = ( running_radius( alpha_0,( handles.params(3)*9.36e-6 ),handles.Diafragma, lambda ) )^2;
          
            TF_frame(ii,jj,3) = temp;
            
        end
  end
   TF_frame(320,240,1) = fitt( rr(1)/(handles.params(3)), handles.koef) + pi/2;
%   TF(:,:,1) = TF_frame(:,:,1)';
%   TF(:,:,2) = TF_frame(:,:,2)';
%   TF(:,:,3) = TF_frame(:,:,3)';
  close(wb);