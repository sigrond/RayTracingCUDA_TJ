%% Intensity correction for angles


ThetaR2=Angular_QE_Asym2Sig(ThetaR);
ThetaG2=Angular_QE_Asym2Sig(ThetaG);
ThetaB2=Angular_QE_Asym2Sig(ThetaB);
    IRcorrection=meshgrid(ThetaR2,1:length(I_R));
    IGcorrection=meshgrid(ThetaG2,1:length(I_G));
    IBcorrection=meshgrid(ThetaB2,1:length(I_B));
    
    %%new Intensities for radius after correction
    
        I_R2=I_R./IRcorrection;
        I_G2=I_G./IGcorrection;
        I_B2=I_B./IBcorrection;
        
        % new intesnities
        I_R=I_R2;
        I_G=I_G2;
        I_B=I_B2;
        
        %%
%          savePath = [ handles.dir handles.f(1,1:end-4) '27.08.2016v2.mat'];
                            
         filename=['E:\Justice_Archer\Pure DEG\PureDEG\Newdata\27.08.2016v19', num2str(savepath)];
               save(filename,'ThetaR', 'ThetaG', 'ThetaB', 'I_R', 'I_G', 'I_B', 'Save');
          
     
        
        
%         save('NewData.mat', 'ThetaR', 'ThetaG', 'ThetaB', 'I_R', 'I_G', 'I_B', 'Save')
%         
%         clear;
%         load NewData.mat
        
   